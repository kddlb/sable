//! Time-varying stochastic sibilant resynthesis (spec §8).
//!
//! For each confident, sufficiently long sibilant the source's smoothed short-time
//! power spectrum (and, for stereo, cross-spectrum) is captured. The already-allocated
//! output interval is then filled with *new* seeded Gaussian noise coloured by the
//! spectrum sampled along the inverse time map, overlap-added with a normalization for
//! independent stochastic grains (`sqrt(Σ w²)`), and composited with equal-power fades.
//! Voiced sibilants keep the WSOLA low band and only replace the high band.

use crate::audio::AudioView;
use crate::features::Features;
use crate::fft::{Complex, RealFft};
use crate::filters::BandSplit;
use crate::options::Options;
use crate::pipeline::Plan;
use crate::report::{AnalysisReport, OverlayReport, Warning};
use crate::rng::GaussianStream;
use crate::segments::EventKind;
use crate::stereo;
use crate::windows::{
    hann_offset, ms_to_samples, pow2_near_ms, raised_cosine_ramp, sqrt_hann_periodic,
};
use crate::wsola::reflect;

#[derive(Clone, Debug)]
pub struct SibilantConfig {
    pub fft_ms: f64,
    pub hop_ms: f64,
    /// Temporal smoothing half-width (frames) of the power estimates.
    pub time_smooth: usize,
    /// Frequency smoothing neighbourhood (fraction of an octave, full width).
    pub freq_octaves: f64,
    pub guard_ms: f64,
    pub fade_ms: f64,
    pub min_interior_ms: f64,
    pub min_confidence: f64,
    /// Bound of the event-level level calibration (dB).
    pub calib_bound_db: f64,
    /// Coherence instability above which stereo resynthesis falls back to WSOLA.
    pub max_coherence_instability: f64,
}

impl Default for SibilantConfig {
    fn default() -> Self {
        Self {
            fft_ms: 25.0,
            hop_ms: 5.0,
            time_smooth: 2,
            freq_octaves: 1.0 / 6.0,
            guard_ms: 10.0,
            fade_ms: 15.0,
            min_interior_ms: 40.0,
            min_confidence: 0.5,
            calib_bound_db: 3.0,
            max_coherence_instability: 0.4,
        }
    }
}

/// Result of the sibilant pass.
pub struct SibilantResult {
    /// Synthetic (noise) amplitude weight `sin θ` per output frame.
    pub mask: Vec<f32>,
    /// Raw synthesized noise per channel (zero outside resynthesized events).
    pub noise: Vec<Vec<f32>>,
}

/// Per-channel STFT power spectra plus an optional stereo cross-spectrum at one instant.
pub type SpectraAt = (Vec<Vec<f64>>, Option<Vec<(f64, f64)>>);

/// Smoothed short-time spectra of one source event.
pub struct SpectralTemplate {
    pub n_fft: usize,
    pub hop: usize,
    /// Source sample of frame 0's centre.
    pub t0: f64,
    /// `p[ch][frame][bin]`: power of the sqrt-Hann-windowed STFT.
    pub p: Vec<Vec<Vec<f64>>>,
    /// Stereo only: smoothed `C_RL = E[X_R conj X_L]` per frame/bin.
    pub c: Option<Vec<Vec<(f64, f64)>>>,
    /// Power-weighted mean deviation of |ρ| over time (0 = perfectly stable).
    pub coherence_instability: f64,
    /// Estimated inter-channel lag in samples (R relative to L), compensated before the
    /// cross-spectrum is smoothed in frequency.
    pub interchannel_lag: f64,
}

/// GCC-PHAT lag (samples, R relative to L) from per-frame cross-spectra `X_R conj X_L`,
/// searched within ±`max_lag` with parabolic refinement.
fn gcc_phat_lag(frames: &[Vec<C2>], n_fft: usize, max_lag: usize) -> f64 {
    let bins = n_fft / 2 + 1;
    let mut g = vec![Complex::new(0.0, 0.0); bins];
    for fr in frames {
        for k in 1..bins - 1 {
            let (re, im) = (fr[k].0, fr[k].1);
            let m = (re * re + im * im).sqrt();
            if m > 1e-30 {
                g[k].re += (re / m) as f32;
                g[k].im += (im / m) as f32;
            }
        }
    }
    let mut fft = RealFft::new(n_fft);
    let r = fft.inverse(&g).to_vec();
    let at = |d: i64| r[d.rem_euclid(n_fft as i64) as usize] as f64;
    let max = max_lag.min(n_fft / 2 - 1) as i64;
    let mut best = 0i64;
    for d in -max..=max {
        if at(d) > at(best) {
            best = d;
        }
    }
    let (a, b, c) = (at(best - 1), at(best), at(best + 1));
    let den = a - 2.0 * b + c;
    let frac = if den.abs() > 1e-12 {
        (0.5 * (a - c) / den).clamp(-0.5, 0.5)
    } else {
        0.0
    };
    best as f64 + frac
}

/// Multiplies every bin of every frame by `e^{j·sign·2πkτ/N}`.
fn rotate(frames: &mut [Vec<C2>], n_fft: usize, lag: f64, sign: f64) {
    for fr in frames {
        for (k, z) in fr.iter_mut().enumerate() {
            let ph = sign * std::f64::consts::TAU * k as f64 * lag / n_fft as f64;
            let (s, c) = ph.sin_cos();
            *z = C2(z.0 * c - z.1 * s, z.0 * s + z.1 * c);
        }
    }
}

/// Frequency neighbourhood half-widths for fractional-octave smoothing.
fn neighbourhoods(bins: usize, octaves: f64) -> Vec<usize> {
    let f = 2f64.powf(octaves / 2.0) - 1.0;
    (0..bins)
        .map(|k| ((k as f64 * f).round() as usize).max(1))
        .collect()
}

fn smooth_time<T: Copy + Default + std::ops::AddAssign + std::ops::Div<f64, Output = T>>(
    frames: &[Vec<T>],
    half: usize,
) -> Vec<Vec<T>> {
    let m = frames.len();
    let bins = frames.first().map_or(0, Vec::len);
    (0..m)
        .map(|i| {
            let a = i.saturating_sub(half);
            let b = (i + half + 1).min(m);
            (0..bins)
                .map(|k| {
                    let mut acc = T::default();
                    for f in &frames[a..b] {
                        acc += f[k];
                    }
                    acc / (b - a) as f64
                })
                .collect()
        })
        .collect()
}

#[derive(Clone, Copy, Default)]
struct C2(f64, f64);
impl std::ops::AddAssign for C2 {
    fn add_assign(&mut self, o: Self) {
        self.0 += o.0;
        self.1 += o.1;
    }
}
impl std::ops::Div<f64> for C2 {
    type Output = C2;
    fn div(self, d: f64) -> C2 {
        C2(self.0 / d, self.1 / d)
    }
}

/// Spike-clipped (≤ 4× local median) neighbourhood mean: removes isolated narrow
/// harmonic peaks without biasing smooth noise spectra.
fn smooth_freq_power(p: &[f64], nb: &[usize], floor: f64) -> Vec<f64> {
    let n = p.len();
    let mut tmp = Vec::new();
    let med: Vec<f64> = (0..n)
        .map(|k| {
            let (a, b) = (k.saturating_sub(nb[k]), (k + nb[k] + 1).min(n));
            tmp.clear();
            tmp.extend_from_slice(&p[a..b]);
            tmp.sort_by(f64::total_cmp);
            tmp[tmp.len() / 2]
        })
        .collect();
    (0..n)
        .map(|k| {
            let (a, b) = (k.saturating_sub(nb[k]), (k + nb[k] + 1).min(n));
            let s: f64 = (a..b).map(|j| p[j].min(4.0 * med[j].max(floor))).sum();
            (s / (b - a) as f64).max(floor)
        })
        .collect()
}

fn smooth_freq_cross(c: &[(f64, f64)], nb: &[usize]) -> Vec<(f64, f64)> {
    let n = c.len();
    (0..n)
        .map(|k| {
            let (a, b) = (k.saturating_sub(nb[k]), (k + nb[k] + 1).min(n));
            let (mut r, mut i) = (0.0, 0.0);
            for v in &c[a..b] {
                r += v.0;
                i += v.1;
            }
            (r / (b - a) as f64, i / (b - a) as f64)
        })
        .collect()
}

impl SpectralTemplate {
    /// Captures smoothed spectra of `channels` around source interval `[s0, s1)`.
    pub fn analyze(
        channels: &[&[f32]],
        s0: usize,
        s1: usize,
        fs: u32,
        cfg: &SibilantConfig,
    ) -> Self {
        let n_fft = pow2_near_ms(cfg.fft_ms, fs);
        let hop = ms_to_samples(cfg.hop_ms, fs);
        let bins = n_fft / 2 + 1;
        let win = sqrt_hann_periodic(n_fft);
        let mut fft = RealFft::new(n_fft);
        let n_in = channels[0].len();
        let ctx = hop * (cfg.time_smooth + 1);
        let start = s0 as i64 - ctx as i64;
        let end = (s1 + ctx) as i64;
        let m = (((end - start) as usize) / hop) + 1;
        let mut raw_p = vec![vec![vec![0f64; bins]; m]; channels.len()];
        let mut raw_c = vec![vec![C2::default(); bins]; m];
        let mut buf = vec![0f32; n_fft];
        let mut specs: Vec<Vec<Complex>> = vec![Vec::new(); channels.len()];
        for f in 0..m {
            let c = start + (f * hop) as i64;
            for (ch, x) in channels.iter().enumerate() {
                for (k, v) in buf.iter_mut().enumerate() {
                    *v = x[reflect(c - (n_fft / 2) as i64 + k as i64, n_in)];
                }
                let s = fft.forward_windowed(&buf, &win);
                specs[ch] = s.to_vec();
                for k in 0..bins {
                    raw_p[ch][f][k] = s[k].norm_sqr() as f64;
                }
            }
            if channels.len() == 2 {
                for k in 0..bins {
                    let (l, r) = (specs[0][k], specs[1][k]);
                    // X_R * conj(X_L)
                    raw_c[f][k] = C2(
                        (r.re * l.re + r.im * l.im) as f64,
                        (r.im * l.re - r.re * l.im) as f64,
                    );
                }
            }
        }
        let peak = raw_p
            .iter()
            .flatten()
            .flatten()
            .fold(0f64, |a, &v| a.max(v));
        let floor = (peak * 1e-10).max(1e-30);
        let nb = neighbourhoods(bins, cfg.freq_octaves);
        let p: Vec<Vec<Vec<f64>>> = raw_p
            .iter()
            .map(|chp| {
                smooth_time(chp, cfg.time_smooth)
                    .iter()
                    .map(|fr| smooth_freq_power(fr, &nb, floor))
                    .collect()
            })
            .collect();
        let mut lag = 0.0;
        let (c, instability) = if channels.len() == 2 {
            // A delay between channels (spaced microphones) turns the cross-spectrum phase
            // into a ramp that fractional-octave averaging would cancel. Remove the
            // dominant lag first, smooth, then restore it.
            lag = gcc_phat_lag(&raw_c, n_fft, ms_to_samples(2.0, fs));
            let mut aligned = raw_c.clone();
            rotate(&mut aligned, n_fft, lag, 1.0);
            let ct = smooth_time(&aligned, cfg.time_smooth);
            let mut c: Vec<Vec<C2>> = ct
                .iter()
                .map(|fr| {
                    let v: Vec<(f64, f64)> = fr.iter().map(|z| (z.0, z.1)).collect();
                    smooth_freq_cross(&v, &nb)
                        .into_iter()
                        .map(|(a, b)| C2(a, b))
                        .collect()
                })
                .collect();
            rotate(&mut c, n_fft, lag, -1.0);
            let c: Vec<Vec<(f64, f64)>> = c
                .into_iter()
                .map(|fr| fr.into_iter().map(|z| (z.0, z.1)).collect())
                .collect();
            // Instability: power-weighted mean |ρ_t - mean_t ρ| per bin.
            let mut num = 0.0;
            let mut den = 0.0;
            for k in 0..bins {
                let rhos: Vec<(f64, f64, f64)> = (0..m)
                    .map(|f| {
                        let (pl, pr) = (p[0][f][k], p[1][f][k]);
                        let r = stereo::rho(c[f][k].0, c[f][k].1, pl, pr, floor * 10.0);
                        (r.0, r.1, (pl * pr).sqrt())
                    })
                    .collect();
                let (mr, mi) = rhos.iter().fold((0.0, 0.0), |a, r| (a.0 + r.0, a.1 + r.1));
                let (mr, mi) = (mr / m as f64, mi / m as f64);
                for r in &rhos {
                    num += r.2 * ((r.0 - mr).powi(2) + (r.1 - mi).powi(2)).sqrt();
                    den += r.2;
                }
            }
            (Some(c), if den > 0.0 { num / den } else { 0.0 })
        } else {
            (None, 0.0)
        };
        Self {
            n_fft,
            hop,
            t0: start as f64,
            p,
            c,
            coherence_instability: instability,
            interchannel_lag: lag,
        }
    }

    /// Linearly interpolated spectra at source time `u`.
    pub fn at(&self, u: f64) -> SpectraAt {
        let m = self.p[0].len();
        let fi = ((u - self.t0) / self.hop as f64).clamp(0.0, (m - 1) as f64);
        let i0 = fi.floor() as usize;
        let i1 = (i0 + 1).min(m - 1);
        let a = fi - i0 as f64;
        let p = self
            .p
            .iter()
            .map(|ch| {
                ch[i0]
                    .iter()
                    .zip(&ch[i1])
                    .map(|(x, y)| x * (1.0 - a) + y * a)
                    .collect()
            })
            .collect();
        let c = self.c.as_ref().map(|c| {
            c[i0]
                .iter()
                .zip(&c[i1])
                .map(|(x, y)| (x.0 * (1.0 - a) + y.0 * a, x.1 * (1.0 - a) + y.1 * a))
                .collect()
        });
        (p, c)
    }
}

/// Overlap-add synthesizer of independent coloured Gaussian grains.
pub struct NoiseSynth {
    n_fft: usize,
    hop: usize,
    /// Σ w_a² of the sqrt-Hann analysis window (converts STFT power to PSD).
    w2: f64,
    win: Vec<f32>,
    fft: RealFft,
}

impl NoiseSynth {
    pub fn new(n_fft: usize) -> Self {
        let wa = sqrt_hann_periodic(n_fft);
        let w2 = wa.iter().map(|&v| (v as f64) * (v as f64)).sum();
        Self {
            n_fft,
            hop: n_fft / 4,
            w2,
            win: hann_offset(n_fft),
            fft: RealFft::new(n_fft),
        }
    }

    /// Renders `len` frames. `spec_at(y)` gives, for output frame position `y`, the
    /// per-channel STFT power (sqrt-Hann analysis) and optional stereo cross-spectrum.
    pub fn render(
        &mut self,
        len: usize,
        nch: usize,
        mut spec_at: impl FnMut(f64) -> SpectraAt,
        stream: &mut GaussianStream,
    ) -> Vec<Vec<f32>> {
        let n = self.n_fft;
        let bins = n / 2 + 1;
        let mut acc = vec![vec![0f64; len]; nch];
        let mut wsq = vec![0f64; len];
        let mut spec = vec![Complex::new(0.0, 0.0); bins];
        let mut st = -(n as i64) + self.hop as i64;
        let floor = 1e-30;
        while st < len as i64 {
            let center = (st + (n / 2) as i64).clamp(0, len.max(1) as i64 - 1) as f64;
            let (p, c) = spec_at(center);
            // Excitations for this grain (Z1 for all, Z2 for the right channel).
            let z1: Vec<(f64, f64)> = (0..bins).map(|_| stream.complex()).collect();
            let z2: Vec<(f64, f64)> = if nch == 2 {
                (0..bins).map(|_| stream.complex()).collect()
            } else {
                Vec::new()
            };
            for ch in 0..nch {
                for k in 0..bins {
                    let s = p[ch][k] / self.w2; // PSD in variance-per-sample units
                    let g = (s * n as f64).sqrt();
                    let z = match (&c, ch) {
                        (_, 0) => z1[k],
                        (None, _) => z2[k],
                        (Some(c), _) => {
                            let r = stereo::rho(c[k].0, c[k].1, p[0][k], p[1][k], floor);
                            stereo::correlate(r, z1[k], z2[k])
                        }
                    };
                    let (re, im) = if k == 0 || k == bins - 1 {
                        // DC/Nyquist are real: z.0 has variance 1/2, so scale to unit variance.
                        (z.0 * std::f64::consts::SQRT_2, 0.0)
                    } else {
                        (z.0, z.1)
                    };
                    spec[k] = Complex::new((g * re) as f32, (g * im) as f32);
                }
                let y = self.fft.inverse(&spec);
                for (i, (&v, &w)) in y.iter().zip(&self.win).enumerate() {
                    let t = st + i as i64;
                    if t >= 0 && (t as usize) < len {
                        acc[ch][t as usize] += w as f64 * v as f64;
                    }
                }
            }
            for (i, &w) in self.win.iter().enumerate() {
                let t = st + i as i64;
                if t >= 0 && (t as usize) < len {
                    wsq[t as usize] += (w as f64) * (w as f64);
                }
            }
            st += self.hop as i64;
        }
        acc.into_iter()
            .map(|a| {
                a.iter()
                    .zip(&wsq)
                    .map(|(&v, &q)| if q > 0.0 { (v / q.sqrt()) as f32 } else { 0.0 })
                    .collect()
            })
            .collect()
    }
}

fn ms_power(x: &[f32], a: usize, b: usize) -> f64 {
    if b <= a {
        return 0.0;
    }
    x[a..b]
        .iter()
        .map(|&v| (v as f64) * (v as f64))
        .sum::<f64>()
        / (b - a) as f64
}

pub fn apply(
    input: &AudioView<'_>,
    features: &Features,
    plan: &Plan,
    opts: &Options,
    plosive_mask: &[f32],
    out: &mut [Vec<f32>],
    report: &mut AnalysisReport,
) -> SibilantResult {
    let cfg = SibilantConfig::default();
    let fs = input.sample_rate;
    let n_out = plan.map.n_out as usize;
    let nch = input.channels.len();
    let mut mask = vec![0f32; n_out];
    let mut noise_layer = vec![vec![0f32; n_out]; nch];
    let guard_in = ms_to_samples(cfg.guard_ms, fs) as u64;
    let min_interior = ms_to_samples(cfg.min_interior_ms, fs) as u64;
    let split = BandSplit::new(fs);
    let silence_ms = 10f64.powf(features.silence_thr_db as f64 / 10.0);

    for (i, seg) in plan.segments.iter().enumerate() {
        if !seg.kind.is_sibilant() {
            continue;
        }
        let o0 = plan.out_start(i) as usize;
        let o1 = plan.out_start(i + 1) as usize;
        let d = o1 - o0;
        let skip = |reason: String, report: &mut AnalysisReport| {
            report.sibilant_overlays.push(OverlayReport {
                segment_id: seg.id,
                applied: false,
                out_start: o0 as u64,
                out_end: o1 as u64,
                detail: reason.clone(),
            });
            report.warnings.push(Warning::SpecialRendererDisabled {
                segment_id: seg.id,
                reason,
            });
        };
        if !opts.enable_sibilant {
            skip("sibilant resynthesis disabled by options".into(), report);
            continue;
        }
        if seg.confidence < cfg.min_confidence {
            skip(
                format!(
                    "confidence {:.2} below {:.2}",
                    seg.confidence, cfg.min_confidence
                ),
                report,
            );
            continue;
        }
        if seg.len() < min_interior + 2 * guard_in {
            skip(
                format!(
                    "interior {} ms after 2x{} ms guards is shorter than {} ms",
                    (seg.len().saturating_sub(2 * guard_in)) * 1000 / fs as u64,
                    cfg.guard_ms,
                    cfg.min_interior_ms
                ),
                report,
            );
            continue;
        }
        // Output fades: guard then raised-cosine fade, both inside [o0, o1).
        let guard = (ms_to_samples(cfg.guard_ms, fs)).min(d / 5);
        let fade = ms_to_samples(cfg.fade_ms, fs).min((d - 2 * guard) / 3);
        if fade == 0 {
            skip(
                format!("allocated output of {d} frames leaves no room for fades"),
                report,
            );
            continue;
        }
        if plosive_mask[o0 + guard..o1 - guard]
            .iter()
            .any(|&w| w > 0.0)
        {
            skip(
                "overlaps a protected plosive burst (plosive wins)".into(),
                report,
            );
            continue;
        }
        let (s0, s1) = (seg.start as usize, seg.end as usize);
        let src_ms: Vec<f64> = (0..nch)
            .map(|ch| {
                ms_power(
                    input.channels[ch],
                    s0 + guard_in as usize,
                    s1 - guard_in as usize,
                )
            })
            .collect();
        let mono_ms = ms_power(
            &features.mono,
            s0 + guard_in as usize,
            s1 - guard_in as usize,
        );
        if mono_ms <= silence_ms {
            skip("event energy below the silence threshold".into(), report);
            continue;
        }
        let template = SpectralTemplate::analyze(input.channels, s0, s1, fs, &cfg);
        if nch == 2 && template.coherence_instability > cfg.max_coherence_instability {
            skip(
                format!(
                    "stereo coherence unstable ({:.2} > {:.2}); keeping WSOLA",
                    template.coherence_instability, cfg.max_coherence_instability
                ),
                report,
            );
            continue;
        }
        let mut synth = NoiseSynth::new(template.n_fft);
        let mut stream = GaussianStream::new(opts.seed, seg.id as u64);
        let map = &plan.map;
        let mut noise = synth.render(
            d,
            nch,
            |y| {
                let u = map
                    .inverse(o0 as f64 + y)
                    .clamp(seg.start as f64, seg.end as f64);
                template.at(u)
            },
            &mut stream,
        );
        // Event-level bounded calibration against the source interior energy.
        let bound = 10f64.powf(cfg.calib_bound_db / 20.0);
        let mut gains = Vec::with_capacity(nch);
        for ch in 0..nch {
            let syn = ms_power(&noise[ch], guard, d - guard);
            let g = if syn > 0.0 {
                (src_ms[ch] / syn).sqrt().clamp(1.0 / bound, bound)
            } else {
                1.0
            };
            gains.push(g);
            noise[ch].iter_mut().for_each(|v| *v *= g as f32);
        }
        // Equal-power fade weights.
        let theta: Vec<f64> = (0..d)
            .map(|t| {
                let ramp = if t < guard || t >= d - guard {
                    0.0
                } else if t < guard + fade {
                    raised_cosine_ramp(t - guard, fade) as f64
                } else if t >= d - guard - fade {
                    raised_cosine_ramp(d - guard - 1 - t, fade) as f64
                } else {
                    1.0
                };
                ramp * std::f64::consts::FRAC_PI_2
            })
            .collect();
        let voiced = seg.kind == EventKind::SibilantVoiced;
        for ch in 0..nch {
            if voiced {
                let base_lp = split.lowpass(&out[ch], o0, o1);
                let noise_lp = split.lowpass(&noise[ch], 0, d);
                for t in 0..d {
                    let (c, s) = (theta[t].cos(), theta[t].sin());
                    let b = out[ch][o0 + t] as f64;
                    let blp = base_lp[t] as f64;
                    let nhp = noise[ch][t] as f64 - noise_lp[t] as f64;
                    out[ch][o0 + t] = (blp + c * (b - blp) + s * nhp) as f32;
                    noise_layer[ch][o0 + t] = nhp as f32;
                }
            } else {
                for t in 0..d {
                    let (c, s) = (theta[t].cos(), theta[t].sin());
                    let b = out[ch][o0 + t] as f64;
                    out[ch][o0 + t] = (c * b + s * noise[ch][t] as f64) as f32;
                    noise_layer[ch][o0 + t] = noise[ch][t];
                }
            }
        }
        for t in 0..d {
            mask[o0 + t] = theta[t].sin() as f32;
        }
        report.sibilant_overlays.push(OverlayReport {
            segment_id: seg.id,
            applied: true,
            out_start: o0 as u64,
            out_end: o1 as u64,
            detail: format!(
                "{} resynthesis: {} frames, guard {} fade {}, n_fft {}, level calibration {} dB{}",
                if voiced {
                    "high-band (voiced)"
                } else {
                    "full-band (unvoiced)"
                },
                d,
                guard,
                fade,
                template.n_fft,
                gains
                    .iter()
                    .map(|g| format!("{:+.2}", 20.0 * g.log10()))
                    .collect::<Vec<_>>()
                    .join("/"),
                if nch == 2 {
                    format!(
                        ", coherence instability {:.2}, inter-channel lag {:.1} samples",
                        template.coherence_instability, template.interchannel_lag
                    )
                } else {
                    String::new()
                }
            ),
        });
    }
    SibilantResult {
        mask,
        noise: noise_layer,
    }
}

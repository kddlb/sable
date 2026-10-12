//! Deterministic synthetic speech-like fixtures for tests, examples and quality runs.
//!
//! Nothing here is a speech model; each generator produces a controlled acoustic event
//! (harmonic vowel, stop release, shaped fricative noise) with known ground truth.

use crate::audio::AudioBuffer;
use crate::fft::{Complex, RealFft};
use crate::windows::ms_to_samples;

/// Small deterministic generator (SplitMix64) with Box–Muller normals.
#[derive(Clone, Debug)]
pub struct FixtureRng(u64, Option<f64>);

impl FixtureRng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15, None)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    pub fn uniform(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.1.take() {
            return v;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let t = std::f64::consts::TAU * u2;
        self.1 = Some(r * t.sin());
        r * t.cos()
    }
}

/// Ground truth for a fixture.
#[derive(Clone, Debug, Default)]
pub struct Truth {
    /// Burst onset sample indices.
    pub bursts: Vec<usize>,
    /// Sibilant regions `(start, end, voiced)`.
    pub sibilants: Vec<(usize, usize, bool)>,
    /// Steady harmonic regions `(start, end, f0_hz)`.
    pub vowels: Vec<(usize, usize, f64)>,
    /// Vowel onsets that must *not* be detected as bursts.
    pub soft_onsets: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: String,
    pub audio: AudioBuffer,
    pub truth: Truth,
}

/// Formant-shaped harmonic vowel with gliding formants and raised-cosine edges.
pub fn vowel(
    fs: u32,
    n: usize,
    f0: f64,
    formants_a: [f64; 3],
    formants_b: [f64; 3],
    amp: f64,
) -> Vec<f32> {
    let nyq = fs as f64 / 2.0;
    let edge = ms_to_samples(15.0, fs).min(n / 2);
    let bw = [90.0, 120.0, 160.0];
    let gain = [1.0, 0.5, 0.25];
    let nh = ((nyq * 0.9) / f0).floor() as usize;
    // Pre-compute per-harmonic phase increments; amplitudes vary with formants.
    let mut phase = vec![0f64; nh];
    let mut out = vec![0f32; n];
    let mut peak = 0f64;
    let mut tmp = vec![0f64; n];
    for (i, t) in tmp.iter_mut().enumerate() {
        let a = i as f64 / n.max(1) as f64;
        let fm: Vec<f64> = (0..3)
            .map(|k| formants_a[k] + (formants_b[k] - formants_a[k]) * a)
            .collect();
        let mut s = 0.0;
        for (h, ph) in phase.iter_mut().enumerate() {
            let f = f0 * (h + 1) as f64;
            let mut g = 0.02;
            for k in 0..3 {
                let d = (f - fm[k]) / bw[k];
                g += gain[k] / (1.0 + d * d);
            }
            // Gentle glottal tilt.
            g /= 1.0 + f / 1500.0;
            s += g * ph.sin();
            *ph += std::f64::consts::TAU * f / fs as f64;
            if *ph > std::f64::consts::TAU {
                *ph -= std::f64::consts::TAU;
            }
        }
        *t = s;
        peak = peak.max(s.abs());
    }
    for (i, o) in out.iter_mut().enumerate() {
        let e = if i < edge {
            0.5 - 0.5 * (std::f64::consts::PI * i as f64 / edge as f64).cos()
        } else if i + edge >= n {
            0.5 - 0.5 * (std::f64::consts::PI * (n - 1 - i) as f64 / edge as f64).cos()
        } else {
            1.0
        };
        *o = (tmp[i] / peak.max(1e-9) * amp * e) as f32;
    }
    out
}

/// Gaussian noise shaped in the frequency domain by `shape(f_hz)` (amplitude), unit RMS.
pub fn shaped_noise(fs: u32, n: usize, seed: u64, shape: impl Fn(f64) -> f64) -> Vec<f32> {
    if n == 0 {
        return Vec::new();
    }
    let m = n.next_power_of_two().max(64);
    let mut rng = FixtureRng::new(seed);
    let mut fft = RealFft::new(m);
    let mut spec = vec![Complex::new(0.0, 0.0); m / 2 + 1];
    for (k, c) in spec.iter_mut().enumerate() {
        let f = k as f64 * fs as f64 / m as f64;
        let a = shape(f);
        *c = Complex::new((rng.normal() * a) as f32, (rng.normal() * a) as f32);
    }
    let y = fft.inverse(&spec).to_vec();
    let rms = (y[..n].iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / n as f64).sqrt();
    y[..n]
        .iter()
        .map(|&v| (v as f64 / rms.max(1e-30)) as f32)
        .collect()
}

/// Smooth band shape with raised-cosine skirts of `skirt` Hz.
pub fn band(f: f64, lo: f64, hi: f64, skirt: f64) -> f64 {
    if f < lo - skirt || f > hi + skirt {
        0.0
    } else if f < lo {
        0.5 + 0.5 * (std::f64::consts::PI * (lo - f) / skirt).cos()
    } else if f > hi {
        0.5 + 0.5 * (std::f64::consts::PI * (f - hi) / skirt).cos()
    } else {
        1.0
    }
}

fn ramp_env(n: usize, rise: usize, fall: usize) -> impl Fn(usize) -> f64 {
    move |i| {
        let a = if rise > 0 && i < rise {
            0.5 - 0.5 * (std::f64::consts::PI * i as f64 / rise as f64).cos()
        } else {
            1.0
        };
        let b = if fall > 0 && i + fall >= n {
            0.5 - 0.5 * (std::f64::consts::PI * (n - 1 - i) as f64 / fall as f64).cos()
        } else {
            1.0
        };
        a * b
    }
}

/// /s/-like noise (high band, 4–9 kHz limited by Nyquist).
pub fn s_noise(fs: u32, n: usize, seed: u64) -> Vec<f32> {
    let nyq = fs as f64 / 2.0;
    let hi = 9000f64.min(nyq * 0.92);
    let lo = 4000f64.min(nyq * 0.55);
    shaped_noise(fs, n, seed, |f| band(f, lo, hi, 600.0) + 0.02)
}

/// /ʃ/-like noise (lower concentration, ~1.8–4.5 kHz).
pub fn sh_noise(fs: u32, n: usize, seed: u64) -> Vec<f32> {
    let nyq = fs as f64 / 2.0;
    let hi = 4500f64.min(nyq * 0.9);
    let lo = 1800f64.min(nyq * 0.4);
    shaped_noise(fs, n, seed, |f| band(f, lo, hi, 400.0) + 0.02)
}

fn add_at(dst: &mut [f32], at: usize, src: &[f32], gain: f64, env: impl Fn(usize) -> f64) {
    for (i, &v) in src.iter().enumerate() {
        if let Some(d) = dst.get_mut(at + i) {
            *d += (v as f64 * gain * env(i)) as f32;
        }
    }
}

fn hiss(fs: u32, n: usize, seed: u64, level_db: f64) -> Vec<f32> {
    let g = 10f64.powf(level_db / 20.0);
    shaped_noise(fs, n, seed, |_| 1.0)
        .into_iter()
        .map(|v| (v as f64 * g) as f32)
        .collect()
}

/// Stop release burst: impulsive broadband noise with a ~1.5 ms decay.
pub fn burst(fs: u32, seed: u64) -> Vec<f32> {
    let n = ms_to_samples(6.0, fs);
    let nyq = fs as f64 / 2.0;
    let noise = shaped_noise(fs, n, seed, |f| 0.3 + band(f, 1500.0, nyq * 0.9, 500.0));
    let tau = 1.5e-3 * fs as f64;
    let att = ms_to_samples(0.2, fs) as f64;
    noise
        .iter()
        .enumerate()
        .map(|(i, &v)| {
            let a = (i as f64 / att).min(1.0);
            (v as f64 * a * (-(i as f64) / tau).exp()) as f32
        })
        .collect()
}

const A_FORM: [f64; 3] = [750.0, 1200.0, 2500.0];
const I_FORM: [f64; 3] = [300.0, 2200.0, 3000.0];
const E_FORM: [f64; 3] = [500.0, 1800.0, 2600.0];

/// Silence, vowel, quiet closure, burst, aspiration, vowel, silence.
pub fn stop_fixture(fs: u32, seed: u64) -> Fixture {
    let ms = |m: f64| ms_to_samples(m, fs);
    let parts = [
        ms(100.0),
        ms(180.0),
        ms(70.0),
        ms(6.0),
        ms(35.0),
        ms(250.0),
        ms(100.0),
    ];
    let n: usize = parts.iter().sum();
    let mut x = hiss(fs, n, seed ^ 1, -72.0);
    let mut pos = parts[0];
    let v1 = vowel(fs, parts[1], 120.0, A_FORM, E_FORM, 0.3);
    add_at(&mut x, pos, &v1, 1.0, |_| 1.0);
    let v1_start = pos;
    pos += parts[1] + parts[2];
    let onset = pos;
    let b = burst(fs, seed ^ 2);
    add_at(&mut x, pos, &b, 0.25, |_| 1.0);
    pos += parts[3];
    let asp = shaped_noise(fs, parts[4], seed ^ 3, |f| band(f, 1000.0, 6000.0, 800.0));
    let na = parts[4];
    add_at(&mut x, pos - ms(1.0), &asp, 0.05, |i| {
        (-(i as f64) / (na as f64 * 0.5)).exp()
    });
    pos += parts[4];
    let v2 = vowel(fs, parts[5], 120.0, E_FORM, A_FORM, 0.3);
    add_at(&mut x, pos, &v2, 1.0, |_| 1.0);
    let v2_start = pos;
    Fixture {
        name: "stop".into(),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![x],
        },
        truth: Truth {
            bursts: vec![onset],
            vowels: vec![
                (v1_start + ms(30.0), v1_start + parts[1] - ms(30.0), 120.0),
                (v2_start + ms(30.0), v2_start + parts[5] - ms(30.0), 120.0),
            ],
            soft_onsets: vec![v1_start],
            ..Truth::default()
        },
    }
}

/// One second of harmonic vowel at `f0` with gliding formants.
pub fn vowel_fixture(fs: u32, f0: f64) -> Fixture {
    let n = fs as usize;
    let x = vowel(fs, n, f0, A_FORM, I_FORM, 0.4);
    Fixture {
        name: format!("vowel_{f0}hz"),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![x],
        },
        truth: Truth {
            vowels: vec![(n / 10, n - n / 10, f0)],
            soft_onsets: vec![0],
            ..Truth::default()
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fricative {
    S,
    Sh,
    Z,
}

/// Silence, vowel, fricative (≈220 ms), vowel, silence.
pub fn fricative_fixture(fs: u32, kind: Fricative, seed: u64) -> Fixture {
    let ms = |m: f64| ms_to_samples(m, fs);
    let parts = [ms(80.0), ms(150.0), ms(220.0), ms(150.0), ms(80.0)];
    let n: usize = parts.iter().sum();
    let mut x = hiss(fs, n, seed ^ 11, -72.0);
    let mut pos = parts[0];
    add_at(
        &mut x,
        pos,
        &vowel(fs, parts[1], 130.0, A_FORM, E_FORM, 0.3),
        1.0,
        |_| 1.0,
    );
    pos += parts[1];
    let nf = parts[2];
    let ramp = ms(25.0);
    let env = ramp_env(nf + 2 * ramp, ramp, ramp);
    let start = pos - ramp;
    let len = nf + 2 * ramp;
    let (noise, gain) = match kind {
        Fricative::S => (s_noise(fs, len, seed ^ 12), 0.08),
        Fricative::Sh => (sh_noise(fs, len, seed ^ 13), 0.08),
        Fricative::Z => (s_noise(fs, len, seed ^ 14), 0.05),
    };
    add_at(&mut x, start, &noise, gain, &env);
    if kind == Fricative::Z {
        let low = vowel(
            fs,
            len,
            120.0,
            [250.0, 600.0, 900.0],
            [250.0, 600.0, 900.0],
            0.12,
        );
        // Harmonic low band (formants at 250/600/900 Hz): the voiced component of /z/.
        add_at(&mut x, start, &low, 1.0, &env);
    }
    pos += parts[2];
    add_at(
        &mut x,
        pos,
        &vowel(fs, parts[3], 130.0, E_FORM, A_FORM, 0.3),
        1.0,
        |_| 1.0,
    );
    let name = match kind {
        Fricative::S => "s",
        Fricative::Sh => "sh",
        Fricative::Z => "z",
    };
    Fixture {
        name: name.into(),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![x],
        },
        truth: Truth {
            sibilants: vec![(
                parts[0] + parts[1],
                parts[0] + parts[1] + nf,
                kind == Fricative::Z,
            )],
            ..Truth::default()
        },
    }
}

/// Stereo layouts for sibilant spatial tests.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StereoLayout {
    Coherent,
    Panned,
    PartiallyDecorrelated,
}

/// Stereo /s/ fixture: the same structure as [`fricative_fixture`] with a spatial layout.
pub fn stereo_s_fixture(fs: u32, layout: StereoLayout, seed: u64) -> Fixture {
    let mono = fricative_fixture(fs, Fricative::S, seed);
    let base = &mono.audio.channels[0];
    let n = base.len();
    let (l, r) = match layout {
        StereoLayout::Coherent => (base.clone(), base.clone()),
        StereoLayout::Panned => (
            base.clone(),
            base.iter().map(|v| v * 0.35).collect::<Vec<_>>(),
        ),
        StereoLayout::PartiallyDecorrelated => {
            let (s0, s1, _) = mono.truth.sibilants[0];
            let ms = |m: f64| ms_to_samples(m, fs);
            let ramp = ms(25.0);
            let len = s1 - s0 + 2 * ramp;
            let env = ramp_env(len, ramp, ramp);
            let mut l = base.clone();
            let mut r = base.clone();
            let ind_l = s_noise(fs, len, seed ^ 21);
            let ind_r = s_noise(fs, len, seed ^ 22);
            add_at(&mut l, s0 - ramp, &ind_l, 0.06, &env);
            add_at(&mut r, s0 - ramp, &ind_r, 0.06, &env);
            (l, r)
        }
    };
    debug_assert_eq!(l.len(), n);
    Fixture {
        name: format!("stereo_s_{layout:?}"),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![l, r],
        },
        truth: mono.truth,
    }
}

/// Sibilant immediately followed by a stop burst ("st"), then a vowel.
pub fn sibilant_stop_fixture(fs: u32, seed: u64) -> Fixture {
    let ms = |m: f64| ms_to_samples(m, fs);
    let parts = [
        ms(80.0),
        ms(200.0),
        ms(45.0),
        ms(6.0),
        ms(25.0),
        ms(200.0),
        ms(80.0),
    ];
    let n: usize = parts.iter().sum();
    let mut x = hiss(fs, n, seed ^ 31, -72.0);
    let mut pos = parts[0];
    let ramp = ms(20.0);
    let ns = parts[1];
    add_at(
        &mut x,
        pos,
        &s_noise(fs, ns, seed ^ 32),
        0.08,
        ramp_env(ns, ramp, ramp),
    );
    let s_range = (pos, pos + ns);
    pos += parts[1] + parts[2];
    let onset = pos;
    add_at(&mut x, pos, &burst(fs, seed ^ 33), 0.25, |_| 1.0);
    pos += parts[3];
    let na = parts[4];
    let asp = shaped_noise(fs, na, seed ^ 34, |f| band(f, 1000.0, 6000.0, 800.0));
    add_at(&mut x, pos - ms(1.0), &asp, 0.05, |i| {
        (-(i as f64) / (na as f64 * 0.5)).exp()
    });
    pos += parts[4];
    add_at(
        &mut x,
        pos,
        &vowel(fs, parts[5], 110.0, A_FORM, E_FORM, 0.3),
        1.0,
        |_| 1.0,
    );
    Fixture {
        name: "s_then_stop".into(),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![x],
        },
        truth: Truth {
            bursts: vec![onset],
            sibilants: vec![(s_range.0, s_range.1, false)],
            ..Truth::default()
        },
    }
}

/// Burst within the first and last few milliseconds of the clip.
pub fn edge_bursts_fixture(fs: u32, seed: u64) -> Fixture {
    let ms = |m: f64| ms_to_samples(m, fs);
    let n = ms(300.0);
    let mut x = hiss(fs, n, seed ^ 41, -72.0);
    add_at(&mut x, 0, &burst(fs, seed ^ 42), 0.25, |_| 1.0);
    add_at(
        &mut x,
        ms(30.0),
        &vowel(fs, ms(200.0), 150.0, A_FORM, A_FORM, 0.3),
        1.0,
        |_| 1.0,
    );
    let last = n - ms(6.0);
    add_at(&mut x, last, &burst(fs, seed ^ 43), 0.25, |_| 1.0);
    Fixture {
        name: "edge_bursts".into(),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![x],
        },
        truth: Truth {
            bursts: vec![0, last],
            ..Truth::default()
        },
    }
}

/// Whispered vowel: formant-shaped aperiodic noise (must not become a sibilant).
pub fn whisper_fixture(fs: u32, seed: u64) -> Fixture {
    let ms = |m: f64| ms_to_samples(m, fs);
    let n = ms(500.0);
    let shape = |f: f64| {
        let mut g = 0.01;
        for (fm, bw, a) in [
            (700.0, 150.0, 1.0),
            (1200.0, 180.0, 0.6),
            (2500.0, 250.0, 0.25),
        ] {
            let d: f64 = (f - fm) / bw;
            g += a / (1.0 + d * d);
        }
        g
    };
    let mut x = hiss(fs, n, seed ^ 51, -72.0);
    let w = shaped_noise(fs, ms(400.0), seed ^ 52, shape);
    add_at(
        &mut x,
        ms(50.0),
        &w,
        0.08,
        ramp_env(ms(400.0), ms(40.0), ms(40.0)),
    );
    Fixture {
        name: "whisper".into(),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![x],
        },
        truth: Truth::default(),
    }
}

/// /s/ fixture with steady background hiss at -40 dBFS RMS.
pub fn hiss_fixture(fs: u32, seed: u64) -> Fixture {
    let mut f = fricative_fixture(fs, Fricative::S, seed);
    let n = f.audio.channels[0].len();
    let h = hiss(fs, n, seed ^ 61, -40.0);
    for (a, b) in f.audio.channels[0].iter_mut().zip(h) {
        *a += b;
    }
    f.name = "s_with_hiss".into();
    f
}

/// ~50 ms clip containing a vowel fragment.
pub fn tiny_fixture(fs: u32) -> Fixture {
    let n = ms_to_samples(50.0, fs);
    Fixture {
        name: "tiny_50ms".into(),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![vowel(fs, n, 200.0, A_FORM, A_FORM, 0.3)],
        },
        truth: Truth::default(),
    }
}

/// A longer "utterance" chaining all events, used for benchmarks and A/B renders.
pub fn utterance_fixture(fs: u32, seed: u64) -> Fixture {
    let pieces = [
        stop_fixture(fs, seed),
        fricative_fixture(fs, Fricative::S, seed ^ 100),
        sibilant_stop_fixture(fs, seed ^ 200),
        fricative_fixture(fs, Fricative::Sh, seed ^ 300),
        fricative_fixture(fs, Fricative::Z, seed ^ 400),
    ];
    let mut x = Vec::new();
    let mut truth = Truth::default();
    for p in pieces {
        let off = x.len();
        truth.bursts.extend(p.truth.bursts.iter().map(|b| b + off));
        truth.sibilants.extend(
            p.truth
                .sibilants
                .iter()
                .map(|&(a, b, v)| (a + off, b + off, v)),
        );
        truth.vowels.extend(
            p.truth
                .vowels
                .iter()
                .map(|&(a, b, f)| (a + off, b + off, f)),
        );
        x.extend_from_slice(&p.audio.channels[0]);
    }
    Fixture {
        name: "utterance".into(),
        audio: AudioBuffer {
            sample_rate: fs,
            channels: vec![x],
        },
        truth,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixtures_are_deterministic_and_finite() {
        for fs in [8_000, 16_000, 44_100, 48_000, 96_000] {
            let a = stop_fixture(fs, 7);
            let b = stop_fixture(fs, 7);
            assert_eq!(a.audio, b.audio);
            assert!(
                a.audio.channels[0]
                    .iter()
                    .all(|v| v.is_finite() && v.abs() < 1.0)
            );
            let s = fricative_fixture(fs, Fricative::Z, 3);
            assert!(
                s.audio.channels[0]
                    .iter()
                    .all(|v| v.is_finite() && v.abs() < 1.0)
            );
        }
    }
}

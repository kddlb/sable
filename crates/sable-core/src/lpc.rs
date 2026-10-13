//! Linear prediction for rendering WSOLA on the prediction residual (experimental,
//! [`crate::Options::lpc_residual`]).
//!
//! The mono reference is analysed every 5 ms (25 ms Hann window, autocorrelation method
//! with a lag window and a small white-noise floor). Each frame yields reflection
//! coefficients `k_1..k_p` from Levinson–Durbin. Between frame centres the coefficients
//! are interpolated linearly, per sample. Interpolating reflection coefficients keeps
//! every intermediate filter stable (`|k| < 1`), and it means the analysis filter
//! (an FIR lattice, [`LpcTrack::whiten`]) and the synthesis filter (the matching IIR
//! lattice, [`LpcTrack::color`]) are exact inverses for the same coefficient sequence.
//!
//! The renderer whitens every channel with the envelope at its own input time, lets
//! WSOLA rearrange the residual, and colours output frame `y` with the envelope at the
//! time map's input position `U(y)`. The spectral envelope therefore follows the time
//! map continuously, even where WSOLA repeats or skips a pitch period.

use crate::par;
use crate::timemap::TimeMap;
use crate::windows::{hann_periodic, ms_to_samples};
use crate::wsola::reflect;

/// Largest reflection-coefficient magnitude (keeps the synthesis lattice well inside
/// the stability limit).
const K_MAX: f64 = 0.999;
/// White-noise correction: `r[0]` is raised by this fraction (about −40 dB).
const NOISE_FLOOR: f64 = 1e-4;
/// Gaussian lag window bandwidth (Hz): smooths very sharp resonances.
const LAG_BW_HZ: f64 = 60.0;
/// Fixed first-order pre-emphasis `1 − μ z⁻¹` before analysis and whitening, undone after
/// colouring. It flattens the spectral tilt so the noise floor above does not limit the
/// whitening.
const PRE_EMPHASIS: f64 = 0.9;

fn pre_emphasize(x: &[f32]) -> Vec<f32> {
    let mut prev = 0.0f64;
    x.iter()
        .map(|&v| {
            let v = v as f64;
            let o = v - PRE_EMPHASIS * prev;
            prev = v;
            o as f32
        })
        .collect()
}

/// Prediction order for a sample rate: two poles per kHz of bandwidth plus four, 10–32.
pub fn order_for(fs: u32) -> usize {
    (fs as usize / 1000 + 4).clamp(10, 32)
}

/// One analysis frame: reflection coefficients and the final prediction error power.
#[derive(Clone, Debug)]
pub struct LpcFrame {
    pub k: Vec<f64>,
    pub err: f64,
}

impl LpcFrame {
    /// Direct-form coefficients `a_1..a_p` of `A(z) = 1 + Σ a_i z^-i` (step-up recursion).
    pub fn direct_form(&self) -> Vec<f64> {
        let mut a: Vec<f64> = Vec::with_capacity(self.k.len());
        for (m, &km) in self.k.iter().enumerate() {
            let prev = a.clone();
            for i in 0..m {
                a[i] = prev[i] + km * prev[m - 1 - i];
            }
            a.push(km);
        }
        a
    }
}

/// Analysis window, lag window and order for a sample rate.
pub struct LpcSetup {
    pub order: usize,
    pub hop: usize,
    window: Vec<f32>,
    lag: Vec<f64>,
}

impl LpcSetup {
    pub fn new(fs: u32) -> Self {
        let order = order_for(fs);
        let n = (ms_to_samples(25.0, fs) + 1) & !1;
        let lag = (0..=order)
            .map(|i| {
                let a = std::f64::consts::TAU * LAG_BW_HZ * i as f64 / fs as f64;
                (-0.5 * a * a).exp()
            })
            .collect();
        Self {
            order,
            hop: ms_to_samples(5.0, fs),
            window: hann_periodic(n),
            lag,
        }
    }

    /// LPC analysis of the window centred on sample `centre` (mirror-reflected at the
    /// edges). A silent frame gives all-zero coefficients.
    pub fn frame(&self, x: &[f32], centre: usize) -> LpcFrame {
        let p = self.order;
        let n = self.window.len();
        let start = centre as i64 - (n / 2) as i64;
        let seg: Vec<f64> = (0..n)
            .map(|i| x[reflect(start + i as i64, x.len())] as f64 * self.window[i] as f64)
            .collect();
        let mut r: Vec<f64> = (0..=p)
            .map(|lag| seg[lag..].iter().zip(&seg).map(|(a, b)| a * b).sum::<f64>())
            .collect();
        let mut k = vec![0.0; p];
        if r[0].is_nan() || r[0] <= 1e-15 {
            return LpcFrame { k, err: 0.0 };
        }
        for (v, w) in r.iter_mut().zip(&self.lag) {
            *v *= w;
        }
        r[0] *= 1.0 + NOISE_FLOOR;
        // Levinson–Durbin.
        let mut a = vec![0.0; p + 1];
        a[0] = 1.0;
        let mut err = r[0];
        for m in 1..=p {
            let acc: f64 = (0..m).map(|i| a[i] * r[m - i]).sum();
            let km = (-acc / err).clamp(-K_MAX, K_MAX);
            let prev = a.clone();
            for i in 1..m {
                a[i] = prev[i] + km * prev[m - i];
            }
            a[m] = km;
            k[m - 1] = km;
            err *= 1.0 - km * km;
        }
        LpcFrame { k, err }
    }
}

/// Reflection coefficients over the whole input, one frame per `hop` samples.
pub struct LpcTrack {
    pub order: usize,
    hop: usize,
    /// `order` coefficients per frame; frame `j` is centred on input sample `j · hop`.
    k: Vec<f64>,
    frames: usize,
}

impl LpcTrack {
    /// Analyses `x` (the mono reference) on up to `threads` workers. The result does not
    /// depend on the thread count.
    pub fn analyze(x: &[f32], fs: u32, threads: usize) -> Self {
        let setup = LpcSetup::new(fs);
        let x = &pre_emphasize(x)[..];
        let frames = if x.is_empty() {
            0
        } else {
            x.len() / setup.hop + 2
        };
        let per_frame = par::map(frames, par::threads(threads), |j| {
            setup
                .frame(x, (j * setup.hop).min(x.len().saturating_sub(1)))
                .k
        });
        Self {
            order: setup.order,
            hop: setup.hop,
            k: per_frame.into_iter().flatten().collect(),
            frames,
        }
    }

    pub fn frames(&self) -> usize {
        self.frames
    }

    /// Coefficients at fractional input position `pos`, linearly interpolated.
    fn at(&self, pos: f64, out: &mut [f64]) {
        let p = self.order;
        if self.frames == 0 {
            out.fill(0.0);
            return;
        }
        let t = (pos / self.hop as f64).clamp(0.0, (self.frames - 1) as f64);
        let j = (t.floor() as usize).min(self.frames - 1);
        let f = t - j as f64;
        let a = &self.k[j * p..(j + 1) * p];
        if f <= 0.0 || j + 1 >= self.frames {
            out.copy_from_slice(a);
        } else {
            let b = &self.k[(j + 1) * p..(j + 2) * p];
            for ((o, &a), &b) in out.iter_mut().zip(a).zip(b) {
                *o = a + f * (b - a);
            }
        }
    }

    /// Prediction residual of `x`: pre-emphasis, then the FIR lattice `A(z)` with the
    /// coefficients at each sample's own position.
    pub fn whiten(&self, x: &[f32]) -> Vec<f32> {
        let x = pre_emphasize(x);
        let p = self.order;
        let mut k = vec![0.0; p];
        // b[m] holds the backward error b_m of the previous sample.
        let mut b = vec![0.0f64; p];
        x.iter()
            .enumerate()
            .map(|(n, &v)| {
                self.at(n as f64, &mut k);
                let mut f = v as f64;
                let mut b_new = f; // b_0 of this sample
                for m in 0..p {
                    let old = b[m];
                    let f_next = f + k[m] * old;
                    let b_next = old + k[m] * f;
                    b[m] = b_new;
                    b_new = b_next;
                    f = f_next;
                }
                f as f32
            })
            .collect()
    }

    /// Colours a rendered residual: the IIR lattice `1 / A(z)` with, for output frame
    /// `y`, the coefficients at input position `map.inverse(y)`, then de-emphasis.
    /// Returns `None` if the result is not finite.
    pub fn color(&self, e: &[f32], map: &TimeMap) -> Option<Vec<f32>> {
        let p = self.order;
        let mut k = vec![0.0; p];
        let mut b = vec![0.0f64; p];
        let mut out = Vec::with_capacity(e.len());
        let mut prev = 0.0f64;
        for (y, &v) in e.iter().enumerate() {
            self.at(map.inverse(y as f64), &mut k);
            let mut f = v as f64;
            for m in (0..p).rev() {
                f -= k[m] * b[m];
                if m + 1 < p {
                    b[m + 1] = b[m] + k[m] * f;
                }
            }
            if p > 0 {
                b[0] = f;
            }
            let o = f + PRE_EMPHASIS * prev;
            prev = o;
            if !o.is_finite() || o.abs() > 1e6 {
                return None;
            }
            out.push(o as f32);
        }
        Some(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::GaussianStream;

    fn vowelish(fs: u32, n: usize) -> Vec<f32> {
        // Pulse train at a gliding F0 through two resonators, plus a little noise.
        let mut rng = GaussianStream::new(7, 0);
        let (mut phase, mut y1, mut y2, mut z1, mut z2) = (0.0f64, 0.0, 0.0, 0.0, 0.0);
        let res = |f: f64, bw: f64| {
            let r = (-std::f64::consts::PI * bw / fs as f64).exp();
            (
                2.0 * r * (std::f64::consts::TAU * f / fs as f64).cos(),
                -r * r,
            )
        };
        (0..n)
            .map(|i| {
                let f0 = 110.0 + 40.0 * (i as f64 / n as f64);
                phase += f0 / fs as f64;
                let pulse = if phase >= 1.0 {
                    phase -= 1.0;
                    1.0
                } else {
                    0.0
                };
                let f1 = 500.0 + 300.0 * (i as f64 / n as f64);
                let (a1, a2) = res(f1, 80.0);
                let s = pulse + 0.01 * rng.normal();
                let y = s + a1 * y1 + a2 * y2;
                (y2, y1) = (y1, y);
                let (b1, b2) = res(1500.0, 120.0);
                let z = y + b1 * z1 + b2 * z2;
                (z2, z1) = (z1, z);
                (0.02 * z) as f32
            })
            .collect()
    }

    #[test]
    fn whiten_then_color_is_identity_on_identity_map() {
        let fs = 16_000;
        let x = vowelish(fs, 8000);
        let t = LpcTrack::analyze(&x, fs, 1);
        let e = t.whiten(&x);
        let y = t.color(&e, &TimeMap::uniform(8000, 8000)).unwrap();
        let err = x
            .iter()
            .zip(&y)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max);
        let peak = x.iter().map(|v| v.abs()).fold(0f32, f32::max);
        assert!(err < 1e-4 * peak, "max error {err} vs peak {peak}");
    }

    #[test]
    fn residual_is_whiter_than_input() {
        let fs = 16_000;
        let x = vowelish(fs, 8000);
        let t = LpcTrack::analyze(&x, fs, 1);
        let e = t.whiten(&x);
        // Lag-1 autocorrelation drops sharply for a whitened signal.
        let rho1 = |s: &[f32]| {
            let num: f64 = s.windows(2).map(|w| w[0] as f64 * w[1] as f64).sum();
            let den: f64 = s.iter().map(|&v| (v as f64).powi(2)).sum();
            num / den
        };
        assert!(rho1(&x) > 0.8, "input lag-1 {}", rho1(&x));
        assert!(rho1(&e).abs() < 0.5, "residual lag-1 {}", rho1(&e));
    }

    #[test]
    fn silence_and_threads() {
        let z = vec![0f32; 3000];
        let t = LpcTrack::analyze(&z, 48_000, 1);
        assert!(t.whiten(&z).iter().all(|&v| v == 0.0));
        let x = vowelish(48_000, 20_000);
        let a = LpcTrack::analyze(&x, 48_000, 1);
        let b = LpcTrack::analyze(&x, 48_000, 3);
        assert_eq!(a.k, b.k);
    }

    #[test]
    fn direct_form_matches_lattice() {
        let fs = 16_000;
        let x = vowelish(fs, 4000);
        let setup = LpcSetup::new(fs);
        let fr = setup.frame(&x, 2000);
        let a = fr.direct_form();
        // A constant-coefficient track: compare FIR lattice to direct form on a probe.
        let t = LpcTrack {
            order: setup.order,
            hop: 1_000_000,
            k: fr.k.clone(),
            frames: 1,
        };
        let probe: Vec<f32> = (0..200)
            .map(|i| ((i * 37 % 11) as f32 - 5.0) * 0.1)
            .collect();
        let lat = t.whiten(&probe);
        let probe = pre_emphasize(&probe);
        for n in 0..probe.len() {
            let mut d = probe[n] as f64;
            for (i, &ai) in a.iter().enumerate() {
                if n > i {
                    d += ai * probe[n - 1 - i] as f64;
                }
            }
            assert!((d - lat[n] as f64).abs() < 1e-4, "n={n}: {d} vs {}", lat[n]);
        }
    }
}

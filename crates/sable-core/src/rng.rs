//! Deterministic excitation streams: ChaCha8 keyed by `(global seed, segment id)`.

use rand_chacha::ChaCha8Rng;
use rand_chacha::rand_core::{Rng, SeedableRng};

/// Seeded standard-normal source (Box–Muller on 53-bit uniforms).
pub struct GaussianStream {
    rng: ChaCha8Rng,
    spare: Option<f64>,
}

impl GaussianStream {
    /// Independent stream for one event: the global seed selects the key, the segment
    /// ID selects the ChaCha stream, so events never share excitation.
    pub fn new(seed: u64, segment_id: u64) -> Self {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        rng.set_stream(segment_id);
        Self { rng, spare: None }
    }

    fn uniform(&mut self) -> f64 {
        // (0, 1]: avoids ln(0).
        ((self.rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    /// Zero-mean, unit-variance real Gaussian.
    pub fn normal(&mut self) -> f64 {
        if let Some(v) = self.spare.take() {
            return v;
        }
        let (u1, u2) = (self.uniform(), self.uniform());
        let r = (-2.0 * u1.ln()).sqrt();
        let t = std::f64::consts::TAU * u2;
        self.spare = Some(r * t.sin());
        r * t.cos()
    }

    /// Circular complex Gaussian with `E|z|^2 = 1`.
    pub fn complex(&mut self) -> (f64, f64) {
        let s = std::f64::consts::FRAC_1_SQRT_2;
        (self.normal() * s, self.normal() * s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn moments_and_determinism() {
        let mut g = GaussianStream::new(42, 7);
        let n = 200_000;
        let (mut m, mut v) = (0.0, 0.0);
        for _ in 0..n {
            let x = g.normal();
            m += x;
            v += x * x;
        }
        m /= n as f64;
        v /= n as f64;
        assert!(m.abs() < 0.01 && (v - 1.0).abs() < 0.02, "m={m} v={v}");
        let a: Vec<f64> = (0..10)
            .map(|_| GaussianStream::new(1, 2).normal())
            .collect();
        assert!(a.windows(2).all(|w| w[0] == w[1]));
        let mut s1 = GaussianStream::new(1, 2);
        let mut s2 = GaussianStream::new(1, 3);
        assert_ne!(s1.normal(), s2.normal());
    }
}

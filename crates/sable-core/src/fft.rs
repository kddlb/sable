//! Thin `realfft` wrappers with explicit normalization.
//!
//! `realfft`/`rustfft` do **not** scale the inverse transform; [`RealFft::inverse`]
//! divides by `n` so that `inverse(forward(x)) == x`.

use std::sync::Arc;

use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};

pub use realfft::num_complex::Complex32 as Complex;

/// A forward/inverse real FFT pair of fixed size with reusable scratch buffers.
pub struct RealFft {
    n: usize,
    fwd: Arc<dyn RealToComplex<f32>>,
    inv: Arc<dyn ComplexToReal<f32>>,
    time: Vec<f32>,
    spec: Vec<Complex32>,
    scratch_f: Vec<Complex32>,
    scratch_i: Vec<Complex32>,
}

impl RealFft {
    pub fn new(n: usize) -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let fwd = planner.plan_fft_forward(n);
        let inv = planner.plan_fft_inverse(n);
        let scratch_f = fwd.make_scratch_vec();
        let scratch_i = inv.make_scratch_vec();
        Self {
            n,
            time: vec![0.0; n],
            spec: vec![Complex32::new(0.0, 0.0); n / 2 + 1],
            fwd,
            inv,
            scratch_f,
            scratch_i,
        }
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    pub fn bins(&self) -> usize {
        self.n / 2 + 1
    }

    /// Forward transform of `input` (length `n`); returns the `n/2+1` spectrum (unscaled).
    pub fn forward(&mut self, input: &[f32]) -> &[Complex32] {
        self.time.copy_from_slice(input);
        self.fwd
            .process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch_f)
            .expect("forward FFT sizes are fixed at construction");
        &self.spec
    }

    /// Forward transform of `input * window`.
    pub fn forward_windowed(&mut self, input: &[f32], window: &[f32]) -> &[Complex32] {
        for ((t, &x), &w) in self.time.iter_mut().zip(input).zip(window) {
            *t = x * w;
        }
        self.fwd
            .process_with_scratch(&mut self.time, &mut self.spec, &mut self.scratch_f)
            .expect("forward FFT sizes are fixed at construction");
        &self.spec
    }

    /// Inverse transform **normalized by 1/n**. DC and Nyquist imaginary parts are zeroed
    /// first so the output is a real signal.
    pub fn inverse(&mut self, spectrum: &[Complex32]) -> &[f32] {
        self.spec.copy_from_slice(spectrum);
        self.spec[0].im = 0.0;
        let last = self.spec.len() - 1;
        if self.n % 2 == 0 {
            self.spec[last].im = 0.0;
        }
        self.inv
            .process_with_scratch(&mut self.spec, &mut self.time, &mut self.scratch_i)
            .expect("inverse FFT sizes are fixed at construction");
        let s = 1.0 / self.n as f32;
        for v in &mut self.time {
            *v *= s;
        }
        &self.time
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_is_identity() {
        let n = 256;
        let mut f = RealFft::new(n);
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 7919) % 101) as f32 / 50.0 - 1.0)
            .collect();
        let spec = f.forward(&x).to_vec();
        let y = f.inverse(&spec).to_vec();
        for (a, b) in x.iter().zip(&y) {
            assert!((a - b).abs() < 1e-5);
        }
    }
}

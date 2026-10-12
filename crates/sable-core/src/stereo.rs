//! Frequency-domain stereo covariance model for synthesized noise (spec §8.3).
//!
//! `Z_L = Z1`, `Z_R = ρ Z1 + sqrt(1 - |ρ|²) Z2` reproduces a target normalized
//! cross-spectrum `ρ = C_RL / sqrt(P_L P_R)` for independent unit excitations.

/// Normalized complex coherence with power floors; magnitude clamped to `<= 0.999`.
pub fn rho(c_re: f64, c_im: f64, p_l: f64, p_r: f64, floor: f64) -> (f64, f64) {
    if p_l <= floor || p_r <= floor {
        return (0.0, 0.0);
    }
    let d = (p_l * p_r).sqrt();
    let (mut re, mut im) = (c_re / d, c_im / d);
    let m = (re * re + im * im).sqrt();
    if m > 0.999 {
        re *= 0.999 / m;
        im *= 0.999 / m;
    }
    (re, im)
}

/// Mixes two independent unit excitations into the right-channel excitation.
pub fn correlate(rho: (f64, f64), z1: (f64, f64), z2: (f64, f64)) -> (f64, f64) {
    let (r, i) = rho;
    let m2 = (r * r + i * i).min(1.0);
    let s = (1.0 - m2).sqrt();
    (
        r * z1.0 - i * z1.1 + s * z2.0,
        r * z1.1 + i * z1.0 + s * z2.1,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::GaussianStream;

    #[test]
    fn achieves_target_coherence() {
        let target = (0.6, -0.3);
        let mut g = GaussianStream::new(3, 0);
        let (mut cr, mut ci, mut pl, mut pr) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..100_000 {
            let z1 = g.complex();
            let z2 = g.complex();
            let zr = correlate(target, z1, z2);
            // C_RL = E[Z_R conj(Z_L)]
            cr += zr.0 * z1.0 + zr.1 * z1.1;
            ci += zr.1 * z1.0 - zr.0 * z1.1;
            pl += z1.0 * z1.0 + z1.1 * z1.1;
            pr += zr.0 * zr.0 + zr.1 * zr.1;
        }
        let (re, im) = rho(cr, ci, pl, pr, 0.0);
        assert!(
            (re - 0.6).abs() < 0.02 && (im + 0.3).abs() < 0.02,
            "{re} {im}"
        );
        assert!((pr / pl - 1.0).abs() < 0.03);
    }
}

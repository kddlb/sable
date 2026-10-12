//! Shared monotonic input→output sample-frame time map (spec §5.4).
//!
//! Integer anchors `S_i` (input) and `O_i` (output) are exact. Inside a segment the
//! map is either linear or the endpoint-preserving curve `F(v; k) = v + k v (1 - v)`.

use serde::Serialize;

/// Maximum warp curvature; keeps `F'(v) = 1 + k (1 - 2v)` strictly positive.
pub const K_MAX: f64 = 0.9;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MapSeg {
    /// Input start `S_i` and length `L_i`.
    pub s: u64,
    pub l: u64,
    /// Output start `O_i` and length `D_i`.
    pub o: u64,
    pub d: u64,
    /// Warp curvature (0 = linear).
    pub k: f64,
}

impl MapSeg {
    fn f(&self, v: f64) -> f64 {
        v + self.k * v * (1.0 - v)
    }

    /// Solves `F(v) = w` for `v ∈ [0,1]` with the cancellation-free quadratic root.
    fn f_inv(&self, w: f64) -> f64 {
        let k = self.k;
        if k.abs() < 1e-12 {
            return w;
        }
        let b = 1.0 + k;
        let disc = (b * b - 4.0 * k * w).max(0.0);
        (2.0 * w / (b + disc.sqrt())).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TimeMap {
    pub segs: Vec<MapSeg>,
    pub n_in: u64,
    pub n_out: u64,
}

/// Curvature for a closure **before** a burst: output slope reaches 1 at `v = 1`.
/// Returns `(k_used, k_ideal)`.
pub fn closure_k(a: f64) -> (f64, f64) {
    let ideal = 1.0 - 1.0 / a;
    (ideal.clamp(-K_MAX, K_MAX), ideal)
}

/// Curvature for aspiration **after** a burst: output slope starts at 1 at `v = 0`.
pub fn aspiration_k(a: f64) -> (f64, f64) {
    let ideal = 1.0 / a - 1.0;
    (ideal.clamp(-K_MAX, K_MAX), ideal)
}

impl TimeMap {
    /// Builds a map from input lengths, output durations and curvatures.
    pub fn new(lens: &[u64], durs: &[u64], ks: &[f64]) -> Result<Self, String> {
        if lens.len() != durs.len() || lens.len() != ks.len() {
            return Err("length mismatch".into());
        }
        let mut segs = Vec::with_capacity(lens.len());
        let (mut s, mut o) = (0u64, 0u64);
        for i in 0..lens.len() {
            if lens[i] == 0 {
                return Err(format!("segment {i} has zero input length"));
            }
            if !ks[i].is_finite() || ks[i].abs() > K_MAX + 1e-12 {
                return Err(format!("segment {i} curvature {} out of range", ks[i]));
            }
            if durs[i] > 0 {
                segs.push(MapSeg {
                    s,
                    l: lens[i],
                    o,
                    d: durs[i],
                    k: ks[i],
                });
            }
            s += lens[i];
            o += durs[i];
        }
        if segs.is_empty() && o > 0 {
            return Err("no nonempty output segment".into());
        }
        Ok(Self {
            segs,
            n_in: s,
            n_out: o,
        })
    }

    pub fn uniform(n_in: u64, n_out: u64) -> Self {
        Self::new(&[n_in], &[n_out], &[0.0]).expect("uniform map is valid")
    }

    fn seg_for_out(&self, y: f64) -> usize {
        // Last segment with o <= y.
        let idx = self.segs.partition_point(|s| (s.o as f64) <= y);
        idx.saturating_sub(1)
    }

    fn seg_for_in(&self, x: f64) -> usize {
        let idx = self.segs.partition_point(|s| (s.s as f64) <= x);
        idx.saturating_sub(1)
    }

    /// Forward map `T(x)`; extends linearly outside `[0, n_in]`.
    pub fn forward(&self, x: f64) -> f64 {
        if self.segs.is_empty() {
            return 0.0;
        }
        if x < 0.0 {
            return x * self.edge_slope_fwd(true);
        }
        if x > self.n_in as f64 {
            return self.n_out as f64 + (x - self.n_in as f64) * self.edge_slope_fwd(false);
        }
        let s = &self.segs[self.seg_for_in(x)];
        let v = ((x - s.s as f64) / s.l as f64).clamp(0.0, 1.0);
        s.o as f64 + s.d as f64 * s.f(v)
    }

    /// Inverse map `U(y)` (output position → expected input position).
    pub fn inverse(&self, y: f64) -> f64 {
        if self.segs.is_empty() {
            return 0.0;
        }
        if y < 0.0 {
            return y / self.edge_slope_fwd(true);
        }
        if y > self.n_out as f64 {
            return self.n_in as f64 + (y - self.n_out as f64) / self.edge_slope_fwd(false);
        }
        let s = &self.segs[self.seg_for_out(y)];
        let w = ((y - s.o as f64) / s.d as f64).clamp(0.0, 1.0);
        s.s as f64 + s.l as f64 * s.f_inv(w)
    }

    /// `dT/dx` at input position `x` (one-sided from the right at segment starts).
    pub fn slope(&self, x: f64) -> f64 {
        let s = &self.segs[self.seg_for_in(x.clamp(0.0, self.n_in as f64))];
        let v = ((x - s.s as f64) / s.l as f64).clamp(0.0, 1.0);
        (s.d as f64 / s.l as f64) * (1.0 + s.k * (1.0 - 2.0 * v))
    }

    /// `dT/dx` at the very end (`v = 1`) of segment `i`.
    pub fn slope_at_end(&self, i: usize) -> f64 {
        let s = &self.segs[i];
        (s.d as f64 / s.l as f64) * (1.0 - s.k)
    }

    /// `dT/dx` at the very start (`v = 0`) of segment `i`.
    pub fn slope_at_start(&self, i: usize) -> f64 {
        let s = &self.segs[i];
        (s.d as f64 / s.l as f64) * (1.0 + s.k)
    }

    fn edge_slope_fwd(&self, start: bool) -> f64 {
        let v = if start {
            self.slope_at_start(0)
        } else {
            self.slope_at_end(self.segs.len() - 1)
        };
        v.max(1e-6)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_endpoints_and_round_trip() {
        let m = TimeMap::new(&[100, 50, 200], &[150, 50, 400], &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(m.forward(0.0), 0.0);
        assert_eq!(m.forward(100.0), 150.0);
        assert_eq!(m.forward(150.0), 200.0);
        assert_eq!(m.forward(350.0), 600.0);
        for i in 0..=600 {
            let y = i as f64;
            assert!((m.forward(m.inverse(y)) - y).abs() < 1e-9);
        }
    }

    #[test]
    fn curved_maps_keep_anchors_and_get_unit_slope() {
        // Closure stretched 1.1x before a 1x burst, aspiration 1.4x after.
        let a_cl = 1.1;
        let a_as = 1.4;
        let (k_cl, _) = closure_k(a_cl);
        let (k_as, _) = aspiration_k(a_as);
        let m = TimeMap::new(&[1000, 100, 500], &[1100, 100, 700], &[k_cl, 0.0, k_as]).unwrap();
        assert_eq!(m.forward(1000.0), 1100.0);
        assert_eq!(m.forward(1100.0), 1200.0);
        assert_eq!(m.forward(1600.0), 1900.0);
        assert!((m.slope_at_end(0) - 1.0).abs() < 1e-12);
        assert!((m.slope_at_start(2) - 1.0).abs() < 1e-12);
        // Monotonic and invertible.
        let mut prev = -1.0;
        for i in 0..=1900 * 4 {
            let y = i as f64 / 4.0;
            let x = m.inverse(y);
            assert!(x >= prev, "non-monotone inverse at {y}");
            prev = x;
            assert!((m.forward(x) - y).abs() < 1e-7, "round trip at {y}");
        }
    }

    #[test]
    fn clamped_warp_reports_ideal() {
        let (k, ideal) = closure_k(0.4);
        assert_eq!(k, -K_MAX);
        assert!(ideal < -K_MAX);
        let (k, ideal) = aspiration_k(20.0);
        assert_eq!(k, -K_MAX);
        assert!((ideal - (1.0 / 20.0 - 1.0)).abs() < 1e-12);
        let (k, ideal) = aspiration_k(5.0);
        assert_eq!(k, ideal);
    }

    #[test]
    fn zero_duration_segments_are_skipped() {
        let m = TimeMap::new(&[10, 10, 10], &[5, 0, 5], &[0.0; 3]).unwrap();
        assert_eq!(m.segs.len(), 2);
        assert_eq!(m.n_out, 10);
        assert!(m.inverse(5.0) >= 10.0 - 1e-9);
    }

    #[test]
    fn extreme_curvature_inverse_is_stable() {
        for &k in &[-0.9, -0.5, -1e-9, 0.0, 1e-9, 0.5, 0.9] {
            let m = TimeMap::new(&[1000], &[3000], &[k]).unwrap();
            for i in 0..=3000 {
                let y = i as f64;
                let x = m.inverse(y);
                assert!(x.is_finite() && (0.0..=1000.0).contains(&x));
                assert!((m.forward(x) - y).abs() < 1e-6, "k={k} y={y}");
            }
        }
    }
}

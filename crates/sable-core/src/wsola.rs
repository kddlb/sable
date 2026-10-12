//! Variable-rate waveform-similarity overlap-add (spec §6).
//!
//! Output grain centres are laid out on a fixed output grid; each grain's expected
//! source centre comes from the shared inverse time map `U(y)`. A small search around
//! that prediction maximizes normalized cross-correlation (on the analysis-only mono
//! reference) with the already-synthesized overlap. The selected source position is
//! applied identically to every channel. Output = Σ(w·grain) / Σw over a buffer of
//! exactly `N_out` frames. No resampling is used.

use crate::options::QualityPreset;
use crate::report::WsolaStats;
use crate::timemap::TimeMap;
use crate::windows::{hann_offset, ms_to_samples};

#[derive(Clone, Debug)]
pub struct WsolaConfig {
    pub grain: usize,
    pub hop: usize,
    pub search: usize,
    pub t_grain: usize,
    pub t_hop: usize,
    pub t_search: usize,
    /// Half-width (frames) of the output transient zone around protected bursts.
    pub t_zone: usize,
    /// Coarse search decimation step (refined afterwards at full resolution).
    pub coarse_step: usize,
}

impl WsolaConfig {
    pub fn new(fs: u32, quality: QualityPreset) -> Self {
        let (grain_ms, hop_ms, search_ms) = match quality {
            QualityPreset::Fast => (32.0, 8.0, 5.0),
            QualityPreset::Standard => (32.0, 8.0, 8.0),
            QualityPreset::High => (36.0, 6.0, 10.0),
        };
        let even = |v: usize| (v + 1) & !1;
        let grain = even(ms_to_samples(grain_ms, fs).max(8));
        let t_grain = even(ms_to_samples(12.0, fs).max(8));
        let coarse = match quality {
            QualityPreset::High => (fs as usize / 24_000).max(1),
            _ => (fs as usize / 16_000).max(1),
        };
        Self {
            grain,
            hop: ms_to_samples(hop_ms, fs).clamp(1, grain / 4),
            search: ms_to_samples(search_ms, fs),
            t_grain,
            t_hop: ms_to_samples(3.0, fs).clamp(1, t_grain / 4),
            t_search: ms_to_samples(2.0, fs),
            t_zone: ms_to_samples(20.0, fs),
            coarse_step: coarse,
        }
    }
}

/// A burst whose output position is exactly mapped (D == L), so WSOLA must not move it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Anchor {
    pub in_onset: u64,
    pub out_onset: u64,
    pub len: u64,
}

pub struct WsolaOutput {
    pub channels: Vec<Vec<f32>>,
    pub stats: WsolaStats,
}

/// Mirror-reflecting read (no edge repetition of the boundary sample).
#[inline]
pub fn reflect(i: i64, n: usize) -> usize {
    if n <= 1 {
        return 0;
    }
    let p = 2 * (n as i64 - 1);
    let mut m = i.rem_euclid(p);
    if m >= n as i64 {
        m = p - m;
    }
    m as usize
}

struct Grain {
    y: i64,
    n: usize,
    search: usize,
    hop: usize,
    transient: bool,
}

fn schedule(n_out: u64, cfg: &WsolaConfig, anchors: &[Anchor]) -> Vec<Grain> {
    let zones: Vec<(i64, i64)> = anchors
        .iter()
        .map(|a| {
            (
                a.out_onset as i64 - cfg.t_zone as i64,
                (a.out_onset + a.len) as i64 + cfg.t_zone as i64,
            )
        })
        .collect();
    let in_zone = |y: i64| zones.iter().any(|&(a, b)| y >= a && y <= b);
    let mut out = Vec::new();
    let mut y = 0i64;
    while y < n_out as i64 {
        let t = in_zone(y);
        let (n, search, hop) = if t {
            (cfg.t_grain, cfg.t_search, cfg.t_hop)
        } else {
            (cfg.grain, cfg.search, cfg.hop)
        };
        out.push(Grain {
            y,
            n,
            search,
            hop,
            transient: t,
        });
        y += hop as i64;
    }
    out
}

/// Renders exactly `map.n_out` frames per channel.
pub fn render(
    channels: &[&[f32]],
    mono: &[f32],
    map: &TimeMap,
    anchors: &[Anchor],
    cfg: &WsolaConfig,
) -> WsolaOutput {
    let n_out = map.n_out as usize;
    let n_in = mono.len();
    let nch = channels.len();
    let mut acc = vec![vec![0f64; n_out]; nch];
    let mut acc_m = vec![0f64; n_out];
    let mut wsum = vec![0f64; n_out];
    let mut stats = WsolaStats::default();
    if n_out == 0 || n_in == 0 {
        return WsolaOutput {
            channels: vec![vec![0.0; n_out]; nch],
            stats,
        };
    }
    let mean_sq = mono.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / n_in as f64;
    let grains = schedule(map.n_out, cfg, anchors);
    let mut windows: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut prev_c: Option<i64> = None;
    let mut covered_to: i64 = i64::MIN;
    let mut abs_off_sum = 0f64;
    let mut tmpl = Vec::new();
    for g in &grains {
        let half = (g.n / 2) as i64;
        let pred = map.inverse(g.y as f64).round() as i64;
        let start_out = g.y - half;
        // Overlap with already-synthesized output.
        let ov_a = start_out.max(0);
        let ov_b = covered_to.min(g.y + half).min(n_out as i64);
        tmpl.clear();
        if ov_b > ov_a {
            for yy in ov_a..ov_b {
                let w = wsum[yy as usize];
                tmpl.push(if w > 1e-12 {
                    acc_m[yy as usize] / w
                } else {
                    0.0
                });
            }
        }
        let t_energy: f64 = tmpl.iter().map(|v| v * v).sum();
        let anchored = anchors.iter().any(|a| {
            let s = a.in_onset as i64;
            s >= pred - half && s < pred + half
        });
        let crosses = |c: i64| {
            anchors.iter().any(|a| {
                let s = a.in_onset as i64;
                s >= c - half && s < c + half
            })
        };
        // Prefer forward progression: never step back by more than one synthesis hop.
        // (Expansion legitimately re-uses a pitch period, which needs a short step back.)
        let lo_bound = prev_c.map_or(i64::MIN, |p| p - g.hop as i64 + 1);
        let r = g.search as i64;
        let c =
            if anchored || tmpl.len() < 4 || t_energy <= 1e-9 * mean_sq * tmpl.len() as f64 + 1e-30
            {
                stats.predicted_only += 1;
                pred
            } else {
                let off = ov_a - g.y; // template index 0 corresponds to source c + off
                let score = |c: i64, step: usize| -> f64 {
                    let (mut num, mut ec, mut et) = (0f64, 0f64, 0f64);
                    let mut i = 0;
                    while i < tmpl.len() {
                        let x = mono[reflect(c + off + i as i64, n_in)] as f64;
                        let t = tmpl[i];
                        num += x * t;
                        ec += x * x;
                        et += t * t;
                        i += step;
                    }
                    if ec <= 1e-30 || et <= 1e-30 {
                        return -1.0;
                    }
                    num / (ec * et).sqrt()
                };
                let ok = |c: i64| c >= lo_bound && !crosses(c);
                let step = cfg.coarse_step.max(1) as i64;
                let mut best: Option<(f64, i64)> = None;
                let mut d = -r;
                while d <= r {
                    let c = pred + d;
                    if ok(c) {
                        let s = score(c, step as usize);
                        if best.is_none_or(|(b, _)| s > b) {
                            best = Some((s, c));
                        }
                    }
                    d += step;
                }
                match best {
                    None => {
                        stats.predicted_only += 1;
                        pred
                    }
                    Some((_, coarse)) => {
                        let mut best = (f64::NEG_INFINITY, coarse);
                        for c in (coarse - step + 1)..=(coarse + step - 1) {
                            if (c - pred).abs() <= r && ok(c) {
                                let s = score(c, 1);
                                if s > best.0 {
                                    best = (s, c);
                                }
                            }
                        }
                        best.1
                    }
                }
            };
        abs_off_sum += (c - pred).abs() as f64;
        prev_c = Some(c);
        stats.grains += 1;
        if g.transient {
            stats.transient_grains += 1;
        }
        let win = match windows.iter().find(|(n, _)| *n == g.n) {
            Some((_, w)) => w.clone(),
            None => {
                let w = hann_offset(g.n);
                windows.push((g.n, w.clone()));
                w
            }
        };
        for (i, &w) in win.iter().enumerate() {
            let yy = start_out + i as i64;
            if yy < 0 || yy >= n_out as i64 {
                continue;
            }
            let src = reflect(c - half + i as i64, n_in);
            let w = w as f64;
            let yy = yy as usize;
            for ch in 0..nch {
                acc[ch][yy] += w * channels[ch][src] as f64;
            }
            acc_m[yy] += w * mono[src] as f64;
            wsum[yy] += w;
        }
        covered_to = covered_to.max(g.y + half);
    }
    stats.mean_abs_offset = if stats.grains > 0 {
        abs_off_sum / stats.grains as f64
    } else {
        0.0
    };
    let mut out = vec![vec![0f32; n_out]; nch];
    for yy in 0..n_out {
        let w = wsum[yy];
        if w > 1e-9 {
            for ch in 0..nch {
                out[ch][yy] = (acc[ch][yy] / w) as f32;
            }
        } else {
            // Deterministic safe fallback for a provably uncovered sample.
            stats.uncovered_filled += 1;
            let src = reflect(map.inverse(yy as f64).round() as i64, n_in);
            for ch in 0..nch {
                out[ch][yy] = channels[ch][src];
            }
        }
    }
    WsolaOutput {
        channels: out,
        stats,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_indices() {
        assert_eq!(reflect(-1, 5), 1);
        assert_eq!(reflect(-4, 5), 4);
        assert_eq!(reflect(5, 5), 3);
        assert_eq!(reflect(8, 5), 0);
        assert_eq!(reflect(9, 5), 1);
        assert_eq!(reflect(3, 1), 0);
    }

    #[test]
    fn exact_length_and_full_coverage() {
        let fs = 16_000;
        let cfg = WsolaConfig::new(fs, QualityPreset::Standard);
        let x: Vec<f32> = (0..3000).map(|i| (i as f32 * 0.07).sin() * 0.5).collect();
        for n_out in [1u64, 2, 17, 1500, 2999, 3000, 6001, 12000] {
            let map = TimeMap::uniform(3000, n_out);
            let out = render(&[&x], &x, &map, &[], &cfg);
            assert_eq!(out.channels[0].len(), n_out as usize);
            assert_eq!(out.stats.uncovered_filled, 0);
            assert!(out.channels[0].iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn silence_stays_zero() {
        let cfg = WsolaConfig::new(8_000, QualityPreset::Standard);
        let z = vec![0f32; 1000];
        let out = render(&[&z, &z], &z, &TimeMap::uniform(1000, 1777), &[], &cfg);
        assert!(out.channels.iter().flatten().all(|&v| v == 0.0));
    }
}

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

/// A signal extended by mirror reflection on both sides, so reads near the edges are
/// plain slices. Equivalent to indexing through [`reflect`] within `pad` of the ends.
struct Padded {
    data: Vec<f32>,
    pad: usize,
}

impl Padded {
    fn new(x: &[f32], pad: usize) -> Self {
        let n = x.len() as i64;
        let data = (-(pad as i64)..n + pad as i64)
            .map(|i| x[reflect(i, x.len())])
            .collect();
        Self { data, pad }
    }

    /// `len` samples starting at signal index `start`, or `None` beyond the padding.
    #[inline]
    fn slice(&self, start: i64, len: usize) -> Option<&[f32]> {
        let a = usize::try_from(start + self.pad as i64).ok()?;
        self.data.get(a..a.checked_add(len)?)
    }
}

/// Signal samples `start..start + len`, mirror-reflected at the edges.
fn read<'a>(p: &'a Padded, x: &[f32], start: i64, len: usize, tmp: &'a mut Vec<f32>) -> &'a [f32] {
    match p.slice(start, len) {
        Some(s) => s,
        None => {
            tmp.clear();
            tmp.extend((0..len as i64).map(|i| x[reflect(start + i, x.len())]));
            tmp
        }
    }
}

/// True if any onset in the sorted `onsets` lies in `[lo, hi)`.
#[inline]
fn any_in(onsets: &[i64], lo: i64, hi: i64) -> bool {
    let k = onsets.partition_point(|&s| s < lo);
    k < onsets.len() && onsets[k] < hi
}

/// Normalized cross-correlation of `tmpl` with the source at each candidate centre in
/// `cands` (template index 0 at source `c + off`), every `step`-th sample, into `out`.
///
/// `LANES` candidates share one pass so their accumulations run as independent chains;
/// each candidate's own summation order is unchanged, so the scores are bit-identical
/// to scoring the candidates one at a time.
#[allow(clippy::too_many_arguments)]
fn ncc_scores(
    p: &Padded,
    x: &[f32],
    tmpl: &[f64],
    off: i64,
    cands: &[i64],
    step: usize,
    et: f64,
    out: &mut Vec<f64>,
) {
    let n = tmpl.len();
    out.clear();
    let finish = |num: f64, ec: f64| {
        if ec <= 1e-30 || et <= 1e-30 {
            -1.0
        } else {
            num / (ec * et).sqrt()
        }
    };
    const LANES: usize = 8;
    let mut owned: [Vec<f32>; LANES] = Default::default();
    for block in cands.chunks(LANES) {
        for (k, &c) in block.iter().enumerate() {
            if p.slice(c + off, n).is_none() {
                owned[k] = (0..n as i64)
                    .map(|i| x[reflect(c + off + i, x.len())])
                    .collect();
            }
        }
        let src = |k: usize| -> &[f32] {
            match p.slice(block[k] + off, n) {
                Some(s) => s,
                None => &owned[k][..n],
            }
        };
        if block.len() == LANES {
            let srcs: [&[f32]; LANES] = std::array::from_fn(|k| &src(k)[..n]);
            let mut num = [0f64; LANES];
            let mut ec = [0f64; LANES];
            let mut i = 0;
            while i < n {
                let t = tmpl[i];
                for k in 0..LANES {
                    let v = srcs[k][i] as f64;
                    num[k] += v * t;
                    ec[k] += v * v;
                }
                i += step;
            }
            out.extend((0..LANES).map(|k| finish(num[k], ec[k])));
        } else {
            for k in 0..block.len() {
                let s = &src(k)[..n];
                let (mut num, mut ec) = (0f64, 0f64);
                let mut i = 0;
                while i < n {
                    let v = s[i] as f64;
                    num += v * tmpl[i];
                    ec += v * v;
                    i += step;
                }
                out.push(finish(num, ec));
            }
        }
    }
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
    progress: &mut dyn FnMut(usize, usize),
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
    // Reads reach at most `search + n/2` beyond the source range; the margin covers that,
    // and anything further falls back to `reflect`.
    let reach = grains
        .iter()
        .map(|g| g.search + g.n / 2 + 2)
        .max()
        .unwrap_or(0);
    let mono_p = Padded::new(mono, reach);
    let chans_p: Vec<Padded> = channels.iter().map(|c| Padded::new(c, reach)).collect();
    let mut tmp_b = Vec::new();
    let (mut cands, mut scores): (Vec<i64>, Vec<f64>) = (Vec::new(), Vec::new());
    let mut onsets: Vec<i64> = anchors.iter().map(|a| a.in_onset as i64).collect();
    onsets.sort_unstable();
    let mut windows: Vec<(usize, Vec<f32>)> = Vec::new();
    let mut prev_c: Option<i64> = None;
    let mut covered_to: i64 = i64::MIN;
    let mut abs_off_sum = 0f64;
    let mut tmpl = Vec::new();
    for (gi, g) in grains.iter().enumerate() {
        if gi % 64 == 0 {
            progress(gi, grains.len());
        }
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
        let anchored = any_in(&onsets, pred - half, pred + half);
        let crosses = |c: i64| any_in(&onsets, c - half, c + half);
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
                // Template energy at a given decimation does not depend on the candidate.
                let tmpl_energy = |step: usize| -> f64 {
                    let mut et = 0f64;
                    let mut i = 0;
                    while i < tmpl.len() {
                        et += tmpl[i] * tmpl[i];
                        i += step;
                    }
                    et
                };
                let et_by_step = [tmpl_energy(cfg.coarse_step.max(1)), tmpl_energy(1)];
                let ok = |c: i64| c >= lo_bound && !crosses(c);
                let step = cfg.coarse_step.max(1) as i64;
                cands.clear();
                cands.extend(
                    (-r..=r)
                        .step_by(step as usize)
                        .map(|d| pred + d)
                        .filter(|&c| ok(c)),
                );
                ncc_scores(
                    &mono_p,
                    mono,
                    &tmpl,
                    off,
                    &cands,
                    step as usize,
                    et_by_step[0],
                    &mut scores,
                );
                // First maximum in scan order (as a sequential scan would pick).
                let mut best: Option<(f64, i64)> = None;
                for (&c, &s) in cands.iter().zip(&scores) {
                    if best.is_none_or(|(b, _)| s > b) {
                        best = Some((s, c));
                    }
                }
                match best {
                    None => {
                        stats.predicted_only += 1;
                        pred
                    }
                    Some((_, coarse)) => {
                        cands.clear();
                        cands.extend(
                            ((coarse - step + 1)..=(coarse + step - 1))
                                .filter(|&c| (c - pred).abs() <= r && ok(c)),
                        );
                        ncc_scores(
                            &mono_p,
                            mono,
                            &tmpl,
                            off,
                            &cands,
                            1,
                            et_by_step[1],
                            &mut scores,
                        );
                        let mut best = (f64::NEG_INFINITY, coarse);
                        for (&c, &s) in cands.iter().zip(&scores) {
                            if s > best.0 {
                                best = (s, c);
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
        let wi = match windows.iter().position(|(n, _)| *n == g.n) {
            Some(k) => k,
            None => {
                windows.push((g.n, hann_offset(g.n)));
                windows.len() - 1
            }
        };
        let win = &windows[wi].1;
        // Output range of this grain inside [0, n_out).
        let i0 = (-start_out).max(0) as usize;
        let i1 = (n_out as i64 - start_out).clamp(0, g.n as i64) as usize;
        if i0 < i1 {
            let y0 = (start_out + i0 as i64) as usize;
            let s0 = c - half + i0 as i64;
            let len = i1 - i0;
            let w = &win[i0..i1];
            for (ch, cp) in chans_p.iter().enumerate() {
                let src = read(cp, channels[ch], s0, len, &mut tmp_b);
                for ((a, &w), &x) in acc[ch][y0..y0 + len].iter_mut().zip(w).zip(src) {
                    *a += w as f64 * x as f64;
                }
            }
            let src = read(&mono_p, mono, s0, len, &mut tmp_b);
            for (((a, ws), &w), &x) in acc_m[y0..y0 + len]
                .iter_mut()
                .zip(&mut wsum[y0..y0 + len])
                .zip(w)
                .zip(src)
            {
                *a += w as f64 * x as f64;
                *ws += w as f64;
            }
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
            let out = render(&[&x], &x, &map, &[], &cfg, &mut |_, _| {});
            assert_eq!(out.channels[0].len(), n_out as usize);
            assert_eq!(out.stats.uncovered_filled, 0);
            assert!(out.channels[0].iter().all(|v| v.is_finite()));
        }
    }

    #[test]
    fn silence_stays_zero() {
        let cfg = WsolaConfig::new(8_000, QualityPreset::Standard);
        let z = vec![0f32; 1000];
        let out = render(
            &[&z, &z],
            &z,
            &TimeMap::uniform(1000, 1777),
            &[],
            &cfg,
            &mut |_, _| {},
        );
        assert!(out.channels.iter().flatten().all(|&v| v == 0.0));
    }
}

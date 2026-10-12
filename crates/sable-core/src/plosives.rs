//! Anchored dry-copy plosive renderer (spec §7).
//!
//! For each confident burst whose allocated duration equals its input length, output
//! frame `O_i + k` receives source frame `S_i + k` one-to-one for the protected core.
//! Short original context wings (≈4 ms before, ≈6 ms after) carry a raised-cosine
//! constant-sum crossfade with the WSOLA baseline, so the fade never passes through
//! the release impulse itself. No frames are inserted or removed.

use crate::audio::AudioView;
use crate::options::Options;
use crate::pipeline::Plan;
use crate::report::{AnalysisReport, OverlayReport, Warning};
use crate::segments::EventKind;
use crate::windows::{ms_to_samples, raised_cosine_ramp};

pub const PRE_WING_MS: f64 = 4.0;
pub const POST_WING_MS: f64 = 6.0;

/// Result of the plosive pass.
pub struct PlosiveResult {
    /// Event weight per output frame (1 inside protected cores, 0 outside wings).
    pub mask: Vec<f32>,
}

/// One planned overlay in output coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Overlay {
    pub seg: usize,
    pub s: u64,
    pub o: u64,
    pub len: u64,
    pub pre: u64,
    pub post: u64,
}

/// Plans overlays with file-bounded wings; overlapping wings of neighbouring bursts
/// are shortened deterministically so they meet in the middle of the gap.
pub fn plan_overlays(plan: &Plan, n_in: u64, fs: u32) -> Vec<Overlay> {
    let pre0 = ms_to_samples(PRE_WING_MS, fs) as u64;
    let post0 = ms_to_samples(POST_WING_MS, fs) as u64;
    let n_out = plan.map.n_out;
    let mut ov: Vec<Overlay> = plan
        .anchored
        .iter()
        .map(|&i| {
            let seg = &plan.segments[i];
            let (s, o, len) = (seg.start, plan.out_start(i), seg.len());
            Overlay {
                seg: i,
                s,
                o,
                len,
                pre: pre0.min(s).min(o),
                post: post0.min(n_in - (s + len)).min(n_out - (o + len)),
            }
        })
        .collect();
    for j in 1..ov.len() {
        let gap = ov[j].o - (ov[j - 1].o + ov[j - 1].len);
        if ov[j - 1].post + ov[j].pre > gap {
            let half = gap / 2;
            ov[j - 1].post = ov[j - 1].post.min(half);
            ov[j].pre = ov[j].pre.min(gap - ov[j - 1].post);
        }
    }
    ov
}

pub fn apply(
    input: &AudioView<'_>,
    plan: &Plan,
    opts: &Options,
    out: &mut [Vec<f32>],
    report: &mut AnalysisReport,
) -> PlosiveResult {
    let n_out = plan.map.n_out as usize;
    let n_in = input.frames() as u64;
    let mut mask = vec![0f32; n_out];
    // Report bursts that were *not* anchored.
    for (i, seg) in plan.segments.iter().enumerate() {
        if seg.kind != EventKind::PlosiveBurst || plan.anchored.contains(&i) {
            continue;
        }
        let detail = if !opts.enable_plosive {
            "plosive protection disabled by options".to_string()
        } else {
            format!(
                "relaxed: allocated {} frames for a {}-frame burst; dry copy would misalign",
                plan.durations[i],
                seg.len()
            )
        };
        report.warnings.push(Warning::SpecialRendererDisabled {
            segment_id: seg.id,
            reason: detail.clone(),
        });
        report.plosive_overlays.push(OverlayReport {
            segment_id: seg.id,
            applied: false,
            out_start: plan.out_start(i),
            out_end: plan.out_start(i + 1),
            detail,
        });
    }
    if !opts.enable_plosive {
        return PlosiveResult { mask };
    }
    for ov in plan_overlays(plan, n_in, input.sample_rate) {
        let total = ov.pre + ov.len + ov.post;
        for k in 0..total {
            let w = if k < ov.pre {
                raised_cosine_ramp(k as usize, ov.pre as usize)
            } else if k < ov.pre + ov.len {
                1.0
            } else {
                let j = k - ov.pre - ov.len;
                1.0 - raised_cosine_ramp(j as usize, ov.post as usize)
            };
            let y = (ov.o - ov.pre + k) as usize;
            let x = (ov.s - ov.pre + k) as usize;
            for (ch, o) in out.iter_mut().enumerate() {
                let src = input.channels[ch][x];
                o[y] = if w >= 1.0 {
                    src
                } else {
                    (1.0 - w) * o[y] + w * src
                };
            }
            mask[y] = w;
        }
        let seg = &plan.segments[ov.seg];
        report.plosive_overlays.push(OverlayReport {
            segment_id: seg.id,
            applied: true,
            out_start: ov.o - ov.pre,
            out_end: ov.o + ov.len + ov.post,
            detail: format!(
                "dry copy src [{}, {}) -> out [{}, {}); wings {}/{} frames; confidence {:.2}",
                ov.s,
                ov.s + ov.len,
                ov.o,
                ov.o + ov.len,
                ov.pre,
                ov.post,
                seg.confidence
            ),
        });
    }
    report.plosive_overlays.sort_by_key(|o| o.out_start);
    PlosiveResult { mask }
}

//! Two-pass offline pipeline: analysis → allocation → time map → synthesis → overlays.

use crate::audio::{AudioBuffer, AudioView};
use crate::budget::{self, AllocError, Stage};
use crate::detect::{EventDetector, HeuristicDetector};
use crate::error::StretchError;
use crate::features::{self, FeatureConfig, Features};
use crate::options::{Options, StereoMode};
use crate::progress::{Phase, Progress};
use crate::ratio::StretchRatio;
use crate::report::{AnalysisReport, FeatureDump, RenderPath, SegmentReport, Warning};
use crate::segments::{EventKind, Segment, coalesce_to_count, renumber, validate_partition};
use crate::timemap::{TimeMap, aspiration_k, closure_k};
use crate::wsola::{self, Anchor, WsolaConfig};
use crate::{Layers, plosives, sibilants};

/// Everything the renderers need, fixed before synthesis starts.
pub struct Plan {
    pub segments: Vec<Segment>,
    pub durations: Vec<u64>,
    pub map: TimeMap,
    /// Indices of protected bursts that kept `D == L`.
    pub anchored: Vec<usize>,
    pub ks: Vec<f64>,
    pub stage: Option<Stage>,
    /// Output start `O_i` of every segment (length = segments + 1, last = `N_out`).
    pub out_starts: Vec<u64>,
}

impl Plan {
    fn new(
        segments: Vec<Segment>,
        durations: Vec<u64>,
        map: TimeMap,
        anchored: Vec<usize>,
        ks: Vec<f64>,
        stage: Option<Stage>,
    ) -> Self {
        let mut out_starts = Vec::with_capacity(durations.len() + 1);
        let mut o = 0;
        out_starts.push(0);
        for &d in &durations {
            o += d;
            out_starts.push(o);
        }
        Self {
            segments,
            durations,
            map,
            anchored,
            ks,
            stage,
            out_starts,
        }
    }

    pub fn out_start(&self, i: usize) -> u64 {
        self.out_starts[i]
    }
}

fn feature_dump(f: &Features) -> FeatureDump {
    FeatureDump {
        hop: f.hop,
        log_rms: f.log_rms.clone(),
        flux: f.flux.clone(),
        hf_ratio: f.hf_ratio.clone(),
        flatness: f.flatness.clone(),
        centroid_hz: f.centroid_hz.clone(),
        periodicity: f.periodicity.clone(),
        f0_hz: f.f0_hz.clone(),
        zcr: f.zcr.clone(),
    }
}

fn is_protected(s: &Segment, opts: &Options) -> bool {
    opts.enable_plosive && s.kind == EventKind::PlosiveBurst
}

/// Allocation with the full relaxation ladder (stages A–D).
fn plan(
    mut segments: Vec<Segment>,
    n_in: u64,
    n_out: u64,
    r: f64,
    opts: &Options,
    report: &mut AnalysisReport,
) -> Plan {
    let mut stage_d_reason = None;
    let alloc = loop {
        let items: Vec<_> = segments
            .iter()
            .map(|s| budget::item_for(s.kind, s.len(), r, is_protected(s, opts)))
            .collect();
        match budget::allocate(&items, n_out) {
            Ok(a) => break Some(a),
            Err(AllocError::TooManySegments { segments: from, .. }) => {
                if coalesce_to_count(&mut segments, n_out as usize) {
                    report.warnings.push(Warning::SegmentsCoalesced {
                        from,
                        to: segments.len(),
                    });
                } else {
                    stage_d_reason = Some(format!(
                        "{from} segments cannot be coalesced into {n_out} output frames without removing burst anchors"
                    ));
                    break None;
                }
            }
            Err(e) => {
                stage_d_reason = Some(format!("allocator failed: {e:?}"));
                break None;
            }
        }
    };
    let Some(alloc) = alloc else {
        report.warnings.push(Warning::UniformFallback {
            reason: stage_d_reason.unwrap_or_default(),
        });
        let mut seg = Segment::new(0, n_in, EventKind::Other, 0.0);
        seg.reasons.push("stage D: whole-clip uniform map".into());
        return Plan::new(
            vec![seg],
            vec![n_out],
            TimeMap::uniform(n_in, n_out),
            vec![],
            vec![0.0],
            None,
        );
    };
    match alloc.stage {
        Stage::A => {}
        Stage::B => report.warnings.push(Warning::RelaxedNonBurstBudget),
        Stage::C => {
            report.warnings.push(Warning::RelaxedNonBurstBudget);
            report.warnings.push(Warning::BurstProtectionWaived {
                segment_ids: alloc
                    .relaxed_fixed
                    .iter()
                    .map(|&i| segments[i].id)
                    .collect(),
            });
        }
    }
    let d = &alloc.durations;
    let anchored: Vec<usize> = (0..segments.len())
        .filter(|&i| is_protected(&segments[i], opts) && d[i] == segments[i].len())
        .collect();
    for &i in &alloc.relaxed_fixed {
        let msg = format!(
            "burst protection waived: D={} != L={}",
            d[i],
            segments[i].len()
        );
        segments[i].reasons.push(msg);
    }
    // Plosive-phase warps next to anchored bursts.
    let mut ks = vec![0.0; segments.len()];
    for i in 0..segments.len() {
        let a = d[i] as f64 / segments[i].len() as f64;
        let warp = match segments[i].kind {
            EventKind::PlosiveClosure if i + 1 < segments.len() && anchored.contains(&(i + 1)) => {
                Some(closure_k(a))
            }
            EventKind::Aspiration if i > 0 && anchored.contains(&(i - 1)) => Some(aspiration_k(a)),
            _ => None,
        };
        if let Some((k, ideal)) = warp {
            ks[i] = k;
            if (k - ideal).abs() > 1e-12 {
                report.warnings.push(Warning::WarpSlopeRelaxed {
                    segment_id: segments[i].id,
                    k_ideal: ideal,
                    k_used: k,
                });
            }
        }
    }
    let lens: Vec<u64> = segments.iter().map(Segment::len).collect();
    let map = TimeMap::new(&lens, d, &ks).expect("allocator output forms a valid map");
    Plan::new(
        segments,
        alloc.durations.clone(),
        map,
        anchored,
        ks,
        Some(alloc.stage),
    )
}

/// `[M, S]` → `[M + S, M − S]`.
fn ms_to_lr(mut ms: Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let s = ms.pop().expect("side channel");
    let m = ms.pop().expect("mid channel");
    let l = m.iter().zip(&s).map(|(a, b)| a + b).collect();
    let r = m.iter().zip(&s).map(|(a, b)| a - b).collect();
    vec![l, r]
}

/// Output of [`render`].
pub struct Rendered {
    /// Exactly `n_out` frames per channel.
    pub channels: Vec<Vec<f32>>,
    pub layers: Option<Layers>,
    /// Upper voiced F0 of the input (85th percentile), clamped to 80–400 Hz; 200 Hz when
    /// nothing is voiced. Sets the envelope lifter for formant-preserving pitch shifts.
    pub f0_ref_hz: f64,
}

/// 85th percentile of the voiced F0 track, clamped to a sane speech range.
fn f0_reference(f: &Features) -> f64 {
    let mut v: Vec<f32> = f.f0_hz.iter().copied().filter(|&x| x > 0.0).collect();
    if v.is_empty() {
        return 200.0;
    }
    v.sort_by(f32::total_cmp);
    let i = ((v.len() - 1) as f64 * 0.85).round() as usize;
    (v[i] as f64).clamp(80.0, 400.0)
}

/// Full rendering path. Returns exactly `n_out` frames per channel.
pub fn render(
    input: &AudioView<'_>,
    ratio: StretchRatio,
    n_out: u64,
    opts: &Options,
    report: &mut AnalysisReport,
    progress: &mut Progress<'_>,
) -> Result<Rendered, StretchError> {
    let fs = input.sample_rate;
    let n_in = input.frames() as u64;
    let r = ratio.as_f64();
    progress.start(Phase::Analysis);
    let feats = features::analyze(input.channels, fs, &FeatureConfig::default());
    if opts.diagnostics {
        report.features = Some(feature_dump(&feats));
    }
    let f0_ref_hz = f0_reference(&feats);
    let voiced = feats.f0_hz.iter().filter(|&&f| f > 0.0).count();
    progress.finish(
        Phase::Analysis,
        format!(
            "{} frames, {:.0}% voiced, F0 ref {:.0} Hz",
            feats.f0_hz.len(),
            100.0 * voiced as f64 / feats.f0_hz.len().max(1) as f64,
            f0_ref_hz
        ),
    );
    progress.start(Phase::Detection);
    let segments = if opts.uniform_timing {
        let mut s = Segment::new(0, n_in, EventKind::Other, 1.0);
        s.reasons
            .push("uniform timing requested (generic WSOLA)".into());
        vec![s]
    } else {
        let mut det = HeuristicDetector::default().detect(&feats);
        if !opts.enable_plosive {
            for s in det
                .segments
                .iter_mut()
                .filter(|s| s.kind == EventKind::PlosiveBurst)
            {
                s.reasons
                    .push("plosive protection disabled by options".into());
            }
        }
        renumber(&mut det.segments);
        det.segments
    };
    validate_partition(&segments, n_in).map_err(StretchError::Internal)?;
    if progress.active() {
        let count = |f: fn(EventKind) -> bool| segments.iter().filter(|s| f(s.kind)).count();
        progress.finish(
            Phase::Detection,
            format!(
                "{} segments: {} bursts, {} sibilants, {} voiced",
                segments.len(),
                count(|k| k == EventKind::PlosiveBurst),
                count(EventKind::is_sibilant),
                count(|k| k == EventKind::Voiced)
            ),
        );
    }
    progress.start(Phase::Allocation);
    let plan = plan(segments, n_in, n_out, r, opts, report);
    progress.finish(
        Phase::Allocation,
        match plan.stage {
            Some(st) => format!("stage {st:?}, {} anchored bursts", plan.anchored.len()),
            None => "uniform map".to_string(),
        },
    );
    report.allocator_stage = plan.stage;
    report.path = if plan.stage.is_some() && !opts.uniform_timing {
        RenderPath::Full
    } else {
        RenderPath::Uniform
    };

    // Working channels: L/R, or M/S in mid/side mode. WSOLA overlap-add is linear and
    // the grain positions are chosen once on the mono reference, so the M/S baseline
    // recombines to exactly the L/R baseline; only the special renderers differ.
    let mid_side = opts.stereo_mode == StereoMode::MidSide && input.channels.len() == 2;
    let ms_src: Vec<Vec<f32>> = if mid_side {
        let (l, rr) = (input.channels[0], input.channels[1]);
        vec![
            l.iter().zip(rr).map(|(a, b)| 0.5 * (a + b)).collect(),
            l.iter().zip(rr).map(|(a, b)| 0.5 * (a - b)).collect(),
        ]
    } else {
        Vec::new()
    };
    let ms_refs: Vec<&[f32]> = ms_src.iter().map(Vec::as_slice).collect();
    let work = if mid_side {
        AudioView {
            sample_rate: fs,
            channels: &ms_refs,
        }
    } else {
        *input
    };
    // Special renderers see only the mid in mid/side mode.
    let special = if mid_side {
        AudioView {
            sample_rate: fs,
            channels: &ms_refs[..1],
        }
    } else {
        *input
    };
    let n_special = special.channels.len();
    if mid_side {
        report.stereo_mode = StereoMode::MidSide;
        report.side_duck_db = Some(opts.side_duck_db.min(0.0));
    }

    // Baseline WSOLA.
    let anchors: Vec<Anchor> = plan
        .anchored
        .iter()
        .map(|&i| Anchor {
            in_onset: plan.segments[i].start,
            out_onset: plan.out_start(i),
            len: plan.segments[i].len(),
        })
        .collect();
    let cfg = WsolaConfig::new(fs, opts.quality);
    progress.start(Phase::Wsola);
    let base = wsola::render(
        work.channels,
        &feats.mono,
        &plan.map,
        &anchors,
        &cfg,
        &mut |d, t| progress.step(Phase::Wsola, d, t),
    );
    progress.finish(
        Phase::Wsola,
        format!(
            "{} grains ({} transient)",
            base.stats.grains, base.stats.transient_grains
        ),
    );
    if base.stats.uncovered_filled > 0 {
        report.warnings.push(Warning::UncoveredSamplesFilled {
            count: base.stats.uncovered_filled,
        });
    }
    report.wsola = Some(base.stats);
    let baseline = base.channels;

    // Plosive overlays at fixed coordinates.
    let mut out = baseline.clone();
    progress.start(Phase::Plosives);
    let plos = plosives::apply(&special, &plan, opts, &mut out[..n_special], report);
    progress.finish(
        Phase::Plosives,
        format!(
            "{} of {} bursts copied",
            report.plosive_overlays.iter().filter(|o| o.applied).count(),
            report.plosive_overlays.len()
        ),
    );
    let after_plosive = opts.keep_layers.then(|| out.clone());

    // Sibilant overlays at fixed coordinates.
    progress.start(Phase::Sibilants);
    let sib = sibilants::apply(
        &special,
        &feats,
        &plan,
        opts,
        &plos.mask,
        &mut out[..n_special],
        report,
    );
    progress.finish(
        Phase::Sibilants,
        format!(
            "{} of {} sibilants resynthesized",
            report
                .sibilant_overlays
                .iter()
                .filter(|o| o.applied)
                .count(),
            report.sibilant_overlays.len()
        ),
    );

    // Mid/side: duck the side where the mid is specially rendered, then recombine.
    let (out, baseline, after_plosive) = if mid_side {
        let duck = 10f64.powf(opts.side_duck_db.min(0.0) / 20.0) as f32;
        for (t, s) in out[1].iter_mut().enumerate() {
            let m = plos.mask[t].max(sib.mask[t]).clamp(0.0, 1.0);
            *s *= 1.0 - m * (1.0 - duck);
        }
        (
            ms_to_lr(out),
            ms_to_lr(baseline),
            after_plosive.map(ms_to_lr),
        )
    } else {
        (out, baseline, after_plosive)
    };

    // Segment table.
    let mut o = 0u64;
    report.segments = plan
        .segments
        .iter()
        .enumerate()
        .map(|(i, s)| {
            let d = plan.durations[i];
            let (pref, _, _, _) = budget::class_policy(
                if is_protected(s, opts) || s.kind != EventKind::PlosiveBurst {
                    s.kind
                } else {
                    EventKind::Transition
                },
                r,
            );
            let rep = SegmentReport {
                id: s.id,
                kind: s.kind,
                confidence: s.confidence,
                in_start: s.start,
                in_len: s.len(),
                out_start: o,
                out_len: d,
                preferred_factor: pref,
                local_ratio: d as f64 / s.len() as f64,
                warp_k: plan.ks[i],
                anchor: s.anchor,
                scores: s.scores.clone(),
                reasons: s.reasons.clone(),
            };
            o += d;
            rep
        })
        .collect();

    let layers = opts.keep_layers.then(|| {
        let mono = |v: Vec<f32>| AudioBuffer {
            sample_rate: fs,
            channels: vec![v],
        };
        Layers {
            baseline: AudioBuffer {
                sample_rate: fs,
                channels: baseline,
            },
            after_plosive: AudioBuffer {
                sample_rate: fs,
                channels: after_plosive.unwrap_or_default(),
            },
            plosive_mask: mono(plos.mask),
            sibilant_mask: mono(sib.mask),
            sibilant_noise: AudioBuffer {
                sample_rate: fs,
                channels: sib.noise,
            },
        }
    });
    Ok(Rendered {
        channels: out,
        layers,
        f0_ref_hz,
    })
}

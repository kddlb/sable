//! M3: anchored burst rendering on synthetic stops (spec §7, §11.2 fixture 1).

use sable_core::segments::EventKind;
use sable_core::synth::{self, FixtureRng};
use sable_core::{AudioView, Options, StretchRatio, Warning, stretch};

const FACTORS: [&str; 10] = [
    "0.5", "0.65", "0.75", "0.9", "1.1", "1.25", "1.5", "1.8", "2", "1/4",
];
const RATES: [u32; 5] = [8_000, 16_000, 44_100, 48_000, 96_000];

fn run(chans: &[&[f32]], fs: u32, f: &str, opts: &Options) -> sable_core::ProcessResult {
    stretch(
        &AudioView {
            sample_rate: fs,
            channels: chans,
        },
        StretchRatio::parse(f).unwrap(),
        opts,
    )
    .unwrap()
}

/// Pre-emphasised RMS envelope over ~1 ms.
fn transient_env(x: &[f32], fs: u32) -> Vec<f64> {
    let w = (fs as usize / 1000).max(1);
    let d: Vec<f64> = (0..x.len())
        .map(|i| (x[i] as f64 - if i > 0 { 0.97 * x[i - 1] as f64 } else { 0.0 }).powi(2))
        .collect();
    (0..d.len())
        .map(|i| {
            let a = i.saturating_sub(w / 2);
            let b = (i + w / 2 + 1).min(d.len());
            (d[a..b].iter().sum::<f64>() / (b - a) as f64).sqrt()
        })
        .collect()
}

/// Number of envelope peaks >= `frac` of the main peak within ±`radius` of `center`,
/// counting peaks separated by at least `sep` frames.
fn release_peaks(
    env: &[f64],
    center: usize,
    radius: usize,
    sep: usize,
    frac: f64,
) -> (usize, usize) {
    let a = center.saturating_sub(radius);
    let b = (center + radius).min(env.len());
    let (mut main_i, mut main_v) = (a, 0.0);
    for (i, &v) in env.iter().enumerate().take(b).skip(a) {
        if v > main_v {
            main_v = v;
            main_i = i;
        }
    }
    let mut count = 0;
    let mut i = a;
    while i < b {
        if env[i] >= frac * main_v {
            count += 1;
            // Skip this peak region.
            let mut j = i;
            while j < b && (env[j] >= 0.25 * main_v || j < i + sep) {
                j += 1;
            }
            i = j;
        } else {
            i += 1;
        }
    }
    (count, main_i)
}

#[test]
fn burst_copied_one_to_one_at_mapped_onset() {
    for fs in RATES {
        let fx = synth::stop_fixture(fs, 1);
        let x = &fx.audio.channels[0];
        for f in FACTORS {
            let out = run(&[x.as_slice()], fs, f, &Options::default());
            let rep = &out.report;
            let seg = rep
                .segments
                .iter()
                .find(|s| s.kind == EventKind::PlosiveBurst)
                .unwrap_or_else(|| panic!("fs={fs} r={f}: no burst"));
            let ov = rep
                .plosive_overlays
                .iter()
                .find(|o| o.segment_id == seg.id)
                .unwrap();
            assert!(ov.applied, "fs={fs} r={f}: {}", ov.detail);
            assert_eq!(seg.out_len, seg.in_len, "fs={fs} r={f}");
            let y = &out.audio.channels[0];
            for k in 0..seg.in_len as usize {
                assert_eq!(
                    y[seg.out_start as usize + k].to_bits(),
                    x[seg.in_start as usize + k].to_bits(),
                    "fs={fs} r={f} core sample {k}"
                );
            }
        }
    }
}

#[test]
fn exactly_one_release_peak() {
    for fs in [16_000, 48_000] {
        let fx = synth::stop_fixture(fs, 2);
        let x = &fx.audio.channels[0];
        let sep = fs as usize * 3 / 1000;
        for f in FACTORS {
            let out = run(&[x.as_slice()], fs, f, &Options::default());
            let segs = &out.report.segments;
            let bi = segs
                .iter()
                .position(|s| s.kind == EventKind::PlosiveBurst)
                .unwrap();
            let burst = &segs[bi];
            // Window: the stop's own phases (closure .. end of aspiration), max ±40 ms.
            let radius = fs as usize * 40 / 1000;
            let out_a = segs[bi - 1].out_start as usize;
            let out_b = (segs[bi + 1].out_start + segs[bi + 1].out_len) as usize;
            let in_a = segs[bi - 1].in_start as usize;
            let in_b = (segs[bi + 1].in_start + segs[bi + 1].in_len) as usize;
            let o = burst.out_start as usize;
            let i = burst.in_start as usize;
            let env_out = transient_env(&out.audio.channels[0], fs);
            let env_in = transient_env(x, fs);
            let r_out = radius.min(o - out_a).min(out_b - o);
            let r_in = radius.min(i - in_a).min(in_b - i);
            let (count, main) = release_peaks(&env_out, o, r_out, sep, 0.5);
            let (count_in, _) = release_peaks(&env_in, i, r_in, sep, 0.5);
            assert_eq!(count_in, 1, "fixture should have a single release");
            assert_eq!(count, 1, "fs={fs} r={f}: {count} comparable release peaks");
            let tol = fs as i64 / 1000;
            assert!(
                (main as i64 - o as i64).abs() <= tol + burst.in_len as i64,
                "fs={fs} r={f}: main peak at {main}, onset {o}"
            );
        }
    }
}

#[test]
fn plosive_disabled_falls_back_to_wsola() {
    let fs = 16_000;
    let fx = synth::stop_fixture(fs, 3);
    let opts = Options {
        enable_plosive: false,
        ..Options::default()
    };
    let out = run(&fx.audio.view_refs(), fs, "1.5", &opts);
    assert!(out.report.plosive_overlays.iter().all(|o| !o.applied));
    assert!(
        out.report
            .plosive_overlays
            .iter()
            .any(|o| o.detail.contains("disabled"))
    );
}

#[test]
fn infeasible_bursts_are_relaxed_and_reported() {
    // A clip of only burst-like clicks at 4x cannot keep every burst at 1x.
    let fs = 16_000;
    let mut x = vec![0f32; 2400];
    let mut rng = FixtureRng::new(9);
    for k in 0..6 {
        let at = 100 + k * 400;
        for i in 0..48 {
            let decay = (-(i as f64) / 10.0).exp();
            x[at + i] = (rng.normal() * 0.3 * decay) as f32;
        }
    }
    let out = run(&[x.as_slice()], fs, "4", &Options::default());
    assert_eq!(out.audio.channels[0].len(), 9600);
    let bursts = out
        .report
        .segments
        .iter()
        .filter(|s| s.kind == EventKind::PlosiveBurst)
        .count();
    if bursts > 0 {
        // Either all bursts kept 1x (silence absorbed the stretch) or the waiver is reported.
        let waived = out.report.warnings.iter().find_map(|w| match w {
            Warning::BurstProtectionWaived { segment_ids } => Some(segment_ids.clone()),
            _ => None,
        });
        for s in out
            .report
            .segments
            .iter()
            .filter(|s| s.kind == EventKind::PlosiveBurst)
        {
            let ov = out
                .report
                .plosive_overlays
                .iter()
                .find(|o| o.segment_id == s.id)
                .unwrap();
            if s.out_len == s.in_len {
                assert!(ov.applied);
            } else {
                assert!(!ov.applied);
                assert!(waived.as_ref().is_some_and(|ids| ids.contains(&s.id)));
            }
        }
    }
}

#[test]
fn edge_bursts_at_file_boundaries() {
    for fs in RATES {
        let fx = synth::edge_bursts_fixture(fs, 9);
        for f in ["0.5", "1.5", "2"] {
            let out = run(&fx.audio.view_refs(), fs, f, &Options::default());
            for ov in out.report.plosive_overlays.iter().filter(|o| o.applied) {
                assert!(ov.out_end <= out.report.n_out);
            }
            assert!(
                out.report
                    .plosive_overlays
                    .iter()
                    .filter(|o| o.applied)
                    .count()
                    >= 1
            );
        }
    }
}

#[test]
fn stereo_burst_copied_on_both_channels() {
    let fs = 48_000;
    let fx = synth::stop_fixture(fs, 5);
    let l = fx.audio.channels[0].clone();
    let r: Vec<f32> = l.iter().map(|v| v * 0.7).collect();
    let out = run(
        &[l.as_slice(), r.as_slice()],
        fs,
        "1.5",
        &Options::default(),
    );
    let seg = out
        .report
        .segments
        .iter()
        .find(|s| s.kind == EventKind::PlosiveBurst)
        .unwrap();
    for k in 0..seg.in_len as usize {
        let o = seg.out_start as usize + k;
        let i = seg.in_start as usize + k;
        assert_eq!(out.audio.channels[0][o], l[i]);
        assert_eq!(out.audio.channels[1][o], r[i]);
    }
}

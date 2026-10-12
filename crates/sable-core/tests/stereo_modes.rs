//! Mid/side stereo mode: equivalence with L/R where no special renderer acts, exact
//! length, side ducking, and mono fallback.

use sable_core::synth::{self, StereoLayout};
use sable_core::{
    AudioBuffer, AudioView, Options, ProcessResult, StereoMode, StretchRatio, stretch,
};

fn run(a: &AudioBuffer, f: &str, opts: &Options) -> ProcessResult {
    let refs = a.view_refs();
    stretch(
        &AudioView {
            sample_rate: a.sample_rate,
            channels: &refs,
        },
        StretchRatio::parse(f).unwrap(),
        opts,
    )
    .unwrap()
}

fn ms(opts: Options) -> Options {
    Options {
        stereo_mode: StereoMode::MidSide,
        ..opts
    }
}

/// Stereo stop + /s/ material with a partly decorrelated image.
fn stereo_utterance(fs: u32) -> AudioBuffer {
    let s = synth::stereo_s_fixture(fs, StereoLayout::PartiallyDecorrelated, 3).audio;
    let stop = synth::stop_fixture(fs, 4).audio.channels.remove(0);
    let mut l = stop.clone();
    let mut r: Vec<f32> = stop.iter().map(|v| v * 0.6).collect();
    l.extend_from_slice(&s.channels[0]);
    r.extend_from_slice(&s.channels[1]);
    AudioBuffer {
        sample_rate: fs,
        channels: vec![l, r],
    }
}

#[test]
fn mid_side_equals_left_right_without_special_renderers() {
    let a = stereo_utterance(16_000);
    let plain = Options {
        enable_plosive: false,
        enable_sibilant: false,
        ..Options::default()
    };
    for f in ["0.75", "1.5"] {
        let lr = run(&a, f, &plain);
        let m = run(&a, f, &ms(plain.clone()));
        assert_eq!(m.report.stereo_mode, StereoMode::MidSide);
        for (x, y) in lr.audio.channels.iter().zip(&m.audio.channels) {
            assert_eq!(x.len(), y.len());
            let dev = x
                .iter()
                .zip(y)
                .map(|(p, q)| (p - q).abs())
                .fold(0f32, f32::max);
            assert!(dev < 1e-5, "r={f}: L/R vs M/S baseline deviation {dev}");
        }
    }
}

#[test]
fn mid_side_exact_length_and_mono_fallback() {
    for fs in [16_000, 48_000] {
        let a = stereo_utterance(fs);
        for f in ["0.5", "0.75", "1.25", "2", "1/4"] {
            let out = run(&a, f, &ms(Options::default()));
            let n = StretchRatio::parse(f)
                .unwrap()
                .target_frames(a.frames() as u64)
                .unwrap() as usize;
            assert!(
                out.audio
                    .channels
                    .iter()
                    .all(|c| c.len() == n && c.iter().all(|v| v.is_finite()))
            );
        }
    }
    let mono = synth::stop_fixture(16_000, 1).audio;
    let out = run(&mono, "1.5", &ms(Options::default()));
    assert_eq!(out.report.stereo_mode, StereoMode::LeftRight);
    assert_eq!(out.audio.channels.len(), 1);
}

#[test]
fn side_is_ducked_inside_resynthesized_sibilants() {
    let fs = 32_000;
    let a = synth::stereo_s_fixture(fs, StereoLayout::PartiallyDecorrelated, 7).audio;
    let side = |p: &ProcessResult, s: usize, e: usize| {
        let (l, r) = (&p.audio.channels[0], &p.audio.channels[1]);
        let ms: f64 = (s..e)
            .map(|i| (0.5 * (l[i] - r[i]) as f64).powi(2))
            .sum::<f64>()
            / (e - s) as f64;
        ms.sqrt()
    };
    let ducked = run(&a, "1.5", &ms(Options::default()));
    let flat = run(
        &a,
        "1.5",
        &ms(Options {
            side_duck_db: 0.0,
            ..Options::default()
        }),
    );
    let ov = ducked
        .report
        .sibilant_overlays
        .iter()
        .find(|o| o.applied)
        .expect("mid sibilant resynthesized");
    // Interior where the synthetic weight is 1 (beyond guard + fade).
    let edge = fs as usize * 30 / 1000;
    let (s, e) = (ov.out_start as usize + edge, ov.out_end as usize - edge);
    let db = 20.0 * (side(&ducked, s, e) / side(&flat, s, e)).log10();
    assert!((db + 6.0).abs() < 0.2, "side duck measured {db:.2} dB");
    // Outside every event the side is untouched.
    let pre = fs as usize / 20;
    assert!((side(&ducked, 0, pre) - side(&flat, 0, pre)).abs() < 1e-9);
    assert_eq!(ducked.report.side_duck_db, Some(-6.0));
}

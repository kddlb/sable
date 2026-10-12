//! M2: generic stretcher quality checks (pitch preservation, stereo alignment, silence).

use sable_core::metrics::median_f0;
use sable_core::synth;
use sable_core::{AudioView, Options, RenderPath, StretchRatio, stretch};

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

#[test]
fn f0_preserved_within_two_percent() {
    for fs in [16_000, 48_000] {
        for f0 in [80.0, 140.0, 250.0] {
            let fx = synth::vowel_fixture(fs, f0);
            let x = &fx.audio.channels[0];
            for f in ["0.5", "0.75", "1.25", "1.5", "2"] {
                let r = StretchRatio::parse(f).unwrap().as_f64();
                for opts in [Options::default(), Options::generic_wsola()] {
                    let out = run(&[x.as_slice()], fs, f, &opts);
                    let y = &out.audio.channels[0];
                    let (a, b, _) = fx.truth.vowels[0];
                    let (oa, ob) = ((a as f64 * r) as usize, (b as f64 * r) as usize);
                    let est = median_f0(y, fs, oa, ob, 0.6)
                        .unwrap_or_else(|| panic!("no F0 fs={fs} f0={f0} r={f}"));
                    let err = (est - f0).abs() / f0;
                    assert!(
                        err < 0.02,
                        "fs={fs} f0={f0} r={f}: estimated {est:.2} Hz ({:.2}%)",
                        err * 100.0
                    );
                }
            }
        }
    }
}

#[test]
fn stereo_grains_shared_across_channels() {
    let fs = 16_000;
    let fx = synth::stop_fixture(fs, 3);
    let l = fx.audio.channels[0].clone();
    let r: Vec<f32> = l.iter().map(|v| v * 0.5).collect();
    for f in ["0.75", "1.5"] {
        let out = run(&[l.as_slice(), r.as_slice()], fs, f, &Options::default());
        let (ol, or) = (&out.audio.channels[0], &out.audio.channels[1]);
        let max_dev = ol
            .iter()
            .zip(or)
            .map(|(a, b)| (a * 0.5 - b).abs())
            .fold(0f32, f32::max);
        assert!(max_dev < 1e-6, "channels diverged by {max_dev}");
    }
}

#[test]
fn full_path_is_used_and_reported() {
    let fs = 16_000;
    let fx = synth::utterance_fixture(fs, 1);
    let out = run(&fx.audio.view_refs(), fs, "1.5", &Options::default());
    assert_eq!(out.report.path, RenderPath::Full);
    assert!(out.report.allocator_stage.is_some());
    let segs = &out.report.segments;
    assert!(!segs.is_empty());
    // Report segment table tiles both input and output exactly.
    let mut ip = 0;
    let mut op = 0;
    for s in segs {
        assert_eq!(s.in_start, ip);
        assert_eq!(s.out_start, op);
        ip += s.in_len;
        op += s.out_len;
    }
    assert_eq!(ip, out.report.n_in);
    assert_eq!(op, out.report.n_out);
    let g = run(&fx.audio.view_refs(), fs, "1.5", &Options::generic_wsola());
    assert_eq!(g.report.path, RenderPath::Uniform);
}

#[test]
fn level_is_preserved_on_steady_vowel() {
    let fs = 16_000;
    let fx = synth::vowel_fixture(fs, 140.0);
    let x = &fx.audio.channels[0];
    let rin = sable_core::metrics::rms(x, fs as usize / 5, fs as usize * 4 / 5);
    for f in ["0.5", "2"] {
        let out = run(&[x.as_slice()], fs, f, &Options::default());
        let y = &out.audio.channels[0];
        let n = y.len();
        let rout = sable_core::metrics::rms(y, n / 5, n * 4 / 5);
        let db = 20.0 * (rout / rin).log10();
        assert!(db.abs() < 1.0, "r={f}: level change {db:.2} dB");
    }
}

//! Same input, configuration and seed → identical samples and reports (spec §2, §11.1).

use sable_core::synth::{self, StereoLayout};
use sable_core::{AudioView, Options, StretchRatio, stretch};

#[test]
fn repeated_runs_are_bit_identical() {
    for fx in [
        synth::utterance_fixture(16_000, 3),
        synth::stereo_s_fixture(48_000, StereoLayout::PartiallyDecorrelated, 4),
    ] {
        let refs = fx.audio.view_refs();
        let view = AudioView {
            sample_rate: fx.audio.sample_rate,
            channels: &refs,
        };
        for f in ["0.75", "1.5"] {
            let r = StretchRatio::parse(f).unwrap();
            let opts = Options {
                diagnostics: true,
                ..Options::default()
            };
            let a = stretch(&view, r, &opts).unwrap();
            let b = stretch(&view, r, &opts).unwrap();
            for (x, y) in a.audio.channels.iter().zip(&b.audio.channels) {
                assert!(x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()));
            }
            assert_eq!(
                serde_json::to_string(&a.report).unwrap(),
                serde_json::to_string(&b.report).unwrap()
            );
        }
    }
}

#[test]
fn layers_are_consistent_and_exact_length() {
    let fx = synth::utterance_fixture(16_000, 5);
    let refs = fx.audio.view_refs();
    let view = AudioView {
        sample_rate: 16_000,
        channels: &refs,
    };
    let opts = Options {
        keep_layers: true,
        ..Options::default()
    };
    let out = stretch(&view, StretchRatio::parse("1.5").unwrap(), &opts).unwrap();
    let layers = out.layers.expect("layers requested");
    let n = out.report.n_out as usize;
    for (name, buf) in layers.named() {
        assert!(buf.channels.iter().all(|c| c.len() == n), "{name}");
    }
    // Where neither mask is active the output equals the WSOLA baseline exactly.
    let (pm, sm) = (
        &layers.plosive_mask.channels[0],
        &layers.sibilant_mask.channels[0],
    );
    for i in 0..n {
        if pm[i] == 0.0 && sm[i] == 0.0 {
            assert_eq!(out.audio.channels[0][i], layers.baseline.channels[0][i]);
        }
    }
    assert!(pm.contains(&1.0), "no protected burst core");
    assert!(
        sm.iter().any(|&w| w > 0.99),
        "no resynthesized sibilant interior"
    );
}

#[test]
fn thread_count_does_not_change_output_or_report() {
    // A longer utterance so every parallel stage actually splits its work.
    let fx = synth::utterance_fixture(48_000, 7);
    let stereo = synth::stereo_s_fixture(48_000, StereoLayout::PartiallyDecorrelated, 4);
    for audio in [fx.audio, stereo.audio] {
        let refs = audio.view_refs();
        let view = AudioView {
            sample_rate: audio.sample_rate,
            channels: &refs,
        };
        for (f, pitch) in [("0.7", 0.0), ("1.5", 0.0), ("1.25", 3.0)] {
            let r = StretchRatio::parse(f).unwrap();
            let run = |threads| {
                let opts = Options {
                    diagnostics: true,
                    pitch_semitones: pitch,
                    threads,
                    ..Options::default()
                };
                stretch(&view, r, &opts).unwrap()
            };
            let one = run(1);
            for t in [2, 3, 8] {
                let many = run(t);
                for (x, y) in one.audio.channels.iter().zip(&many.audio.channels) {
                    assert!(
                        x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits()),
                        "{f} pitch {pitch}: {t} threads differ from 1"
                    );
                }
                assert_eq!(
                    serde_json::to_string(&one.report).unwrap(),
                    serde_json::to_string(&many.report).unwrap()
                );
            }
        }
    }
}

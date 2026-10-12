//! M1: detector behaviour on deterministic synthetic fixtures.

use sable_core::detect::Detection;
use sable_core::segments::{EventKind, validate_partition};
use sable_core::synth::{self, Fixture, Fricative};
use sable_core::{AudioView, analyze_events};

const RATES: [u32; 5] = [8_000, 16_000, 44_100, 48_000, 96_000];

fn detect(fx: &Fixture) -> Detection {
    let refs = fx.audio.view_refs();
    let view = AudioView {
        sample_rate: fx.audio.sample_rate,
        channels: &refs,
    };
    let (_, det) = analyze_events(&view).unwrap();
    validate_partition(&det.segments, fx.audio.frames() as u64).unwrap();
    det
}

fn bursts(det: &Detection) -> Vec<u64> {
    det.segments
        .iter()
        .filter(|s| s.kind == EventKind::PlosiveBurst)
        .map(|s| s.start)
        .collect()
}

fn overlap(a: (u64, u64), b: (u64, u64)) -> u64 {
    a.1.min(b.1).saturating_sub(a.0.max(b.0))
}

#[test]
fn stop_burst_found_within_one_ms() {
    for fs in RATES {
        let fx = synth::stop_fixture(fs, 1);
        let det = detect(&fx);
        let b = bursts(&det);
        assert_eq!(b.len(), 1, "fs={fs}: bursts {b:?}");
        let truth = fx.truth.bursts[0] as i64;
        let tol = (fs / 1000) as i64;
        assert!(
            (b[0] as i64 - truth).abs() <= tol,
            "fs={fs}: onset {} vs {truth}",
            b[0]
        );
        // Closure before, aspiration after.
        let i = det
            .segments
            .iter()
            .position(|s| s.kind == EventKind::PlosiveBurst)
            .unwrap();
        assert_eq!(
            det.segments[i - 1].kind,
            EventKind::PlosiveClosure,
            "fs={fs}"
        );
        assert_eq!(det.segments[i + 1].kind, EventKind::Aspiration, "fs={fs}");
        // The soft vowel onset is not a burst.
        for &on in &fx.truth.soft_onsets {
            assert!(
                b.iter()
                    .all(|&x| (x as i64 - on as i64).abs() > (fs / 50) as i64)
            );
        }
    }
}

#[test]
fn vowels_are_voiced_and_never_bursts() {
    for fs in RATES {
        for f0 in [80.0, 140.0, 250.0] {
            let fx = synth::vowel_fixture(fs, f0);
            let det = detect(&fx);
            assert!(bursts(&det).is_empty(), "fs={fs} f0={f0}");
            let (a, b, _) = fx.truth.vowels[0];
            let voiced: u64 = det
                .segments
                .iter()
                .filter(|s| s.kind == EventKind::Voiced)
                .map(|s| overlap((s.start, s.end), (a as u64, b as u64)))
                .sum();
            assert!(
                voiced as f64 >= 0.95 * (b - a) as f64,
                "fs={fs} f0={f0}: voiced {voiced}"
            );
        }
    }
}

#[test]
fn fricatives_are_classified() {
    for fs in RATES {
        for (kind, expect) in [
            (Fricative::S, EventKind::SibilantUnvoiced),
            (Fricative::Sh, EventKind::SibilantUnvoiced),
            (Fricative::Z, EventKind::SibilantVoiced),
        ] {
            let fx = synth::fricative_fixture(fs, kind, 3);
            let det = detect(&fx);
            let (a, b, _) = fx.truth.sibilants[0];
            let covered: u64 = det
                .segments
                .iter()
                .filter(|s| s.kind == expect)
                .map(|s| overlap((s.start, s.end), (a as u64, b as u64)))
                .sum();
            assert!(
                covered as f64 >= 0.8 * (b - a) as f64,
                "fs={fs} {kind:?}: covered {covered} of {}",
                b - a
            );
            assert!(bursts(&det).is_empty(), "fs={fs} {kind:?}");
        }
    }
}

#[test]
fn sibilant_next_to_stop_and_edge_bursts() {
    for fs in RATES {
        let fx = synth::sibilant_stop_fixture(fs, 8);
        let det = detect(&fx);
        assert_eq!(bursts(&det).len(), 1, "fs={fs}");
        assert!(
            det.segments
                .iter()
                .any(|s| s.kind == EventKind::SibilantUnvoiced)
        );

        let fx = synth::edge_bursts_fixture(fs, 9);
        let det = detect(&fx);
        let b = bursts(&det);
        let tol = (fs / 1000) as i64;
        for t in &fx.truth.bursts {
            assert!(
                b.iter().any(|&x| (x as i64 - *t as i64).abs() <= tol),
                "fs={fs}: edge burst at {t} not in {b:?}"
            );
        }
    }
}

#[test]
fn whisper_and_hiss_are_conservative() {
    for fs in RATES {
        let det = detect(&synth::whisper_fixture(fs, 10));
        assert!(
            !det.segments.iter().any(|s| s.kind.is_sibilant()),
            "fs={fs} whisper"
        );
        assert!(bursts(&det).is_empty(), "fs={fs} whisper");
        let fx = synth::hiss_fixture(fs, 11);
        let det = detect(&fx);
        assert!(bursts(&det).is_empty(), "fs={fs} hiss");
        // Sibilant still found; background hiss alone is not.
        let (a, b, _) = fx.truth.sibilants[0];
        for s in det.segments.iter().filter(|s| s.kind.is_sibilant()) {
            assert!(
                overlap((s.start, s.end), (a as u64, b as u64)) > 0,
                "fs={fs} hiss sibilant {s:?}"
            );
        }
    }
}

#[test]
fn silence_and_tiny_inputs() {
    let z = vec![0.0f32; 8000];
    let chans = [z.as_slice()];
    let (_, det) = analyze_events(&AudioView {
        sample_rate: 16_000,
        channels: &chans,
    })
    .unwrap();
    assert_eq!(det.segments.len(), 1);
    assert_eq!(det.segments[0].kind, EventKind::Silence);
    for fs in RATES {
        let fx = synth::tiny_fixture(fs);
        detect(&fx);
        let one = [0.5f32];
        let chans = [one.as_slice()];
        let (_, det) = analyze_events(&AudioView {
            sample_rate: fs,
            channels: &chans,
        })
        .unwrap();
        validate_partition(&det.segments, 1).unwrap();
    }
}

#[test]
fn detection_is_deterministic() {
    let fx = synth::utterance_fixture(16_000, 12);
    let a = detect(&fx);
    let b = detect(&fx);
    assert_eq!(a.segments, b.segments);
    assert_eq!(a.burst_candidates, b.burst_candidates);
}

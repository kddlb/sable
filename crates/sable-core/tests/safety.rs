//! Typed errors, identity and silence contracts (spec §2, §11.1).

use sable_core::{AudioView, Options, RenderPath, StretchError, StretchRatio, stretch};

const FACTORS: [&str; 13] = [
    "1/4", "0.5", "0.65", "0.75", "0.9", "1", "1.1", "1.25", "1.5", "1.8", "2", "4", "7/3",
];

fn ratio(s: &str) -> StretchRatio {
    StretchRatio::parse(s).unwrap()
}

fn run(chans: &[&[f32]], fs: u32, r: &str) -> Result<sable_core::ProcessResult, StretchError> {
    stretch(
        &AudioView {
            sample_rate: fs,
            channels: chans,
        },
        ratio(r),
        &Options::default(),
    )
}

#[test]
fn identity_is_sample_exact_including_signed_zero() {
    let mono: Vec<f32> = (0..1000)
        .map(|i| {
            if i % 7 == 0 {
                -0.0
            } else {
                (i as f32 * 0.37).sin()
            }
        })
        .collect();
    let right: Vec<f32> = mono.iter().map(|v| -v * 0.5).collect();
    for chans in [
        vec![mono.as_slice()],
        vec![mono.as_slice(), right.as_slice()],
    ] {
        let out = run(&chans, 44_100, "1").unwrap();
        assert_eq!(out.report.path, RenderPath::Identity);
        for (o, i) in out.audio.channels.iter().zip(&chans) {
            assert_eq!(o.len(), i.len());
            for (a, b) in o.iter().zip(i.iter()) {
                assert_eq!(a.to_bits(), b.to_bits());
            }
        }
    }
    // Identity via a non-reduced fraction also takes the fast path.
    let out = run(&[mono.as_slice()], 44_100, "3/3").unwrap();
    assert_eq!(out.report.path, RenderPath::Identity);
}

#[test]
fn silence_stays_exactly_zero_for_every_factor() {
    let z = vec![0.0f32; 4410];
    for f in FACTORS {
        for chans in [vec![z.as_slice()], vec![z.as_slice(), z.as_slice()]] {
            let out = run(&chans, 44_100, f).unwrap();
            let expect = ratio(f).target_frames(4410).unwrap() as usize;
            for c in &out.audio.channels {
                assert_eq!(c.len(), expect);
                assert!(c.iter().all(|&v| v == 0.0));
            }
        }
    }
}

#[test]
fn empty_input_gives_empty_output() {
    let e: Vec<f32> = vec![];
    for f in FACTORS {
        let out = run(&[e.as_slice(), e.as_slice()], 16_000, f).unwrap();
        assert_eq!(out.audio.channels.len(), 2);
        assert!(out.audio.channels.iter().all(Vec::is_empty));
    }
}

#[test]
fn typed_errors() {
    let x = vec![0.1f32; 100];
    let mut nan = x.clone();
    nan[42] = f32::NAN;
    let mut inf = x.clone();
    inf[7] = f32::NEG_INFINITY;
    let short = vec![0.1f32; 99];

    assert_eq!(
        run(&[nan.as_slice()], 16_000, "2").unwrap_err(),
        StretchError::NonFiniteSample {
            channel: 0,
            index: 42
        }
    );
    assert_eq!(
        run(&[x.as_slice(), inf.as_slice()], 16_000, "2").unwrap_err(),
        StretchError::NonFiniteSample {
            channel: 1,
            index: 7
        }
    );
    assert!(matches!(
        run(&[x.as_slice(), short.as_slice()], 16_000, "2").unwrap_err(),
        StretchError::UnequalChannelLengths { channel: 1, .. }
    ));
    assert_eq!(
        run(&[x.as_slice()], 7_999, "2").unwrap_err(),
        StretchError::UnsupportedSampleRate(7_999)
    );
    assert_eq!(
        run(&[x.as_slice()], 96_001, "2").unwrap_err(),
        StretchError::UnsupportedSampleRate(96_001)
    );
    assert_eq!(
        run(&[x.as_slice(); 3], 16_000, "2").unwrap_err(),
        StretchError::UnsupportedChannelCount(3)
    );
    assert_eq!(
        run(&[], 16_000, "2").unwrap_err(),
        StretchError::UnsupportedChannelCount(0)
    );
    // NaN rejected even on the identity path.
    assert!(run(&[nan.as_slice()], 16_000, "1").is_err());

    // 1 frame * 1/3 rounds to 0.
    let one = [0.5f32];
    assert_eq!(
        run(&[one.as_slice()], 16_000, "1/3").unwrap_err(),
        StretchError::OutputTooShort { n_in: 1 }
    );
    // Output limit.
    let opts = Options {
        max_output_frames: 150,
        ..Options::default()
    };
    let chans = [x.as_slice()];
    let view = AudioView {
        sample_rate: 16_000,
        channels: &chans,
    };
    assert!(matches!(
        stretch(&view, ratio("2"), &opts).unwrap_err(),
        StretchError::OutputTooLarge { .. }
    ));
    // Absurd ratio does not panic.
    let huge = StretchRatio::new(u64::MAX, 1).unwrap();
    assert!(stretch(&view, huge, &Options::default()).is_err());
}

#[test]
fn ratio_errors_are_typed() {
    assert_eq!(
        StretchRatio::new(0, 3).unwrap_err(),
        StretchError::ZeroRatio
    );
    assert_eq!(
        StretchRatio::new(3, 0).unwrap_err(),
        StretchError::ZeroRatio
    );
    assert!(matches!(
        StretchRatio::parse("abc").unwrap_err(),
        StretchError::MalformedRatio(_)
    ));
}

//! Hard invariant: exactly `N_out` frames on every channel, every path (spec §2, §11.1).

use proptest::prelude::*;
use sable_core::synth::{self, FixtureRng};
use sable_core::{AudioView, Options, StretchRatio, stretch};

const FACTORS: [&str; 12] = [
    "0.5", "0.65", "0.75", "0.9", "1", "1.1", "1.25", "1.5", "1.8", "2", "1/4", "4",
];
const RATES: [u32; 5] = [8_000, 16_000, 44_100, 48_000, 96_000];

fn check(chans: &[&[f32]], fs: u32, ratio: StretchRatio, opts: &Options) {
    let n_in = chans[0].len() as u64;
    let expect = ratio.target_frames(n_in).unwrap() as usize;
    let res = stretch(
        &AudioView {
            sample_rate: fs,
            channels: chans,
        },
        ratio,
        opts,
    );
    let out = match res {
        Ok(o) => o,
        Err(sable_core::StretchError::OutputTooShort { .. }) if expect == 0 => return,
        Err(e) => panic!("fs={fs} ratio={ratio} n_in={n_in}: {e}"),
    };
    assert_eq!(out.audio.sample_rate, fs);
    assert_eq!(out.audio.channels.len(), chans.len());
    for c in &out.audio.channels {
        assert_eq!(c.len(), expect, "fs={fs} ratio={ratio} n_in={n_in}");
        assert!(c.iter().all(|v| v.is_finite()));
    }
    assert_eq!(out.report.actual_frames as usize, expect);
}

#[test]
fn fixtures_exact_length_all_rates_and_factors() {
    for fs in RATES {
        let stop = synth::stop_fixture(fs, 1);
        let st = synth::stereo_s_fixture(fs, synth::StereoLayout::Panned, 2);
        for f in FACTORS {
            let r = StretchRatio::parse(f).unwrap();
            for opts in [Options::default(), Options::generic_wsola()] {
                check(&stop.audio.view_refs(), fs, r, &opts);
                check(&st.audio.view_refs(), fs, r, &opts);
            }
        }
    }
}

#[test]
fn edge_cases_exact_length() {
    for fs in [8_000, 48_000] {
        for fx in [
            synth::edge_bursts_fixture(fs, 3),
            synth::tiny_fixture(fs),
            synth::sibilant_stop_fixture(fs, 4),
        ] {
            for f in FACTORS {
                check(
                    &fx.audio.view_refs(),
                    fs,
                    StretchRatio::parse(f).unwrap(),
                    &Options::default(),
                );
            }
        }
    }
    // Very short inputs and odd ratios.
    for n in 1..40usize {
        let x: Vec<f32> = (0..n)
            .map(|i| ((i * 37) % 11) as f32 / 11.0 - 0.5)
            .collect();
        for (p, q) in [(1, 2), (2, 1), (3, 7), (7, 3), (1, 4), (4, 1), (999, 1000)] {
            check(
                &[x.as_slice()],
                8_000,
                StretchRatio::new(p, q).unwrap(),
                &Options::default(),
            );
        }
    }
}

#[test]
fn all_burst_clip_exact_length_with_relaxation() {
    // Ten closely spaced clicks: forces stage B/C at strong stretch.
    let fs = 16_000;
    let mut x = vec![0f32; 1600];
    let mut rng = FixtureRng::new(5);
    for k in 0..10 {
        for i in 0..24 {
            let decay = (-(i as f64) / 6.0).exp();
            x[k * 160 + i] = (rng.normal() * 0.3 * decay) as f32;
        }
    }
    for f in ["2", "4", "1/4", "0.5"] {
        check(
            &[x.as_slice()],
            fs,
            StretchRatio::parse(f).unwrap(),
            &Options::default(),
        );
    }
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 120, ..ProptestConfig::default() })]

    #[test]
    fn random_inputs_exact_length(
        n in 1usize..6000,
        seed in any::<u64>(),
        (p, q) in prop_oneof![
            (1u64..=20, 1u64..=20).prop_filter("quality range", |(p, q)| 2 * p >= *q && *p <= 2 * q),
            Just((1u64, 4u64)),
            Just((4u64, 1u64)),
        ],
        fs_idx in 0usize..5,
        stereo in any::<bool>(),
    ) {
        let fs = RATES[fs_idx];
        let mut rng = FixtureRng::new(seed);
        let l: Vec<f32> = (0..n).map(|_| (rng.normal() * 0.1) as f32).collect();
        let r: Vec<f32> = l.iter().map(|v| v * -0.7).collect();
        let ratio = StretchRatio::new(p, q).unwrap();
        if stereo {
            check(&[l.as_slice(), r.as_slice()], fs, ratio, &Options::default());
        } else {
            check(&[l.as_slice()], fs, ratio, &Options::default());
        }
    }
}

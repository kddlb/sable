//! M4: stochastic sibilant resynthesis (spec §8, §11.1 boundary normalization, §11.2 3–6).

use sable_core::filters::BandSplit;
use sable_core::metrics::{
    band_levels_db, centroid_hz, max_autocorr, mean_coherence, median_f0, welch_psd,
};
use sable_core::report::OverlayReport;
use sable_core::rng::GaussianStream;
use sable_core::segments::EventKind;
use sable_core::sibilants::NoiseSynth;
use sable_core::synth::{self, Fricative, StereoLayout};
use sable_core::windows::sqrt_hann_periodic;
use sable_core::{AudioView, Options, ProcessResult, StretchRatio, stretch};

const FACTORS: [&str; 6] = ["0.5", "0.75", "1.25", "1.5", "1.8", "2"];

fn run(chans: &[&[f32]], fs: u32, f: &str, opts: &Options) -> ProcessResult {
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

fn w2(n: usize) -> f64 {
    sqrt_hann_periodic(n)
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum()
}

/// Source interior and output interior (excluding guards + fades) of the first applied sibilant.
fn interiors(out: &ProcessResult, fs: u32) -> ((usize, usize), (usize, usize), OverlayReport) {
    let ov = out
        .report
        .sibilant_overlays
        .iter()
        .find(|o| o.applied)
        .unwrap_or_else(|| {
            panic!(
                "no sibilant resynthesized: {:?}",
                out.report.sibilant_overlays
            )
        })
        .clone();
    let seg = out
        .report
        .segments
        .iter()
        .find(|s| s.id == ov.segment_id)
        .unwrap();
    let edge_in = (fs as usize) * 25 / 1000;
    let edge_out = (fs as usize) * 25 / 1000;
    (
        (
            seg.in_start as usize + edge_in,
            (seg.in_start + seg.in_len) as usize - edge_in,
        ),
        (
            ov.out_start as usize + edge_out,
            ov.out_end as usize - edge_out,
        ),
        ov,
    )
}

#[test]
fn stationary_white_and_colored_noise_levels() {
    let n_fft = 1024;
    let len = 400_000;
    let sigma2 = 0.01;
    let flat = vec![sigma2 * w2(n_fft); n_fft / 2 + 1];
    let mut synth = NoiseSynth::new(n_fft);
    let mut g = GaussianStream::new(1, 1);
    let y = synth.render(len, 1, |_| (vec![flat.clone()], None), &mut g);
    let x = &y[0];
    let var = x[n_fft..len - n_fft]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        / (len - 2 * n_fft) as f64;
    let db = 10.0 * (var / sigma2).log10();
    assert!(db.abs() < 0.25, "white level error {db:.3} dB");
    // No periodic level modulation at the synthesis hop: average the power in 16
    // phase groups of the hop (≈25k samples each, so chance spread is ≈ ±0.05 dB).
    let hop = n_fft / 4;
    let groups = 16;
    let mut phase = vec![0f64; groups];
    let mut cnt = vec![0usize; groups];
    for (i, &v) in x.iter().enumerate().take(len - n_fft).skip(n_fft) {
        let g = (i % hop) * groups / hop;
        phase[g] += (v as f64).powi(2);
        cnt[g] += 1;
    }
    let lv: Vec<f64> = phase
        .iter()
        .zip(&cnt)
        .map(|(p, &c)| 10.0 * (p / c as f64).log10())
        .collect();
    let (mn, mx) = lv
        .iter()
        .fold((f64::MAX, f64::MIN), |a, &v| (a.0.min(v), a.1.max(v)));
    assert!(mx - mn < 0.25, "hop-periodic modulation {:.2} dB", mx - mn);
    // Coloured: power ∝ 1 / (1 + (k / 64)^2); expected variance = mean of full spectrum.
    let shape: Vec<f64> = (0..=n_fft / 2)
        .map(|k| 1.0 / (1.0 + (k as f64 / 64.0).powi(2)))
        .collect();
    let p: Vec<f64> = shape.iter().map(|s| s * sigma2 * w2(n_fft)).collect();
    let full_mean = (shape[0] + 2.0 * shape[1..n_fft / 2].iter().sum::<f64>() + shape[n_fft / 2])
        / n_fft as f64;
    let y = synth.render(len, 1, |_| (vec![p.clone()], None), &mut g);
    let var = y[0][n_fft..len - n_fft]
        .iter()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        / (len - 2 * n_fft) as f64;
    let db = 10.0 * (var / (sigma2 * full_mean)).log10();
    assert!(db.abs() < 0.3, "colored level error {db:.3} dB");
    // Shape check: low vs high band ratio matches.
    let psd = welch_psd(&y[0], 0, len, n_fft);
    let r_meas = psd[16] / psd[256];
    let r_exp = shape[16] / shape[256];
    assert!((10.0 * (r_meas / r_exp).log10()).abs() < 1.0);
}

#[test]
fn s_spectrum_matches_source() {
    for fs in [16_000, 48_000] {
        let fx = synth::fricative_fixture(fs, Fricative::S, 2);
        let x = &fx.audio.channels[0];
        for f in FACTORS {
            let out = run(&[x.as_slice()], fs, f, &Options::default());
            let ((ia, ib), (oa, ob), _) = interiors(&out, fs);
            let n_fft = 512 * (fs as usize / 16_000).max(1);
            let pin = welch_psd(x, ia, ib, n_fft);
            let pout = welch_psd(&out.audio.channels[0], oa, ob, n_fft);
            let hi = 0.9 * fs as f64 / 2.0;
            let bin = band_levels_db(&pin, fs, 1000.0, hi);
            let bout = band_levels_db(&pout, fs, 1000.0, hi);
            let top = bin.iter().fold(f64::MIN, |a, b| a.max(b.1));
            let mut devs: Vec<f64> = bin
                .iter()
                .zip(&bout)
                .filter(|(a, _)| a.1 > top - 30.0)
                .map(|(a, b)| (a.1 - b.1).abs())
                .collect();
            devs.sort_by(f64::total_cmp);
            let med = devs[devs.len() / 2];
            assert!(
                med <= 3.0,
                "fs={fs} r={f}: median band deviation {med:.2} dB"
            );
        }
    }
}

#[test]
fn sh_keeps_its_own_lower_spectrum() {
    let fs = 44_100;
    for f in ["0.75", "1.5", "2"] {
        let mut cents = Vec::new();
        for kind in [Fricative::S, Fricative::Sh] {
            let fx = synth::fricative_fixture(fs, kind, 3);
            let x = &fx.audio.channels[0];
            let out = run(&[x.as_slice()], fs, f, &Options::default());
            let ((ia, ib), (oa, ob), _) = interiors(&out, fs);
            let cin = centroid_hz(&welch_psd(x, ia, ib, 1024), fs);
            let cout = centroid_hz(&welch_psd(&out.audio.channels[0], oa, ob, 1024), fs);
            assert!(
                (cout - cin).abs() / cin < 0.1,
                "{kind:?} r={f}: centroid {cout:.0} vs {cin:.0}"
            );
            cents.push(cout);
        }
        assert!(
            cents[1] < 0.75 * cents[0],
            "sh {:.0} not below s {:.0}",
            cents[1],
            cents[0]
        );
    }
}

#[test]
fn no_periodic_repetition_in_stretched_sibilant() {
    let fs = 16_000;
    let fx = synth::fricative_fixture(fs, Fricative::S, 4);
    let x = &fx.audio.channels[0];
    for f in ["1.5", "2"] {
        let out = run(&[x.as_slice()], fs, f, &Options::default());
        let (_, (oa, ob), _) = interiors(&out, fs);
        let rep = max_autocorr(
            &out.audio.channels[0],
            oa,
            ob,
            fs as usize / 500,
            fs as usize * 60 / 1000,
        );
        assert!(rep < 0.3, "r={f}: max autocorrelation {rep:.2}");
    }
}

#[test]
fn z_keeps_low_band_voicing() {
    for fs in [16_000, 48_000] {
        let fx = synth::fricative_fixture(fs, Fricative::Z, 5);
        let x = &fx.audio.channels[0];
        for f in ["0.75", "1.5", "2"] {
            let out = run(&[x.as_slice()], fs, f, &Options::default());
            let (_, (oa, ob), ov) = interiors(&out, fs);
            assert!(ov.detail.contains("voiced"), "{}", ov.detail);
            let seg = out
                .report
                .segments
                .iter()
                .find(|s| s.id == ov.segment_id)
                .unwrap();
            assert_eq!(seg.kind, EventKind::SibilantVoiced);
            let lp = BandSplit::new(fs).lowpass(&out.audio.channels[0], 0, out.audio.frames());
            let f0 = median_f0(&lp, fs, oa, ob, 0.5).expect("voicing lost");
            assert!(
                (f0 - 120.0).abs() / 120.0 < 0.02,
                "fs={fs} r={f}: low-band F0 {f0:.1}"
            );
        }
    }
}

#[test]
fn stereo_image_is_preserved() {
    let fs = 32_000;
    for layout in [
        StereoLayout::Coherent,
        StereoLayout::Panned,
        StereoLayout::PartiallyDecorrelated,
    ] {
        let fx = synth::stereo_s_fixture(fs, layout, 6);
        let (l, r) = (&fx.audio.channels[0], &fx.audio.channels[1]);
        let out = run(
            &[l.as_slice(), r.as_slice()],
            fs,
            "1.5",
            &Options::default(),
        );
        let ((ia, ib), (oa, ob), _) = interiors(&out, fs);
        let (ol, or) = (&out.audio.channels[0], &out.audio.channels[1]);
        let e = |x: &[f32], a: usize, b: usize| {
            x[a..b].iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / (b - a) as f64
        };
        let ratio_in = 10.0 * (e(l, ia, ib) / e(r, ia, ib)).log10();
        let ratio_out = 10.0 * (e(ol, oa, ob) / e(or, oa, ob)).log10();
        assert!(
            (ratio_in - ratio_out).abs() < 1.0,
            "{layout:?}: L/R {ratio_in:.2} -> {ratio_out:.2} dB"
        );
        let cin = mean_coherence(l, r, ia, ib, 512, fs, (4000.0, 12000.0));
        let cout = mean_coherence(ol, or, oa, ob, 512, fs, (4000.0, 12000.0));
        assert!(
            (cin - cout).abs() < 0.2,
            "{layout:?}: coherence {cin:.2} -> {cout:.2}"
        );
    }
}

#[test]
fn seed_changes_only_the_noise() {
    let fs = 16_000;
    let fx = synth::fricative_fixture(fs, Fricative::S, 7);
    let x = &fx.audio.channels[0];
    let a = run(&[x.as_slice()], fs, "1.5", &Options::default());
    let b = run(&[x.as_slice()], fs, "1.5", &Options::default());
    assert_eq!(a.audio, b.audio);
    let c = run(
        &[x.as_slice()],
        fs,
        "1.5",
        &Options {
            seed: 99,
            ..Options::default()
        },
    );
    let ov = a
        .report
        .sibilant_overlays
        .iter()
        .find(|o| o.applied)
        .unwrap();
    let (s, e) = (ov.out_start as usize, ov.out_end as usize);
    let ya = &a.audio.channels[0];
    let yc = &c.audio.channels[0];
    assert_eq!(&ya[..s], &yc[..s]);
    assert_eq!(&ya[e..], &yc[e..]);
    assert_ne!(&ya[s..e], &yc[s..e]);
}

#[test]
fn short_or_disabled_sibilants_keep_wsola() {
    let fs = 16_000;
    let fx = synth::fricative_fixture(fs, Fricative::S, 8);
    let x = &fx.audio.channels[0];
    let opts = Options {
        enable_sibilant: false,
        ..Options::default()
    };
    let out = run(&[x.as_slice()], fs, "1.5", &opts);
    assert!(out.report.sibilant_overlays.iter().all(|o| !o.applied));
}

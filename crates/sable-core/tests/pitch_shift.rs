//! Pitch shifting: exact length, F0 moves by 2^(st/12), and the spectral envelope stays
//! (FormantMode::Preserve) or moves with the pitch (FormantMode::Shift).

use sable_core::fft::RealFft;
use sable_core::windows::hann_periodic;
use sable_core::{
    AudioView, FormantMode, Options, Phase, ProgressEvent, RenderPath, StretchError, StretchRatio,
    stretch, stretch_with_progress,
};

const FS: u32 = 16_000;
const F0: f64 = 150.0;
/// Centre of the Gaussian "formant" envelope of the synthetic vowel.
const FORMANT: f64 = 900.0;

/// 1.2 s harmonic vowel with a slight vibrato and a single smooth envelope peak.
fn vowel() -> Vec<f32> {
    let n = (1.2 * FS as f64) as usize;
    let mut phase = 0.0f64;
    (0..n)
        .map(|i| {
            let t = i as f64 / FS as f64;
            let f0 = F0 * (1.0 + 0.01 * (std::f64::consts::TAU * 5.0 * t).sin());
            phase += std::f64::consts::TAU * f0 / FS as f64;
            let mut v = 0.0;
            for k in 1..=((0.45 * FS as f64) / F0) as usize {
                let fk = k as f64 * F0;
                let a = (-((fk - FORMANT) / 300.0).powi(2)).exp() + 0.02;
                v += a * (k as f64 * phase).sin();
            }
            (0.08 * v) as f32
        })
        .collect()
}

fn run(x: &[f32], ratio: &str, st: f64, formants: FormantMode) -> sable_core::ProcessResult {
    let chans = [x];
    let view = AudioView {
        sample_rate: FS,
        channels: &chans,
    };
    let opts = Options {
        pitch_semitones: st,
        formants,
        ..Options::default()
    };
    stretch(&view, StretchRatio::parse(ratio).unwrap(), &opts).unwrap()
}

/// Magnitude spectrum (bin width FS / 4096) of a Hann-windowed block at the middle.
fn mid_spectrum(y: &[f32]) -> Vec<f32> {
    let n = 4096;
    let a = y.len() / 2 - n / 2;
    let mut fft = RealFft::new(n);
    fft.forward_windowed(&y[a..a + n], &hann_periodic(n))
        .iter()
        .map(|c| c.norm())
        .collect()
}

fn hz(bin: usize) -> f64 {
    bin as f64 * FS as f64 / 4096.0
}

/// Strongest peak frequency between `lo` and `hi` Hz.
fn peak_hz(spec: &[f32], lo: f64, hi: f64) -> f64 {
    let (a, b) = (
        (lo / hz(1)) as usize,
        ((hi / hz(1)) as usize).min(spec.len() - 1),
    );
    let k = (a..=b)
        .max_by(|&i, &j| spec[i].total_cmp(&spec[j]))
        .unwrap();
    hz(k)
}

/// F0 from the autocorrelation of a mid block: the first lag (60–500 Hz range) whose
/// normalized ACF is within 10 % of the best one, which avoids octave-down errors.
fn f0_hz(y: &[f32]) -> f64 {
    let n = 4096;
    let x = &y[y.len() / 2 - n / 2..][..n];
    let acf = |lag: usize| -> f64 {
        let (mut num, mut e0, mut e1) = (0.0, 0.0, 0.0);
        for i in 0..n - lag {
            let (a, b) = (x[i] as f64, x[i + lag] as f64);
            num += a * b;
            e0 += a * a;
            e1 += b * b;
        }
        num / (e0 * e1).sqrt().max(1e-12)
    };
    let lags: Vec<(usize, f64)> = (FS as usize / 500..=FS as usize / 60)
        .map(|l| (l, acf(l)))
        .collect();
    let best = lags.iter().fold(f64::MIN, |m, &(_, v)| m.max(v));
    // First local maximum close to the best value, refined parabolically.
    let i = (1..lags.len() - 1)
        .find(|&i| {
            lags[i].1 >= 0.9 * best && lags[i].1 >= lags[i - 1].1 && lags[i].1 >= lags[i + 1].1
        })
        .unwrap();
    let (a, b, c) = (lags[i - 1].1, lags[i].1, lags[i + 1].1);
    let d = 0.5 * (a - c) / (a - 2.0 * b + c);
    FS as f64 / (lags[i].0 as f64 + d)
}

/// Energy-weighted centroid of the 200–3000 Hz band.
fn centroid_hz(spec: &[f32]) -> f64 {
    let (mut num, mut den) = (0.0, 0.0);
    for (k, &m) in spec.iter().enumerate() {
        let f = hz(k);
        if (200.0..3000.0).contains(&f) {
            let p = (m as f64).powi(2);
            num += f * p;
            den += p;
        }
    }
    num / den
}

#[test]
fn exact_length_for_every_combination() {
    let x = vowel();
    for ratio in ["1", "3/4", "1.5", "2"] {
        for st in [-12.0, -7.0, -0.5, 3.0, 12.0] {
            for fm in [FormantMode::Preserve, FormantMode::Shift] {
                let r = StretchRatio::parse(ratio).unwrap();
                let out = run(&x, ratio, st, fm);
                let want = r.target_frames(x.len() as u64).unwrap();
                assert_eq!(
                    out.audio.channels[0].len() as u64,
                    want,
                    "{ratio} {st} {fm:?}"
                );
                assert_eq!(out.report.actual_frames, want);
                let p = out.report.pitch.as_ref().unwrap();
                assert_eq!(p.formant.is_some(), fm == FormantMode::Preserve);
                let last = out.report.segments.last().unwrap();
                assert_eq!(
                    last.out_start + last.out_len,
                    want,
                    "segment table ends at N_out"
                );
            }
        }
    }
}

#[test]
fn f0_scales_by_semitone_factor() {
    let x = vowel();
    let f_in = f0_hz(&x);
    assert!((f_in / F0 - 1.0).abs() < 0.02, "input F0 {f_in}");
    for st in [-5.0, 4.0, 7.0] {
        for fm in [FormantMode::Preserve, FormantMode::Shift] {
            let y = run(&x, "1", st, fm).audio.channels.remove(0);
            let f_out = f0_hz(&y);
            let want = f_in * (st / 12.0f64).exp2();
            assert!(
                (f_out / want - 1.0).abs() < 0.02,
                "{st} st {fm:?}: F0 {f_out:.1} Hz, expected {want:.1}"
            );
        }
    }
}

#[test]
fn formants_stay_or_move() {
    let x = vowel();
    let c_in = centroid_hz(&mid_spectrum(&x));
    let p_in = peak_hz(&mid_spectrum(&x), 400.0, 2500.0);
    assert!((p_in - FORMANT).abs() < 60.0);
    for st in [-4.0, 5.0] {
        let s = (st / 12.0f64).exp2();
        let keep = mid_spectrum(&run(&x, "1.25", st, FormantMode::Preserve).audio.channels[0]);
        let shift = mid_spectrum(&run(&x, "1.25", st, FormantMode::Shift).audio.channels[0]);
        let (ck, cs) = (centroid_hz(&keep), centroid_hz(&shift));
        // Preserve: envelope centroid within 8 % of the input's.
        assert!(
            (ck / c_in - 1.0).abs() < 0.08,
            "{st} st preserve: centroid {ck:.0} vs input {c_in:.0}"
        );
        // Shift: centroid follows the pitch factor.
        assert!(
            (cs / (c_in * s) - 1.0).abs() < 0.08,
            "{st} st shift: centroid {cs:.0}, expected {:.0}",
            c_in * s
        );
        // The strongest harmonic stays near the formant only when preserving.
        let pk = peak_hz(&keep, 400.0, 2500.0);
        let half_spacing = 0.5 * F0 * s + 20.0;
        assert!(
            (pk - FORMANT).abs() <= half_spacing,
            "{st} st preserve: peak {pk:.0} Hz"
        );
        let ps = peak_hz(&shift, 400.0, 2500.0);
        assert!(
            (ps - FORMANT * s).abs() <= half_spacing,
            "{st} st shift: peak {ps:.0} Hz, expected {:.0}",
            FORMANT * s
        );
    }
}

#[test]
fn zero_semitones_is_the_plain_stretch() {
    let x = vowel();
    let plain = run(&x, "1", 0.0, FormantMode::Preserve);
    assert_eq!(plain.report.path, RenderPath::Identity);
    assert!(plain.report.pitch.is_none());
    assert_eq!(plain.audio.channels[0], x);
    // A shift at ratio 1 is not the identity path.
    let shifted = run(&x, "1", 2.0, FormantMode::Preserve);
    assert_ne!(shifted.report.path, RenderPath::Identity);
}

#[test]
fn silence_and_determinism() {
    let z = vec![0f32; 9000];
    let out = run(&z, "1.5", 7.0, FormantMode::Preserve);
    assert_eq!(out.report.path, RenderPath::Silence);
    assert!(out.audio.channels[0].iter().all(|&v| v == 0.0));
    let x = vowel();
    let a = run(&x, "0.8", -3.0, FormantMode::Preserve);
    let b = run(&x, "0.8", -3.0, FormantMode::Preserve);
    assert_eq!(a.audio.channels, b.audio.channels);
}

#[test]
fn invalid_pitch_is_typed() {
    let x = vowel();
    let chans = [x.as_slice()];
    let view = AudioView {
        sample_rate: FS,
        channels: &chans,
    };
    for st in [f64::NAN, f64::INFINITY, 24.5, -30.0] {
        let opts = Options {
            pitch_semitones: st,
            ..Options::default()
        };
        assert!(matches!(
            stretch(&view, StretchRatio::parse("1").unwrap(), &opts),
            Err(StretchError::InvalidPitch(_))
        ));
    }
}

#[test]
fn progress_reports_every_phase_and_changes_nothing() {
    let x = vowel();
    let chans = [x.as_slice(), x.as_slice()];
    let view = AudioView {
        sample_rate: FS,
        channels: &chans,
    };
    let opts = Options {
        pitch_semitones: 3.0,
        keep_layers: true,
        ..Options::default()
    };
    let r = StretchRatio::parse("1.3").unwrap();
    let mut events = Vec::new();
    let a = stretch_with_progress(&view, r, &opts, &mut |e| events.push(e)).unwrap();
    let b = stretch(&view, r, &opts).unwrap();
    assert_eq!(a.audio.channels, b.audio.channels);
    let finished: Vec<Phase> = events
        .iter()
        .filter_map(|e| match e {
            ProgressEvent::Finished { phase, .. } => Some(*phase),
            _ => None,
        })
        .collect();
    assert_eq!(finished, Phase::ALL.to_vec());
    for e in &events {
        if let ProgressEvent::Progress { fraction, .. } = e {
            assert!((0.0..=1.0).contains(fraction));
        }
    }
    // Layers are resampled to the output length too.
    let n = a.report.n_out as usize;
    for (name, buf) in a.layers.as_ref().unwrap().named() {
        assert!(buf.channels.iter().all(|c| c.len() == n), "{name}");
    }
}

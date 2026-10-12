//! Evaluates experimental pitch-period blending (`Options::period_blend`) for strong
//! slow-downs against the plain renderer: share of blended grains, period-to-period
//! similarity swing, high-band balance and level, plus an A/B listening set.
//!
//! The swing is only a weak indicator: a smooth period-to-period morph and a verbatim
//! repeat both score as "similar", so listening decides.
//!
//! ```text
//! cargo run --release -p sable-core --example blend_eval [-- out_dir [extra.wav ...]]
//! ```
//!
//! Inputs: the local recordings (see `common::speech_paths`), any extra WAVs given on
//! the command line, and a synthetic utterance. Output WAVs go to `ab/blend/` by default.

mod common;

use std::path::PathBuf;

use sable_core::metrics::{period_similarity, welch_psd};
use sable_core::synth;
use sable_core::{AudioBuffer, AudioView, Options, StretchRatio, stretch};

const FACTORS: [&str; 4] = ["2", "2.5", "3", "4"];

/// Mean frame-to-frame change of the period-to-period similarity of `x`.
fn swing(x: &[f32], fs: u32) -> f64 {
    let s = period_similarity(x, fs);
    if s.len() < 2 {
        return 0.0;
    }
    s.windows(2).map(|w| (w[1] - w[0]).abs()).sum::<f64>() / (s.len() - 1) as f64
}

/// Level of 3–8 kHz relative to 100 Hz–1 kHz, in dB.
fn hf_balance_db(x: &[f32], fs: u32) -> f64 {
    let n_fft = 2048;
    let psd = welch_psd(x, 0, x.len(), n_fft);
    let band = |lo: f64, hi: f64| {
        psd.iter()
            .enumerate()
            .filter(|(k, _)| {
                let f = *k as f64 * fs as f64 / n_fft as f64;
                f >= lo && f < hi
            })
            .map(|(_, p)| p)
            .sum::<f64>()
    };
    10.0 * ((band(3000.0, 8000.0) + 1e-30) / (band(100.0, 1000.0) + 1e-30)).log10()
}

fn rms_db(x: &[f32]) -> f64 {
    let ms = x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64;
    10.0 * (ms + 1e-30).log10()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let dir = args
        .next()
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../ab/blend")));
    std::fs::create_dir_all(&dir).unwrap();
    let mut items: Vec<(String, AudioBuffer)> = Vec::new();
    for (i, p) in common::speech_paths().iter().enumerate() {
        items.push((format!("speech{}", i + 1), common::read_wav(p).unwrap()));
    }
    for (i, p) in args.enumerate() {
        items.push((
            format!("extra{}", i + 1),
            common::read_wav(p.as_ref()).unwrap(),
        ));
    }
    let fx = synth::utterance_fixture(48_000, 1);
    items.push((format!("synth_{}", fx.name), fx.audio));

    println!(
        "| input | r | blended grains | similarity swing off → on | HF balance change off / on (dB) | level change off / on (dB) |"
    );
    println!("|---|---:|---:|---:|---:|---:|");
    for (name, audio) in &items {
        let fs = audio.sample_rate;
        let x0 = &audio.channels[0];
        let (hf0, lv0) = (hf_balance_db(x0, fs), rms_db(x0));
        println!("| {name} | 1 (source) | – | {:.3} | – | – |", swing(x0, fs));
        let refs = audio.view_refs();
        let view = AudioView {
            sample_rate: fs,
            channels: &refs,
        };
        for f in FACTORS {
            let r = StretchRatio::parse(f).unwrap();
            let off = stretch(&view, r, &Options::default()).unwrap();
            let on = stretch(
                &view,
                r,
                &Options {
                    period_blend: true,
                    ..Options::default()
                },
            )
            .unwrap();
            let (y0, y1) = (&off.audio.channels[0], &on.audio.channels[0]);
            let (sa, sb) = (swing(y0, fs), swing(y1, fs));
            let w = on.report.wsola.clone().unwrap_or_default();
            println!(
                "| {name} | {f} | {:.0} % | {sa:.3} → {sb:.3} | {:+.1} / {:+.1} | {:+.2} / {:+.2} |",
                100.0 * w.blended_grains as f64 / w.grains.max(1) as f64,
                hf_balance_db(y0, fs) - hf0,
                hf_balance_db(y1, fs) - hf0,
                rms_db(y0) - lv0,
                rms_db(y1) - lv0,
            );
            let tag = f.replace('.', "_");
            common::write_wav(&dir.join(format!("{name}_x{tag}_plain.wav")), &off.audio);
            common::write_wav(&dir.join(format!("{name}_x{tag}_blend.wav")), &on.audio);
        }
    }
    println!("\nA/B files written to {}", dir.display());
}

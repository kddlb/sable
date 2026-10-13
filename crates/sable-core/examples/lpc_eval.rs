//! Evaluates experimental LPC-residual rendering (`Options::lpc_residual`) against the
//! plain renderer: how closely the output's spectral envelope follows the input's at the
//! time-mapped position, how much it jitters between neighbouring frames, high-band
//! balance, level and run time, plus an A/B listening set.
//!
//! The envelope is measured independently of LPC: third-octave band levels (100 Hz to
//! 8 kHz) of a 25 ms Hann frame. "Tracking" is the RMS difference (dB) between the output
//! frame at `y` and the input frame at `U(y)`, over frames within 30 dB of the loudest;
//! `U` is rebuilt piecewise-linearly from the report's segment table (burst warps are
//! ignored). "Jitter" is the mean absolute band-level change between output frames 5 ms
//! apart. Lower is better for both, but listening decides.
//!
//! ```text
//! cargo run --release -p sable-core --example lpc_eval [-- out_dir [extra.wav ...]]
//! ```
//!
//! Inputs: the local recordings (see `common::speech_paths`), any extra WAVs given on
//! the command line, and a synthetic utterance. Output WAVs go to `ab/lpc/` by default.

mod common;

use std::path::PathBuf;
use std::time::Instant;

use sable_core::metrics::{band_levels_db, welch_psd};
use sable_core::report::SegmentReport;
use sable_core::synth;
use sable_core::{AudioBuffer, AudioView, Options, StretchRatio, stretch};

const FACTORS: [&str; 4] = ["1.5", "2", "3", "0.75"];

fn n_fft(fs: u32) -> usize {
    sable_core::windows::pow2_near_ms(25.0, fs)
}

/// Third-octave band levels of the frame centred on `c`, or `None` near the edges.
fn bands(x: &[f32], c: usize, fs: u32) -> Option<Vec<f64>> {
    let n = n_fft(fs);
    let a = c.checked_sub(n / 2)?;
    if a + n > x.len() {
        return None;
    }
    let psd = welch_psd(x, a, a + n, n);
    let hi = 8000f64.min(0.45 * fs as f64);
    Some(
        band_levels_db(&psd, fs, 100.0, hi)
            .into_iter()
            .map(|(_, d)| d)
            .collect(),
    )
}

fn level(b: &[f64]) -> f64 {
    10.0 * b.iter().map(|d| 10f64.powf(d / 10.0)).sum::<f64>().log10()
}

fn rms_diff(a: &[f64], b: &[f64]) -> f64 {
    (a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / a.len().max(1) as f64).sqrt()
}

/// Input position for output frame `y`, piecewise-linear over the segment table.
fn inverse(segs: &[SegmentReport], y: usize) -> f64 {
    let k = segs
        .partition_point(|s| s.out_start as usize <= y)
        .saturating_sub(1);
    let s = &segs[k];
    let w = (y as f64 - s.out_start as f64) / s.out_len.max(1) as f64;
    s.in_start as f64 + w * s.in_len as f64
}

/// (tracking dB, jitter dB) of output `y` against input `x`.
fn envelope_scores(x: &[f32], y: &[f32], segs: &[SegmentReport], fs: u32) -> (f64, f64) {
    let hop = (fs / 200) as usize;
    let frames: Vec<(usize, Vec<f64>)> = (0..y.len())
        .step_by(hop)
        .filter_map(|c| bands(y, c, fs).map(|b| (c, b)))
        .collect();
    let loud = frames
        .iter()
        .map(|(_, b)| level(b))
        .fold(f64::MIN, f64::max);
    let (mut track, mut nt, mut jit, mut nj) = (0.0, 0usize, 0.0, 0usize);
    for (i, (c, b)) in frames.iter().enumerate() {
        if level(b) < loud - 30.0 {
            continue;
        }
        let u = inverse(segs, *c).round() as usize;
        if let Some(bx) = bands(x, u, fs) {
            track += rms_diff(b, &bx).powi(2);
            nt += 1;
        }
        if i > 0 && level(&frames[i - 1].1) >= loud - 30.0 {
            let d = &frames[i - 1].1;
            jit += b.iter().zip(d).map(|(p, q)| (p - q).abs()).sum::<f64>() / b.len() as f64;
            nj += 1;
        }
    }
    ((track / nt.max(1) as f64).sqrt(), jit / nj.max(1) as f64)
}

/// Level of 3–8 kHz relative to 100 Hz–1 kHz, in dB.
fn hf_balance_db(x: &[f32], fs: u32) -> f64 {
    let n = 2048;
    let psd = welch_psd(x, 0, x.len(), n);
    let band = |lo: f64, hi: f64| {
        psd.iter()
            .enumerate()
            .filter(|(k, _)| {
                let f = *k as f64 * fs as f64 / n as f64;
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
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../ab/lpc")));
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
        "| input | r | tracking off → on (dB) | jitter off → on (dB) | HF balance change off / on (dB) | level change off / on (dB) | time off / on (ms) |"
    );
    println!("|---|---:|---:|---:|---:|---:|---:|");
    for (name, audio) in &items {
        let fs = audio.sample_rate;
        let x0 = &audio.channels[0];
        let (hf0, lv0) = (hf_balance_db(x0, fs), rms_db(x0));
        let refs = audio.view_refs();
        let view = AudioView {
            sample_rate: fs,
            channels: &refs,
        };
        for f in FACTORS {
            let r = StretchRatio::parse(f).unwrap();
            let t0 = Instant::now();
            let off = stretch(&view, r, &Options::default()).unwrap();
            let t1 = Instant::now();
            let on = stretch(
                &view,
                r,
                &Options {
                    lpc_residual: true,
                    ..Options::default()
                },
            )
            .unwrap();
            let t2 = Instant::now();
            let (y0, y1) = (&off.audio.channels[0], &on.audio.channels[0]);
            let (ta, ja) = envelope_scores(x0, y0, &off.report.segments, fs);
            let (tb, jb) = envelope_scores(x0, y1, &on.report.segments, fs);
            println!(
                "| {name} | {f} | {ta:.2} → {tb:.2} | {ja:.2} → {jb:.2} | {:+.1} / {:+.1} | {:+.2} / {:+.2} | {:.0} / {:.0} |",
                hf_balance_db(y0, fs) - hf0,
                hf_balance_db(y1, fs) - hf0,
                rms_db(y0) - lv0,
                rms_db(y1) - lv0,
                (t1 - t0).as_secs_f64() * 1e3,
                (t2 - t1).as_secs_f64() * 1e3,
            );
            if !on.report.warnings.is_empty() {
                let lpc_fb = on
                    .report
                    .warnings
                    .iter()
                    .any(|w| matches!(w, sable_core::Warning::LpcResidualFallback));
                if lpc_fb {
                    println!("|  | (LPC fell back to plain WSOLA) | | | | | |");
                }
            }
            let tag = f.replace('.', "_");
            common::write_wav(&dir.join(format!("{name}_x{tag}_plain.wav")), &off.audio);
            common::write_wav(&dir.join(format!("{name}_x{tag}_lpc.wav")), &on.audio);
        }
    }
    println!("\nA/B files written to {}", dir.display());
}

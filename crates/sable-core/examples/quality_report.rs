//! Objective comparison of SABLE against generic uniform WSOLA on synthetic fixtures
//! (and the local speech recordings, if present). Prints Markdown; with `--write`, it
//! replaces the generated section of `QUALITY.md`.
//!
//! ```text
//! cargo run --release -p sable-core --example quality_report -- --write
//! ```

mod common;

use std::fmt::Write as _;
use std::time::Instant;

use sable_core::filters::BandSplit;
use sable_core::metrics::{
    band_levels_db, centroid_hz, max_autocorr, mean_coherence, median_f0, release_peaks,
    transient_env, welch_psd,
};
use sable_core::segments::EventKind;
use sable_core::synth::{self, Fixture, Fricative, StereoLayout};
use sable_core::{AudioView, Options, ProcessResult, StretchRatio, stretch};

const FACTORS: [&str; 10] = [
    "0.5", "0.65", "0.75", "0.9", "1.1", "1.25", "1.5", "1.8", "2", "1/4",
];

fn run(fx: &Fixture, f: &str, opts: &Options) -> ProcessResult {
    let refs = fx.audio.view_refs();
    stretch(
        &AudioView {
            sample_rate: fx.audio.sample_rate,
            channels: &refs,
        },
        StretchRatio::parse(f).unwrap(),
        opts,
    )
    .unwrap()
}

fn ratio(f: &str) -> f64 {
    StretchRatio::parse(f).unwrap().as_f64()
}

fn sable() -> Options {
    Options::default()
}

fn generic() -> Options {
    Options::generic_wsola()
}

/// Median absolute third-octave level deviation over populated bands (dB).
fn band_dev(
    x: &[f32],
    (ia, ib): (usize, usize),
    y: &[f32],
    (oa, ob): (usize, usize),
    fs: u32,
) -> f64 {
    let n = 512 * (fs as usize / 16_000).max(1);
    let pin = welch_psd(x, ia, ib, n);
    let pout = welch_psd(y, oa, ob, n);
    let hi = 0.9 * fs as f64 / 2.0;
    let a = band_levels_db(&pin, fs, 1000.0, hi);
    let b = band_levels_db(&pout, fs, 1000.0, hi);
    let top = a.iter().fold(f64::MIN, |m, v| m.max(v.1));
    let mut d: Vec<f64> = a
        .iter()
        .zip(&b)
        .filter(|(p, _)| p.1 > top - 30.0)
        .map(|(p, q)| (p.1 - q.1).abs())
        .collect();
    d.sort_by(f64::total_cmp);
    d[d.len() / 2]
}

fn pitch_section(md: &mut String) {
    writeln!(
        md,
        "### Pitch preservation (harmonic vowel, steady portion)\n"
    )
    .unwrap();
    writeln!(md, "Max |F0 error| over factors 0.5–2.0 (excluding 1/4).\n").unwrap();
    writeln!(
        md,
        "| fs | F0 | SABLE max error | generic WSOLA max error |"
    )
    .unwrap();
    writeln!(md, "|---:|---:|---:|---:|").unwrap();
    for fs in [16_000, 48_000] {
        for f0 in [80.0, 140.0, 250.0] {
            let fx = synth::vowel_fixture(fs, f0);
            let (a, b, _) = fx.truth.vowels[0];
            let mut errs = [0f64; 2];
            for f in FACTORS.iter().filter(|f| **f != "1/4") {
                let r = ratio(f);
                for (k, o) in [sable(), generic()].iter().enumerate() {
                    let out = run(&fx, f, o);
                    let est = median_f0(
                        &out.audio.channels[0],
                        fs,
                        (a as f64 * r) as usize,
                        (b as f64 * r) as usize,
                        0.6,
                    )
                    .unwrap_or(0.0);
                    errs[k] = errs[k].max((est - f0).abs() / f0 * 100.0);
                }
            }
            writeln!(
                md,
                "| {fs} | {f0} Hz | {:.2} % | {:.2} % |",
                errs[0], errs[1]
            )
            .unwrap();
        }
    }
    md.push('\n');
}

fn plosive_section(md: &mut String) {
    writeln!(
        md,
        "### Plosive release (synthetic stop: closure, 6 ms burst, aspiration, vowel)\n"
    )
    .unwrap();
    writeln!(
        md,
        "*Peaks* = release peaks ≥ 50 % of the main one inside the stop's closure–aspiration span \
         (max ±40 ms). *Core* = burst samples copied bit-exactly. *Onset err* = |main peak − mapped onset|.\n"
    )
    .unwrap();
    writeln!(md, "| fs | r | SABLE peaks | SABLE core exact | SABLE onset err | generic peaks | generic onset err |").unwrap();
    writeln!(md, "|---:|---:|---:|:---:|---:|---:|---:|").unwrap();
    for fs in [16_000, 48_000] {
        let fx = synth::stop_fixture(fs, 2);
        let x = &fx.audio.channels[0];
        let sep = fs as usize * 3 / 1000;
        let radius = fs as usize * 40 / 1000;
        let ms = |v: i64| v as f64 * 1000.0 / fs as f64;
        for f in FACTORS {
            let r = ratio(f);
            let s = run(&fx, f, &sable());
            let segs = &s.report.segments;
            let Some(bi) = segs.iter().position(|v| v.kind == EventKind::PlosiveBurst) else {
                writeln!(md, "| {fs} | {f} | no burst | | | | |").unwrap();
                continue;
            };
            let b = &segs[bi];
            let (wa, wb) = (
                segs[bi - 1].out_start as usize,
                (segs[bi + 1].out_start + segs[bi + 1].out_len) as usize,
            );
            let o = b.out_start as usize;
            let env = transient_env(&s.audio.channels[0], fs);
            let (cnt, main) = release_peaks(&env, o, radius.min(o - wa).min(wb - o), sep, 0.5);
            let exact = (0..b.in_len as usize)
                .all(|k| s.audio.channels[0][o + k] == x[b.in_start as usize + k]);
            // Generic: uniform map, same phase span scaled by r.
            let g = run(&fx, f, &generic());
            let go = (b.in_start as f64 * r).round() as usize;
            let (ga, gb) = (
                (segs[bi - 1].in_start as f64 * r) as usize,
                ((segs[bi + 1].in_start + segs[bi + 1].in_len) as f64 * r) as usize,
            );
            let genv = transient_env(&g.audio.channels[0], fs);
            let (gcnt, gmain) =
                release_peaks(&genv, go, radius.min(go - ga).min(gb - go), sep, 0.5);
            writeln!(
                md,
                "| {fs} | {f} | {cnt} | {} | {:.2} ms | {gcnt} | {:.2} ms |",
                if exact { "yes" } else { "no" },
                ms((main as i64 - o as i64).abs()),
                ms((gmain as i64 - go as i64).abs())
            )
            .unwrap();
        }
    }
    md.push('\n');
}

fn sibilant_section(md: &mut String) {
    writeln!(
        md,
        "### Sibilants (vowel–fricative–vowel, 220 ms fricative)\n"
    )
    .unwrap();
    writeln!(
        md,
        "*Band dev* = median |third-octave level difference| between source and output interiors over \
         populated bands. *Repeat* = max normalized autocorrelation of the output interior at lags \
         2–60 ms (grain repetition shows up as a high value). *Centroid err* relative to source.\n"
    )
    .unwrap();
    writeln!(
        md,
        "| sound | fs | r | SABLE band dev | generic band dev | SABLE repeat | generic repeat | SABLE centroid err |"
    )
    .unwrap();
    writeln!(md, "|---|---:|---:|---:|---:|---:|---:|---:|").unwrap();
    for (name, kind) in [("/s/", Fricative::S), ("/ʃ/", Fricative::Sh)] {
        for fs in [16_000, 44_100] {
            let fx = synth::fricative_fixture(fs, kind, 3);
            let x = &fx.audio.channels[0];
            let (ts, te, _) = fx.truth.sibilants[0];
            let edge = fs as usize * 25 / 1000;
            for f in ["0.5", "0.75", "1.25", "1.5", "2"] {
                let r = ratio(f);
                let s = run(&fx, f, &sable());
                let Some(ov) = s.report.sibilant_overlays.iter().find(|o| o.applied) else {
                    writeln!(md, "| {name} | {fs} | {f} | not resynthesized | | | | |").unwrap();
                    continue;
                };
                let seg = s
                    .report
                    .segments
                    .iter()
                    .find(|v| v.id == ov.segment_id)
                    .unwrap();
                let src = (
                    seg.in_start as usize + edge,
                    (seg.in_start + seg.in_len) as usize - edge,
                );
                let so = (ov.out_start as usize + edge, ov.out_end as usize - edge);
                let go = (
                    ((ts + edge) as f64 * r) as usize,
                    ((te - edge) as f64 * r) as usize,
                );
                let srcg = (ts + edge, te - edge);
                let g = run(&fx, f, &generic());
                let lags = (fs as usize / 500, fs as usize * 60 / 1000);
                let n = 1024;
                let cin = centroid_hz(&welch_psd(x, src.0, src.1, n), fs);
                let cout = centroid_hz(&welch_psd(&s.audio.channels[0], so.0, so.1, n), fs);
                writeln!(
                    md,
                    "| {name} | {fs} | {f} | {:.2} dB | {:.2} dB | {:.2} | {:.2} | {:.1} % |",
                    band_dev(x, src, &s.audio.channels[0], so, fs),
                    band_dev(x, srcg, &g.audio.channels[0], go, fs),
                    max_autocorr(&s.audio.channels[0], so.0, so.1, lags.0, lags.1),
                    max_autocorr(&g.audio.channels[0], go.0, go.1, lags.0, lags.1),
                    (cout - cin).abs() / cin * 100.0
                )
                .unwrap();
            }
        }
    }
    md.push('\n');
    writeln!(
        md,
        "#### Voiced /z/: low-band F0 after resynthesis of the high band\n"
    )
    .unwrap();
    writeln!(
        md,
        "| fs | r | low-band F0 (target 120 Hz) | high band resynthesized |"
    )
    .unwrap();
    writeln!(md, "|---:|---:|---:|:---:|").unwrap();
    for fs in [16_000, 48_000] {
        let fx = synth::fricative_fixture(fs, Fricative::Z, 5);
        for f in ["0.75", "1.5", "2"] {
            let s = run(&fx, f, &sable());
            let ov = s.report.sibilant_overlays.iter().find(|o| o.applied);
            let (a, b) = ov.map_or((0, 0), |o| {
                (
                    o.out_start as usize + fs as usize / 40,
                    o.out_end as usize - fs as usize / 40,
                )
            });
            let lp = BandSplit::new(fs).lowpass(&s.audio.channels[0], 0, s.audio.frames());
            let f0 = median_f0(&lp, fs, a, b, 0.5).unwrap_or(0.0);
            writeln!(
                md,
                "| {fs} | {f} | {f0:.1} Hz | {} |",
                if ov.is_some_and(|o| o.detail.contains("voiced")) {
                    "yes"
                } else {
                    "no"
                }
            )
            .unwrap();
        }
    }
    md.push('\n');
}

fn stereo_section(md: &mut String) {
    writeln!(md, "### Stereo sibilance (r = 1.5, 32 kHz)\n").unwrap();
    writeln!(
        md,
        "Coherence = mean magnitude-squared coherence, 4–12 kHz.\n"
    )
    .unwrap();
    writeln!(md, "| layout | L/R in | SABLE L/R | generic L/R | coherence in | SABLE coherence | generic coherence |").unwrap();
    writeln!(md, "|---|---:|---:|---:|---:|---:|---:|").unwrap();
    let fs = 32_000;
    for layout in [
        StereoLayout::Coherent,
        StereoLayout::Panned,
        StereoLayout::PartiallyDecorrelated,
    ] {
        let fx = synth::stereo_s_fixture(fs, layout, 6);
        let (ts, te, _) = fx.truth.sibilants[0];
        let edge = fs as usize / 40;
        let src = (ts + edge, te - edge);
        let e = |x: &[f32], (a, b): (usize, usize)| {
            x[a..b].iter().map(|&v| (v as f64).powi(2)).sum::<f64>()
        };
        let lr = |c: &[Vec<f32>], w| 10.0 * (e(&c[0], w) / e(&c[1], w)).log10();
        let coh = |c: &[Vec<f32>], (a, b): (usize, usize)| {
            mean_coherence(&c[0], &c[1], a, b, 512, fs, (4000.0, 12000.0))
        };
        let s = run(&fx, "1.5", &sable());
        let ov = s.report.sibilant_overlays.iter().find(|o| o.applied);
        let so = ov.map_or((src.0 * 3 / 2, src.1 * 3 / 2), |o| {
            (o.out_start as usize + edge, o.out_end as usize - edge)
        });
        let g = run(&fx, "1.5", &generic());
        let go = (src.0 * 3 / 2, src.1 * 3 / 2);
        writeln!(
            md,
            "| {layout:?} | {:.2} dB | {:.2} dB | {:.2} dB | {:.2} | {:.2} | {:.2} |",
            lr(&fx.audio.channels, src),
            lr(&s.audio.channels, so),
            lr(&g.audio.channels, go),
            coh(&fx.audio.channels, src),
            coh(&s.audio.channels, so),
            coh(&g.audio.channels, go)
        )
        .unwrap();
    }
    md.push('\n');
}

fn exactness_section(md: &mut String) {
    let mut runs = 0;
    let mut exact = 0;
    let mut warn_relaxed = 0;
    for fs in [8_000, 16_000, 44_100, 48_000, 96_000] {
        for fx in common::fixtures(fs) {
            if fx.name.starts_with("vowel") && fs == 96_000 {
                continue;
            }
            for f in FACTORS {
                for o in [sable(), generic()] {
                    let out = run(&fx, f, &o);
                    runs += 1;
                    let n_out = StretchRatio::parse(f)
                        .unwrap()
                        .target_frames(fx.audio.frames() as u64)
                        .unwrap();
                    if out.audio.channels.iter().all(|c| c.len() as u64 == n_out) {
                        exact += 1;
                    }
                    if out
                        .report
                        .allocator_stage
                        .is_some_and(|s| s != sable_core::budget::Stage::A)
                    {
                        warn_relaxed += 1;
                    }
                }
            }
        }
    }
    writeln!(md, "### Exact-length sweep\n").unwrap();
    writeln!(
        md,
        "{exact} / {runs} runs (all fixtures × 5 sample rates × 10 factors × SABLE/generic) returned exactly \
         `N_out` frames on every channel; {warn_relaxed} used a relaxed allocator stage (B/C).\n"
    )
    .unwrap();
}

fn speech_section(md: &mut String) {
    let paths = common::speech_paths();
    if paths.is_empty() {
        return;
    }
    writeln!(md, "### Local speech recordings (not committed)\n").unwrap();
    writeln!(
        md,
        "| file | ch | r | N_in | N_out | stage | bursts copied | sibilants resynth. (of detected) | time |"
    )
    .unwrap();
    writeln!(md, "|---|---:|---:|---:|---:|:---:|---:|---:|---:|").unwrap();
    for (i, p) in paths.iter().enumerate() {
        let audio = common::read_wav(p).unwrap();
        let fx = Fixture {
            name: format!("speech{}", i + 1),
            audio,
            truth: Default::default(),
        };
        for f in ["0.65", "0.75", "1.25", "1.5", "2"] {
            let t = Instant::now();
            let s = run(&fx, f, &sable());
            let el = t.elapsed().as_secs_f64();
            let r = &s.report;
            let det = r.sibilant_overlays.len();
            let app = r.sibilant_overlays.iter().filter(|o| o.applied).count();
            let bursts = r.plosive_overlays.iter().filter(|o| o.applied).count();
            writeln!(
                md,
                "| speech {} | {} | {f} | {} | {} | {:?} | {bursts} | {app} ({det}) | {:.0} ms |",
                i + 1,
                r.channels,
                r.n_in,
                r.actual_frames,
                r.allocator_stage.unwrap(),
                el * 1000.0
            )
            .unwrap();
        }
    }
    md.push('\n');
}

fn main() {
    let write = std::env::args().any(|a| a == "--write");
    let mut md = String::new();
    writeln!(
        md,
        "<!-- BEGIN GENERATED: cargo run --release -p sable-core --example quality_report -- --write -->\n"
    )
    .unwrap();
    exactness_section(&mut md);
    pitch_section(&mut md);
    plosive_section(&mut md);
    sibilant_section(&mut md);
    stereo_section(&mut md);
    speech_section(&mut md);
    writeln!(md, "<!-- END GENERATED -->").unwrap();
    print!("{md}");
    if write {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../QUALITY.md");
        let old = std::fs::read_to_string(path).unwrap_or_default();
        let new = match (
            old.find("<!-- BEGIN GENERATED"),
            old.find("<!-- END GENERATED -->"),
        ) {
            (Some(a), Some(b)) => format!(
                "{}{}{}",
                &old[..a],
                md.trim_end(),
                &old[b + "<!-- END GENERATED -->".len()..]
            ),
            _ => format!("{old}\n{md}"),
        };
        std::fs::write(path, new).unwrap();
        eprintln!("updated {path}");
    }
}

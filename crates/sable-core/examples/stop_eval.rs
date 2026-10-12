//! Evaluates the plosive-burst detector against hand labels.
//!
//! Labels live in the gitignored `stop_labels.txt` at the workspace root (or the file given
//! as the first argument), one recording per line:
//!
//! ```text
//! # path | true burst onsets in ms | candidates to ignore (unsure), in ms
//! C:\audio\carlos.wav | 61.6, 566.9, 775.3 | 1716.9
//! ```
//!
//! A candidate within ±10 ms of a labelled onset counts as that stop. Unlabelled candidates
//! not in the ignore list count as non-stops. WAV only (convert other formats with the CLI).
//!
//! ```text
//! cargo run --release -p sable-core --example stop_eval [-- labels.txt] [--verbose]
//! ```

mod common;

use std::path::{Path, PathBuf};

use sable_core::{AudioView, analyze_events};

const TOL_MS: f64 = 10.0;

fn nums(s: &str) -> Vec<f64> {
    s.split(',')
        .map(str::trim)
        .filter(|v| !v.is_empty())
        .filter_map(|v| v.parse().ok())
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let verbose = args.iter().any(|a| a == "--verbose");
    let file = args
        .iter()
        .find(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from(concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../stop_labels.txt"
            ))
        });
    let Ok(text) = std::fs::read_to_string(&file) else {
        eprintln!("no labels file at {}", file.display());
        return;
    };
    let (mut tp, mut fn_, mut fp, mut total) = (0, 0, 0, 0);
    for line in text.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut parts = line.split('|');
        let path = parts.next().unwrap_or("").trim();
        let truth = nums(parts.next().unwrap_or(""));
        let ignore = nums(parts.next().unwrap_or(""));
        let audio = match common::read_wav(Path::new(path)) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("skip {path}: {e}");
                continue;
            }
        };
        let fs = audio.sample_rate as f64;
        let refs = audio.view_refs();
        let (_, det) = analyze_events(&AudioView {
            sample_rate: audio.sample_rate,
            channels: &refs,
        })
        .unwrap();
        let name = Path::new(path).file_name().unwrap().to_string_lossy();
        let (mut ftp, mut ffp) = (0, 0);
        let mut found = vec![false; truth.len()];
        if verbose {
            println!("== {name}");
            println!(
                "{:>9} {:>5} {:>5} {:>5} {:>6} {:>6} {:>5} {:>6} {:>5}  label",
                "ms", "jump", "per", "perF", "dip", "spike", "hf+", "valley", "conf"
            );
        }
        for b in &det.burst_candidates {
            let ms = b.onset as f64 * 1000.0 / fs;
            let hit = truth.iter().position(|t| (t - ms).abs() <= TOL_MS);
            let ignored = ignore.iter().any(|t| (t - ms).abs() <= TOL_MS);
            let label = match (hit, ignored) {
                (Some(i), _) => {
                    found[i] |= b.accepted;
                    if b.accepted { "stop  ok" } else { "stop  MISS" }
                }
                (None, true) => "unsure",
                (None, false) => {
                    if b.accepted {
                        ffp += 1;
                        "-     FALSE+"
                    } else {
                        "-"
                    }
                }
            };
            if verbose {
                println!(
                    "{ms:>9.1} {:>5.1} {:>5.2} {:>5.2} {:>6.1} {:>6.1} {:>5.2} {:>6} {:>5.2}  {label}",
                    b.jump_db,
                    b.periodicity_after,
                    b.periodicity_follow,
                    b.dip_db,
                    b.spike_db,
                    b.hf_added,
                    b.valley_depth_db.map_or("-".into(), |v| format!("{v:.1}")),
                    b.confidence
                );
            }
        }
        ftp += found.iter().filter(|&&f| f).count();
        let ffn = truth.len() - ftp;
        println!(
            "{name}: {ftp}/{} stops protected, {ffp} false positives",
            truth.len()
        );
        tp += ftp;
        fn_ += ffn;
        fp += ffp;
        total += truth.len();
    }
    println!("TOTAL: {tp}/{total} labelled stops protected ({fn_} missed), {fp} false positives");
}

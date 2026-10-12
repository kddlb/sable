//! Prints the detected event segmentation of a WAV file or a synthetic fixture.
//!
//! ```text
//! cargo run --release -p sable-core --example inspect_events -- path/to/file.wav [--json out.json]
//! cargo run --release -p sable-core --example inspect_events -- fixture:stop [48000]
//! cargo run --release -p sable-core --example inspect_events -- speech
//! ```

mod common;

use std::path::PathBuf;

use sable_core::{AudioView, analyze_events};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let target = args
        .first()
        .cloned()
        .unwrap_or_else(|| "fixture:stop".into());
    let json_out = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from);
    let audio = if let Some(name) = target.strip_prefix("fixture:") {
        let fs: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(48_000);
        common::fixtures(fs)
            .into_iter()
            .find(|f| f.name == name)
            .unwrap_or_else(|| panic!("unknown fixture {name}"))
            .audio
    } else if target == "speech" {
        let p = common::speech_path().expect("speech recording not found (set SABLE_SPEECH_WAV)");
        common::read_wav(&p).unwrap()
    } else {
        common::read_wav(&PathBuf::from(&target)).unwrap()
    };
    let refs = audio.view_refs();
    let view = AudioView {
        sample_rate: audio.sample_rate,
        channels: &refs,
    };
    let (feats, det) = analyze_events(&view).unwrap();
    let ms = |s: u64| s as f64 * 1000.0 / audio.sample_rate as f64;
    println!(
        "fs={} frames={} floor={:.1} dB speech={:.1} dB silence<{:.1} dB",
        audio.sample_rate,
        audio.frames(),
        feats.noise_floor_db,
        feats.speech_level_db,
        feats.silence_thr_db
    );
    println!(
        "{:>4} {:>17} {:>9} {:>9} {:>5}  reasons",
        "id", "kind", "start_ms", "len_ms", "conf"
    );
    for s in &det.segments {
        println!(
            "{:>4} {:>17} {:>9.1} {:>9.1} {:>5.2}  {}",
            s.id,
            format!("{:?}", s.kind),
            ms(s.start),
            ms(s.len()),
            s.confidence,
            s.reasons.join("; ")
        );
    }
    println!("\nburst candidates:");
    for b in &det.burst_candidates {
        println!(
            "  {:>9.1} ms  jump {:>5.1} dB  rise {:>4.1} ms  hf {:.2}  valley {:>6}  conf {:.2} {}",
            ms(b.onset),
            b.jump_db,
            b.rise_ms,
            b.hf_frac,
            b.valley_depth_db
                .map_or("-".to_string(), |v| format!("{v:.1}")),
            b.confidence,
            if b.accepted { "ACCEPTED" } else { "rejected" }
        );
    }
    if let Some(p) = json_out {
        let json = serde_json::json!({
            "segments": det.segments,
            "burst_candidates": det.burst_candidates,
        });
        std::fs::write(&p, serde_json::to_string_pretty(&json).unwrap()).unwrap();
        println!("wrote {}", p.display());
    }
}

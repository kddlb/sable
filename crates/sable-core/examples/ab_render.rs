//! Renders an A/B listening set: SABLE vs generic uniform WSOLA, for the local speech
//! recordings (if present) and selected synthetic fixtures, at several factors.
//! Output goes to `ab/` (gitignored) with diagnostic JSON for every SABLE render.
//!
//! ```text
//! cargo run --release -p sable-core --example ab_render [-- out_dir]
//! ```

mod common;

use std::path::PathBuf;

use sable_core::synth::{self, Fricative};
use sable_core::{AudioBuffer, AudioView, Options, StretchRatio, stretch};

const FACTORS: [&str; 5] = ["0.65", "0.75", "1.25", "1.5", "2"];

fn main() {
    let dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../ab")));
    std::fs::create_dir_all(&dir).unwrap();
    let mut items: Vec<(String, AudioBuffer)> = Vec::new();
    for (i, p) in common::speech_paths().iter().enumerate() {
        items.push((format!("speech{}", i + 1), common::read_wav(p).unwrap()));
    }
    for fx in [
        synth::utterance_fixture(48_000, 1),
        synth::stop_fixture(48_000, 2),
        synth::fricative_fixture(48_000, Fricative::S, 3),
        synth::fricative_fixture(48_000, Fricative::Z, 4),
    ] {
        items.push((format!("synth_{}", fx.name), fx.audio));
    }
    let mut index = String::from("file\tmode\tratio\tframes\tbursts\tsibilants\n");
    for (name, audio) in &items {
        common::write_wav(&dir.join(format!("{name}_orig.wav")), audio);
        let refs = audio.view_refs();
        let view = AudioView {
            sample_rate: audio.sample_rate,
            channels: &refs,
        };
        for f in FACTORS {
            let r = StretchRatio::parse(f).unwrap();
            for (mode, opts) in [
                ("sable", Options::default()),
                ("generic", Options::generic_wsola()),
            ] {
                let out = stretch(&view, r, &opts).unwrap();
                let tag = format!("{name}_x{}_{mode}", f.replace('.', "p"));
                common::write_wav(&dir.join(format!("{tag}.wav")), &out.audio);
                if mode == "sable" {
                    std::fs::write(
                        dir.join(format!("{tag}.json")),
                        serde_json::to_string_pretty(&out.report).unwrap(),
                    )
                    .unwrap();
                }
                let b = out
                    .report
                    .plosive_overlays
                    .iter()
                    .filter(|o| o.applied)
                    .count();
                let s = out
                    .report
                    .sibilant_overlays
                    .iter()
                    .filter(|o| o.applied)
                    .count();
                index.push_str(&format!(
                    "{tag}.wav\t{mode}\t{f}\t{}\t{b}\t{s}\n",
                    out.audio.frames()
                ));
            }
        }
    }
    std::fs::write(dir.join("index.tsv"), index).unwrap();
    println!("A/B set written to {}", dir.display());
}

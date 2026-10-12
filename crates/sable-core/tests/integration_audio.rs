//! Real recordings (local only). Set `SABLE_SPEECH_WAV` to a `;`-separated list of WAV
//! paths; the maintainer's default paths are used otherwise. Missing files are skipped,
//! so CI stays green without any committed speech.

use std::path::PathBuf;

use sable_core::{AudioView, Options, StretchRatio, stretch};

const DEFAULTS: [&str; 2] = [
    r"speech1.wav",
    r"speech2.wav",
];

fn speech_files() -> Vec<PathBuf> {
    let list: Vec<PathBuf> = match std::env::var("SABLE_SPEECH_WAV") {
        Ok(v) => v
            .split(';')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect(),
        Err(_) => DEFAULTS.iter().map(PathBuf::from).collect(),
    };
    list.into_iter().filter(|p| p.exists()).collect()
}

fn read(p: &PathBuf) -> (u32, Vec<Vec<f32>>) {
    let mut r = hound::WavReader::open(p).unwrap();
    let spec = r.spec();
    let nch = spec.channels as usize;
    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(Result::unwrap).collect(),
        hound::SampleFormat::Int => {
            let s = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f64;
            r.samples::<i32>()
                .map(|v| (v.unwrap() as f64 * s) as f32)
                .collect()
        }
    };
    let mut ch = vec![Vec::new(); nch];
    for fr in data.chunks_exact(nch) {
        for (c, &v) in fr.iter().enumerate() {
            ch[c].push(v);
        }
    }
    (spec.sample_rate, ch)
}

#[test]
fn real_speech_exact_length_finite_and_engaged() {
    let files = speech_files();
    if files.is_empty() {
        eprintln!("skipping: no local speech recordings");
        return;
    }
    for p in files {
        let (fs, ch) = read(&p);
        let refs: Vec<&[f32]> = ch.iter().map(Vec::as_slice).collect();
        let view = AudioView {
            sample_rate: fs,
            channels: &refs,
        };
        for f in ["0.5", "0.75", "1", "1.5", "2", "1/4", "4"] {
            let r = StretchRatio::parse(f).unwrap();
            let n_out = r.target_frames(ch[0].len() as u64).unwrap() as usize;
            for opts in [Options::default(), Options::generic_wsola()] {
                let out = stretch(&view, r, &opts).unwrap();
                for c in &out.audio.channels {
                    assert_eq!(c.len(), n_out, "{} r={f}", p.display());
                    assert!(c.iter().all(|v| v.is_finite()));
                }
            }
        }
        let out = stretch(
            &view,
            StretchRatio::parse("1.5").unwrap(),
            &Options::default(),
        )
        .unwrap();
        assert!(
            out.report.sibilant_overlays.iter().any(|o| o.applied),
            "{}: no sibilant resynthesized",
            p.display()
        );
    }
}

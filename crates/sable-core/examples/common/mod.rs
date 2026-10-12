//! Helpers shared by the examples (WAV I/O and fixture lookup).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use sable_core::AudioBuffer;
use sable_core::synth::{self, Fixture, Fricative, StereoLayout};

/// Local real-speech recordings (never committed): `$SABLE_SPEECH_WAV` (`;`-separated),
/// otherwise the lines of the gitignored `speech_files.txt` at the workspace root
/// (one path per line, `#` comments). Missing files are skipped.
pub fn speech_paths() -> Vec<PathBuf> {
    let list: Vec<PathBuf> = match std::env::var("SABLE_SPEECH_WAV") {
        Ok(v) => v
            .split(';')
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .collect(),
        Err(_) => {
            let file = concat!(env!("CARGO_MANIFEST_DIR"), "/../../speech_files.txt");
            std::fs::read_to_string(file)
                .unwrap_or_default()
                .lines()
                .map(str::trim)
                .filter(|l| !l.is_empty() && !l.starts_with('#'))
                .map(PathBuf::from)
                .collect()
        }
    };
    list.into_iter().filter(|p| p.exists()).collect()
}

/// First available real speech recording.
pub fn speech_path() -> Option<PathBuf> {
    speech_paths().into_iter().next()
}

pub fn read_wav(path: &Path) -> Result<AudioBuffer, String> {
    let mut r = hound::WavReader::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let spec = r.spec();
    let nch = spec.channels as usize;
    let data: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => r.samples::<f32>().map(|s| s.unwrap()).collect(),
        hound::SampleFormat::Int => {
            let scale = 1.0 / (1u64 << (spec.bits_per_sample - 1)) as f64;
            r.samples::<i32>()
                .map(|s| (s.unwrap() as f64 * scale) as f32)
                .collect()
        }
    };
    let mut channels = vec![Vec::with_capacity(data.len() / nch); nch];
    for fr in data.chunks_exact(nch) {
        for (c, &v) in fr.iter().enumerate() {
            channels[c].push(v);
        }
    }
    Ok(AudioBuffer {
        sample_rate: spec.sample_rate,
        channels,
    })
}

pub fn write_wav(path: &Path, a: &AudioBuffer) {
    let spec = hound::WavSpec {
        channels: a.channels.len() as u16,
        sample_rate: a.sample_rate,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(path, spec).expect("create wav");
    for i in 0..a.frames() {
        for c in &a.channels {
            w.write_sample(c[i]).expect("write sample");
        }
    }
    w.finalize().expect("finalize wav");
}

/// All named synthetic fixtures at `fs`.
pub fn fixtures(fs: u32) -> Vec<Fixture> {
    vec![
        synth::stop_fixture(fs, 1),
        synth::vowel_fixture(fs, 80.0),
        synth::vowel_fixture(fs, 140.0),
        synth::vowel_fixture(fs, 250.0),
        synth::fricative_fixture(fs, Fricative::S, 2),
        synth::fricative_fixture(fs, Fricative::Sh, 3),
        synth::fricative_fixture(fs, Fricative::Z, 4),
        synth::stereo_s_fixture(fs, StereoLayout::Coherent, 5),
        synth::stereo_s_fixture(fs, StereoLayout::Panned, 6),
        synth::stereo_s_fixture(fs, StereoLayout::PartiallyDecorrelated, 7),
        synth::sibilant_stop_fixture(fs, 8),
        synth::edge_bursts_fixture(fs, 9),
        synth::whisper_fixture(fs, 10),
        synth::hiss_fixture(fs, 11),
        synth::tiny_fixture(fs),
        synth::utterance_fixture(fs, 12),
    ]
}

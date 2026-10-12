//! Helpers shared by the examples (WAV I/O and fixture lookup).
#![allow(dead_code)]

use std::path::{Path, PathBuf};

use sable_core::AudioBuffer;
use sable_core::synth::{self, Fixture, Fricative, StereoLayout};

/// Default location of the maintainer's own speech recording (never committed).
pub const DEFAULT_SPEECH: &str = r"speech1.wav";

/// Path to a real speech recording: `$SABLE_SPEECH_WAV` or the default, if it exists.
pub fn speech_path() -> Option<PathBuf> {
    let p = std::env::var_os("SABLE_SPEECH_WAV")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(DEFAULT_SPEECH));
    p.exists().then_some(p)
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

//! Shared front-end helpers for the `sable` CLI and the optional `sable-gui` app.

#![forbid(unsafe_code)]

pub mod wav {
    //! WAV reading/writing via `hound` (integer or float PCM, mono or stereo).

    use std::path::Path;

    use clap::ValueEnum;
    use sable_core::AudioBuffer;

    /// Output sample format.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
    pub enum Format {
        /// 32-bit IEEE float (default; no clipping or quantization).
        Float32,
        /// 16-bit integer PCM (clips to [-1, 1), no dither).
        Pcm16,
        /// 24-bit integer PCM (clips to [-1, 1), no dither).
        Pcm24,
    }

    impl Format {
        pub const ALL: [Format; 3] = [Format::Float32, Format::Pcm16, Format::Pcm24];

        pub fn label(self) -> &'static str {
            match self {
                Format::Float32 => "32-bit float",
                Format::Pcm16 => "16-bit PCM",
                Format::Pcm24 => "24-bit PCM",
            }
        }
    }

    /// Reads a mono or stereo WAV into planar `f32`.
    pub fn read_wav(path: &Path) -> Result<AudioBuffer, String> {
        let ctx = |e: hound::Error| format!("{}: {e}", path.display());
        let mut reader = hound::WavReader::open(path).map_err(ctx)?;
        let spec = reader.spec();
        let nch = spec.channels as usize;
        if nch == 0 || nch > 2 {
            return Err(format!(
                "{}: {nch} channels; only mono and stereo are supported (no automatic downmix)",
                path.display()
            ));
        }
        let interleaved: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Float => {
                if spec.bits_per_sample != 32 {
                    return Err(format!("{}: unsupported float bit depth", path.display()));
                }
                reader
                    .samples::<f32>()
                    .collect::<Result<_, _>>()
                    .map_err(ctx)?
            }
            hound::SampleFormat::Int => {
                let bits = spec.bits_per_sample;
                if !(1..=32).contains(&bits) {
                    return Err(format!("{}: unsupported bit depth {bits}", path.display()));
                }
                let scale = 1.0 / (1u64 << (bits - 1)) as f64;
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| (v as f64 * scale) as f32))
                    .collect::<Result<_, _>>()
                    .map_err(ctx)?
            }
        };
        let frames = interleaved.len() / nch;
        let mut channels = vec![Vec::with_capacity(frames); nch];
        for f in interleaved.chunks_exact(nch) {
            for (c, &v) in f.iter().enumerate() {
                channels[c].push(v);
            }
        }
        Ok(AudioBuffer {
            sample_rate: spec.sample_rate,
            channels,
        })
    }

    /// Writes planar audio as an interleaved WAV.
    pub fn write_wav(path: &Path, audio: &AudioBuffer, format: Format) -> Result<(), String> {
        let ctx = |e: hound::Error| format!("{}: {e}", path.display());
        let (bits, sample_format) = match format {
            Format::Float32 => (32, hound::SampleFormat::Float),
            Format::Pcm16 => (16, hound::SampleFormat::Int),
            Format::Pcm24 => (24, hound::SampleFormat::Int),
        };
        let spec = hound::WavSpec {
            channels: audio.channels.len() as u16,
            sample_rate: audio.sample_rate,
            bits_per_sample: bits,
            sample_format,
        };
        let mut w = hound::WavWriter::create(path, spec).map_err(ctx)?;
        let full = (1i64 << (bits - 1)) as f64;
        for i in 0..audio.frames() {
            for c in &audio.channels {
                match format {
                    Format::Float32 => w.write_sample(c[i]).map_err(ctx)?,
                    Format::Pcm16 | Format::Pcm24 => {
                        let v = (c[i] as f64 * full).round().clamp(-full, full - 1.0) as i32;
                        w.write_sample(v).map_err(ctx)?
                    }
                }
            }
        }
        w.finalize().map_err(ctx)
    }
}

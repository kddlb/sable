//! Shared front-end helpers for the `sable` CLI and the optional `sable-gui` app:
//! audio file input (WAV via hound; other formats via symphonia with the `decode`
//! feature), WAV output, and optional leading-silence trimming.

#![forbid(unsafe_code)]

use std::path::Path;

use sable_core::AudioBuffer;

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

    /// Reads a WAV into planar `f32` (any channel count; callers enforce limits).
    pub fn read_wav(path: &Path) -> Result<AudioBuffer, String> {
        let ctx = |e: hound::Error| format!("{}: {e}", path.display());
        let mut reader = hound::WavReader::open(path).map_err(ctx)?;
        let spec = reader.spec();
        let nch = spec.channels as usize;
        if nch == 0 {
            return Err(format!("{}: no channels", path.display()));
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
        Ok(super::deinterleave(&interleaved, nch, spec.sample_rate))
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

#[cfg(feature = "decode")]
pub mod decode {
    //! Decoding of non-WAV formats with symphonia (pure Rust): FLAC, MP3/MP2/MP1,
    //! Ogg Vorbis, AAC-LC and ALAC (MP4/M4A, CAF), AIFF, MKV/WebM audio, ADPCM.

    use std::fs::File;
    use std::path::Path;

    use sable_core::AudioBuffer;
    use symphonia::core::codecs::audio::AudioDecoderOptions;
    use symphonia::core::errors::Error;
    use symphonia::core::formats::probe::Hint;
    use symphonia::core::formats::{FormatOptions, TrackType};
    use symphonia::core::io::{MediaSourceStream, MediaSourceStreamOptions};
    use symphonia::core::meta::MetadataOptions;

    /// Decodes the default audio track of `path` to planar `f32` at its native rate.
    pub fn decode_file(path: &Path) -> Result<AudioBuffer, String> {
        let err = |e: Error| format!("{}: {e}", path.display());
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let mss = MediaSourceStream::new(Box::new(file), MediaSourceStreamOptions::default());
        let mut hint = Hint::new();
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            hint.with_extension(ext);
        }
        let mut format = symphonia::default::get_probe()
            .probe(
                &hint,
                mss,
                FormatOptions::default(),
                MetadataOptions::default(),
            )
            .map_err(err)?;
        let track = format
            .default_track(TrackType::Audio)
            .ok_or_else(|| format!("{}: no audio track", path.display()))?;
        let track_id = track.id;
        let params = track
            .codec_params
            .as_ref()
            .and_then(|p| p.audio())
            .ok_or_else(|| format!("{}: unknown audio codec", path.display()))?
            .clone();
        let mut decoder = symphonia::default::get_codecs()
            .make_audio_decoder(&params, &AudioDecoderOptions::default())
            .map_err(err)?;

        let mut rate = params.sample_rate;
        let mut nch = params.channels.as_ref().map(|c| c.count());
        let mut interleaved: Vec<f32> = Vec::new();
        let mut scratch: Vec<f32> = Vec::new();
        loop {
            let packet = match format.next_packet() {
                Ok(Some(p)) => p,
                Ok(None) => break,
                Err(Error::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(err(e)),
            };
            if packet.track_id != track_id {
                continue;
            }
            let buf = match decoder.decode(&packet) {
                Ok(b) => b,
                // Corrupt packet: skip it (standard symphonia practice), keep decoding.
                Err(Error::DecodeError(_)) => continue,
                Err(e) => return Err(err(e)),
            };
            let spec = buf.spec();
            let (r, c) = (spec.rate(), spec.channels().count());
            match (rate, nch) {
                (Some(r0), Some(c0)) if r0 != r || c0 != c => {
                    return Err(format!(
                        "{}: stream changes format mid-way ({r0} Hz/{c0} ch -> {r} Hz/{c} ch)",
                        path.display()
                    ));
                }
                _ => {
                    rate = Some(r);
                    nch = Some(c);
                }
            }
            scratch.clear();
            buf.copy_to_vec_interleaved(&mut scratch);
            interleaved.extend_from_slice(&scratch[..buf.frames() * c]);
        }
        let (Some(rate), Some(nch)) = (rate, nch) else {
            return Err(format!("{}: no decodable audio", path.display()));
        };
        if nch == 0 {
            return Err(format!("{}: no channels", path.display()));
        }
        Ok(super::deinterleave(&interleaved, nch, rate))
    }
}

/// File extensions accepted by [`read_audio`] (lowercase, without dot).
pub fn supported_extensions() -> &'static [&'static str] {
    #[cfg(feature = "decode")]
    {
        &[
            "wav", "wave", "flac", "mp3", "mp2", "mp1", "ogg", "oga", "m4a", "mp4", "aac", "aif",
            "aiff", "aifc", "caf", "mkv", "mka", "webm",
        ]
    }
    #[cfg(not(feature = "decode"))]
    {
        &["wav", "wave"]
    }
}

/// True for lossy MPEG/AAC inputs whose decoded length may include encoder padding
/// (symphonia trims MP3 padding only when the file records it, and does not trim AAC).
pub fn may_have_encoder_padding(path: &Path) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    matches!(ext.as_str(), "mp3" | "mp2" | "mp1" | "m4a" | "mp4" | "aac")
}

/// Reads any supported audio file to planar `f32`. WAV uses hound (falling back to
/// symphonia for WAV encodings hound cannot read); everything else needs the `decode`
/// feature. More than two channels is an error: SABLE never downmixes silently.
pub fn read_audio(path: &Path) -> Result<AudioBuffer, String> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();
    let audio = if ext == "wav" || ext == "wave" {
        match wav::read_wav(path) {
            Ok(a) => a,
            #[cfg(feature = "decode")]
            Err(_) => decode::decode_file(path)?,
            #[cfg(not(feature = "decode"))]
            Err(e) => return Err(e),
        }
    } else {
        #[cfg(feature = "decode")]
        {
            decode::decode_file(path)?
        }
        #[cfg(not(feature = "decode"))]
        {
            return Err(format!(
                "{}: only WAV is supported in this build (enable the `decode` feature)",
                path.display()
            ));
        }
    };
    let nch = audio.channels.len();
    if nch == 0 || nch > 2 {
        return Err(format!(
            "{}: {nch} channels; only mono and stereo are supported (no automatic downmix)",
            path.display()
        ));
    }
    Ok(audio)
}

fn deinterleave(interleaved: &[f32], nch: usize, sample_rate: u32) -> AudioBuffer {
    let frames = interleaved.len() / nch;
    let mut channels = vec![Vec::with_capacity(frames); nch];
    for f in interleaved.chunks_exact(nch) {
        for (c, &v) in f.iter().enumerate() {
            channels[c].push(v);
        }
    }
    AudioBuffer {
        sample_rate,
        channels,
    }
}

pub mod trim {
    //! Optional removal of leading and trailing silence before stretching (front-end
    //! preprocessing; the stretch itself and its exact-length contract apply to the
    //! trimmed audio).

    use sable_core::AudioBuffer;

    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct SilenceTrim {
        /// Drop silence before the first sound.
        pub leading: bool,
        /// Drop silence after the last sound.
        pub trailing: bool,
        /// Level (dBFS RMS over `window_ms`, loudest channel) that counts as sound.
        pub threshold_db: f64,
        /// Audio kept before the first sound so its onset is not clipped.
        pub preroll_ms: f64,
        /// Audio kept after the last sound so its decay is not clipped.
        pub postroll_ms: f64,
        /// RMS window used for detection.
        pub window_ms: f64,
    }

    impl Default for SilenceTrim {
        fn default() -> Self {
            Self {
                leading: true,
                trailing: false,
                threshold_db: -50.0,
                preroll_ms: 20.0,
                postroll_ms: 50.0,
                window_ms: 5.0,
            }
        }
    }

    /// Frames removed from each end of the input.
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    pub struct Trimmed {
        pub leading: usize,
        pub trailing: usize,
    }

    /// Index of the first (or, with `from_end`, last) sample at the threshold level,
    /// found window by window from that end.
    fn edge_sound(audio: &AudioBuffer, cfg: &SilenceTrim, from_end: bool) -> Option<usize> {
        let n = audio.frames();
        let fs = audio.sample_rate as f64;
        let w = ((cfg.window_ms * fs / 1000.0).round() as usize).max(1);
        let thr = 10f64.powf(cfg.threshold_db / 10.0); // mean-square threshold
        let amp = thr.sqrt() as f32;
        let windows = (0..n.div_ceil(w)).map(|k| (k * w, ((k + 1) * w).min(n)));
        let loud = |&(start, end): &(usize, usize)| {
            audio.channels.iter().any(|c| {
                let ms = c[start..end]
                    .iter()
                    .map(|&v| (v as f64).powi(2))
                    .sum::<f64>()
                    / (end - start) as f64;
                ms >= thr
            })
        };
        let peak = |i: &usize| audio.channels.iter().any(|c| c[*i].abs() >= amp);
        // Refine to the outermost sample in the loud window above the RMS threshold level.
        if from_end {
            let (start, end) = windows.rev().find(loud)?;
            Some((start..end).rev().find(peak).unwrap_or(end - 1))
        } else {
            let (start, end) = windows.clone().find(loud)?;
            Some((start..end).find(peak).unwrap_or(start))
        }
    }

    /// Number of leading frames to drop. Returns 0 if the whole clip is below the
    /// threshold (never empties a file).
    pub fn leading_silence_frames(audio: &AudioBuffer, cfg: &SilenceTrim) -> usize {
        let Some(first) = edge_sound(audio, cfg, false) else {
            return 0;
        };
        let pre = (cfg.preroll_ms * audio.sample_rate as f64 / 1000.0).round() as usize;
        first.saturating_sub(pre)
    }

    /// Number of trailing frames to drop. Returns 0 if the whole clip is below the
    /// threshold (never empties a file).
    pub fn trailing_silence_frames(audio: &AudioBuffer, cfg: &SilenceTrim) -> usize {
        let Some(last) = edge_sound(audio, cfg, true) else {
            return 0;
        };
        let post = (cfg.postroll_ms * audio.sample_rate as f64 / 1000.0).round() as usize;
        audio.frames().saturating_sub(last + 1).saturating_sub(post)
    }

    /// What `cfg` removes from `audio` (only the ends it enables).
    pub fn silence_to_trim(audio: &AudioBuffer, cfg: &SilenceTrim) -> Trimmed {
        let leading = if cfg.leading {
            leading_silence_frames(audio, cfg)
        } else {
            0
        };
        let trailing = if cfg.trailing {
            trailing_silence_frames(audio, cfg)
        } else {
            0
        };
        Trimmed { leading, trailing }
    }

    /// Copy of `audio` without the frames `t` removes from each end.
    pub fn trim_frames(audio: &AudioBuffer, t: Trimmed) -> AudioBuffer {
        AudioBuffer {
            sample_rate: audio.sample_rate,
            channels: audio
                .channels
                .iter()
                .map(|c| {
                    let a = t.leading.min(c.len());
                    let b = c.len().saturating_sub(t.trailing).max(a);
                    c[a..b].to_vec()
                })
                .collect(),
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn buf(x: Vec<f32>) -> AudioBuffer {
            AudioBuffer {
                sample_rate: 1000,
                channels: vec![x],
            }
        }

        #[test]
        fn trims_to_onset_minus_preroll() {
            let mut x = vec![0.0f32; 1000];
            for v in &mut x[600..] {
                *v = 0.5;
            }
            let a = buf(x);
            let cfg = SilenceTrim::default(); // 20 ms preroll at 1 kHz = 20 frames
            assert_eq!(leading_silence_frames(&a, &cfg), 580);
            let t = trim_frames(
                &a,
                Trimmed {
                    leading: 580,
                    trailing: 0,
                },
            );
            assert_eq!(t.frames(), 420);
        }

        #[test]
        fn quiet_noise_is_skipped_but_all_silent_is_kept() {
            let mut x = vec![1e-4f32; 1000]; // -80 dBFS
            x[700] = 0.3;
            assert_eq!(
                leading_silence_frames(&buf(x), &SilenceTrim::default()),
                680
            );
            assert_eq!(
                leading_silence_frames(&buf(vec![0.0; 500]), &SilenceTrim::default()),
                0
            );
            assert_eq!(
                leading_silence_frames(&buf(vec![0.2; 500]), &SilenceTrim::default()),
                0
            );
        }

        #[test]
        fn trailing_trim_keeps_postroll() {
            let mut x = vec![0.0f32; 1000];
            for v in &mut x[100..400] {
                *v = 0.5;
            }
            let a = buf(x);
            let cfg = SilenceTrim {
                trailing: true,
                ..SilenceTrim::default()
            }; // 20 ms pre-roll, 50 ms post-roll at 1 kHz
            assert_eq!(
                silence_to_trim(&a, &cfg),
                Trimmed {
                    leading: 80,
                    trailing: 550
                }
            );
            let t = trim_frames(&a, silence_to_trim(&a, &cfg));
            assert_eq!(t.frames(), 370);
            assert_eq!(trailing_silence_frames(&buf(vec![0.0; 500]), &cfg), 0);
            assert_eq!(trailing_silence_frames(&buf(vec![0.2; 500]), &cfg), 0);
        }

        #[test]
        fn stereo_uses_loudest_channel() {
            let a = AudioBuffer {
                sample_rate: 1000,
                channels: vec![vec![0.0; 300], {
                    let mut r = vec![0.0f32; 300];
                    r[100] = 0.5;
                    r
                }],
            };
            assert_eq!(leading_silence_frames(&a, &SilenceTrim::default()), 80);
        }
    }
}

/// Convenience used by both front ends: read a file, optionally trim silence at its ends.
/// Returns the audio to process and the frames removed from each end.
pub fn load_input(
    path: &Path,
    trim: Option<&trim::SilenceTrim>,
) -> Result<(AudioBuffer, trim::Trimmed), String> {
    let audio = read_audio(path)?;
    match trim {
        Some(cfg) => {
            let t = trim::silence_to_trim(&audio, cfg);
            Ok((trim::trim_frames(&audio, t), t))
        }
        None => Ok((audio, trim::Trimmed::default())),
    }
}

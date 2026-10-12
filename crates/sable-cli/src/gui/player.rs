//! Minimal audio preview through the default output device (cpal).
//!
//! Playback only: if the device runs at a different rate than the file, samples are
//! linearly interpolated for listening. This never touches the processed audio.

use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{FromSample, SampleFormat, SizedSample, Stream, StreamConfig};
use sable_core::AudioBuffer;

/// Which buffer is playing.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Source {
    Original,
    Result,
}

struct State {
    buf: Option<Arc<AudioBuffer>>,
    source: Source,
    /// Read position in source frames.
    pos: f64,
    /// Source frames per device frame.
    step: f64,
    playing: bool,
}

pub struct Player {
    _stream: Stream,
    state: Arc<Mutex<State>>,
    pub device_name: String,
    pub device_rate: u32,
    pub device_channels: u16,
}

impl Player {
    pub fn new() -> Result<Self, String> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or("no default audio output device")?;
        let device_name = device
            .description()
            .map(|d| d.name().to_string())
            .unwrap_or_else(|_| "default output".into());
        let supported = device.default_output_config().map_err(|e| e.to_string())?;
        let config: StreamConfig = supported.config();
        let state = Arc::new(Mutex::new(State {
            buf: None,
            source: Source::Original,
            pos: 0.0,
            step: 1.0,
            playing: false,
        }));
        let stream = match supported.sample_format() {
            SampleFormat::F32 => build::<f32>(&device, &config, state.clone()),
            SampleFormat::F64 => build::<f64>(&device, &config, state.clone()),
            SampleFormat::I16 => build::<i16>(&device, &config, state.clone()),
            SampleFormat::I32 => build::<i32>(&device, &config, state.clone()),
            SampleFormat::U16 => build::<u16>(&device, &config, state.clone()),
            other => Err(format!("unsupported device sample format {other:?}")),
        }?;
        stream.play().map_err(|e| e.to_string())?;
        Ok(Self {
            _stream: stream,
            state,
            device_name,
            device_rate: config.sample_rate,
            device_channels: config.channels,
        })
    }

    /// Starts playing `buf` from `from_frame`.
    pub fn play(&self, buf: Arc<AudioBuffer>, source: Source, from_frame: usize) {
        let mut s = self.state.lock().unwrap();
        s.step = buf.sample_rate as f64 / self.device_rate as f64;
        s.pos = from_frame.min(buf.frames()) as f64;
        s.buf = Some(buf);
        s.source = source;
        s.playing = true;
    }

    pub fn stop(&self) {
        self.state.lock().unwrap().playing = false;
    }

    /// Moves the read position of the current buffer (clamped to its length).
    /// Works while playing; a no-op when nothing is loaded.
    pub fn seek(&self, frame: f64) {
        let mut s = self.state.lock().unwrap();
        if let Some(len) = s.buf.as_ref().map(|b| b.frames() as f64) {
            s.pos = frame.clamp(0.0, (len - 1.0).max(0.0));
        }
    }

    /// `(source, position, length)` in frames of the playing buffer, while playing.
    pub fn status(&self) -> Option<(Source, f64, f64)> {
        let s = self.state.lock().unwrap();
        let len = s.buf.as_ref().map_or(0.0, |b| b.frames() as f64);
        s.playing.then_some((s.source, s.pos, len))
    }
}

fn build<T>(
    device: &cpal::Device,
    config: &StreamConfig,
    state: Arc<Mutex<State>>,
) -> Result<Stream, String>
where
    T: SizedSample + FromSample<f32>,
{
    let out_ch = config.channels as usize;
    device
        .build_output_stream::<T, _, _>(
            *config,
            move |data: &mut [T], _| {
                let mut s = state.lock().unwrap();
                for frame in data.chunks_mut(out_ch) {
                    let mut vals = [0f32; 2];
                    let mut nch = 1;
                    if s.playing {
                        let pos = s.pos;
                        match &s.buf {
                            Some(b) if (pos as usize) < b.frames() => {
                                nch = b.channels.len();
                                let i = pos as usize;
                                let frac = (pos - i as f64) as f32;
                                for (c, v) in vals.iter_mut().enumerate().take(nch.min(2)) {
                                    let ch = &b.channels[c];
                                    let a = ch[i];
                                    let n = *ch.get(i + 1).unwrap_or(&a);
                                    *v = a + (n - a) * frac;
                                }
                            }
                            _ => s.playing = false,
                        }
                        if s.playing {
                            s.pos += s.step;
                        }
                    }
                    for (c, out) in frame.iter_mut().enumerate() {
                        let v = if nch == 1 {
                            vals[0]
                        } else if c < 2 {
                            vals[c]
                        } else {
                            0.0
                        };
                        *out = T::from_sample(v);
                    }
                }
            },
            |e| eprintln!("audio stream error: {e}"),
            None,
        )
        .map_err(|e| e.to_string())
}

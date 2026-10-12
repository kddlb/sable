//! Planar PCM containers and input validation.

use crate::error::StretchError;

pub const MIN_SAMPLE_RATE: u32 = 8_000;
pub const MAX_SAMPLE_RATE: u32 = 96_000;
pub const MAX_CHANNELS: usize = 2;

/// Borrowed planar `f32` input; all channels must have equal length.
#[derive(Clone, Copy, Debug)]
pub struct AudioView<'a> {
    pub sample_rate: u32,
    pub channels: &'a [&'a [f32]],
}

/// Owned planar `f32` audio.
#[derive(Clone, Debug, PartialEq)]
pub struct AudioBuffer {
    pub sample_rate: u32,
    pub channels: Vec<Vec<f32>>,
}

impl AudioBuffer {
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, Vec::len)
    }

    pub fn view_refs(&self) -> Vec<&[f32]> {
        self.channels.iter().map(Vec::as_slice).collect()
    }
}

impl AudioView<'_> {
    pub fn frames(&self) -> usize {
        self.channels.first().map_or(0, |c| c.len())
    }

    /// Checks sample rate, channel layout and finiteness.
    pub fn validate(&self) -> Result<(), StretchError> {
        if !(MIN_SAMPLE_RATE..=MAX_SAMPLE_RATE).contains(&self.sample_rate) {
            return Err(StretchError::UnsupportedSampleRate(self.sample_rate));
        }
        let nch = self.channels.len();
        if nch == 0 || nch > MAX_CHANNELS {
            return Err(StretchError::UnsupportedChannelCount(nch));
        }
        let expected = self.channels[0].len();
        for (channel, c) in self.channels.iter().enumerate() {
            if c.len() != expected {
                return Err(StretchError::UnequalChannelLengths {
                    channel,
                    len: c.len(),
                    expected,
                });
            }
            if let Some(index) = c.iter().position(|v| !v.is_finite()) {
                return Err(StretchError::NonFiniteSample { channel, index });
            }
        }
        Ok(())
    }

    pub fn is_all_zero(&self) -> bool {
        self.channels.iter().all(|c| c.iter().all(|&v| v == 0.0))
    }
}

/// Peak and RMS of one channel, for diagnostics.
pub fn peak_rms(x: &[f32]) -> (f32, f32) {
    if x.is_empty() {
        return (0.0, 0.0);
    }
    let mut peak = 0.0f32;
    let mut acc = 0.0f64;
    for &v in x {
        peak = peak.max(v.abs());
        acc += (v as f64) * (v as f64);
    }
    (peak, (acc / x.len() as f64).sqrt() as f32)
}

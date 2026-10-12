//! Typed errors returned by the public API.

use thiserror::Error;

/// Every failure mode of [`crate::stretch`] and the ratio parsers.
///
/// Valid input never produces a panic; anything that would otherwise be undefined
/// behaviour is reported through one of these variants.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum StretchError {
    #[error("stretch ratio numerator and denominator must both be positive")]
    ZeroRatio,
    #[error("malformed stretch ratio {0:?}")]
    MalformedRatio(String),
    #[error("malformed or unusable target duration {0:?}")]
    MalformedDuration(String),
    #[error("unsupported sample rate {0} Hz (supported: 8000..=96000)")]
    UnsupportedSampleRate(u32),
    #[error("unsupported channel count {0} (supported: 1 or 2)")]
    UnsupportedChannelCount(usize),
    #[error("channel {channel} has {len} frames but channel 0 has {expected}")]
    UnequalChannelLengths {
        channel: usize,
        len: usize,
        expected: usize,
    },
    #[error("non-finite sample in channel {channel} at frame {index}")]
    NonFiniteSample { channel: usize, index: usize },
    #[error("nonempty input of {n_in} frames rounds to zero output frames")]
    OutputTooShort { n_in: u64 },
    #[error("target output of {n_out} frames exceeds configured limit of {limit}")]
    OutputTooLarge { n_out: u128, limit: u64 },
    #[error("pitch shift of {0} semitones is outside -24..=24 or not finite")]
    InvalidPitch(String),
    #[error("arithmetic overflow while computing the target frame count")]
    Overflow,
    #[error("internal invariant violated: {0}")]
    Internal(String),
}

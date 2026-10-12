//! Time targeting: derive the stretch ratio from a wanted output duration instead of a
//! factor (for example, fitting a disclaimer read into a fixed slot).
//!
//! Durations are parsed exactly (decimal text, never binary floats) and converted to a
//! frame count at the input's sample rate. The ratio `n_target / n_in` then yields
//! exactly `n_target` output frames, because `floor(n_in · n_target / n_in + 1/2)` is
//! `n_target` for any `n_in > 0`.

use std::fmt;

use serde::Serialize;

use crate::error::StretchError;
use crate::ratio::StretchRatio;

/// A wanted output duration: an exact number of seconds or an exact frame count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetDuration {
    /// `numerator / denominator` seconds (denominator is a power of ten, times 1000 for
    /// millisecond input).
    Seconds { numerator: u128, denominator: u128 },
    /// An exact frame count, independent of the sample rate.
    Frames(u64),
}

/// How a target duration constrains the output.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum TargetMode {
    /// The output has exactly the target length (shorter inputs are lengthened).
    #[default]
    Exact,
    /// The output is shortened to the target only if the input is longer; shorter
    /// inputs are left at their own length (ratio 1).
    AtMost,
}

impl TargetDuration {
    /// Parses a duration:
    ///
    /// * seconds: `"29.5"`, `"29.5s"`, `".75"`
    /// * milliseconds: `"1500ms"`
    /// * clock time: `"0:29.5"`, `"1:02.250"`, `"1:00:00"` (`[[h:]m:]s`, later fields
    ///   below 60)
    /// * frames: `"720000f"` or `"720000 frames"`
    pub fn parse(s: &str) -> Result<Self, StretchError> {
        let bad = || StretchError::MalformedDuration(s.to_string());
        let t = s.trim().to_ascii_lowercase();
        if let Some(n) = t
            .strip_suffix("frames")
            .or_else(|| t.strip_suffix('f'))
            .map(str::trim_end)
        {
            let f = parse_uint(n).ok_or_else(bad)?;
            return u64::try_from(f).map(Self::Frames).map_err(|_| bad());
        }
        let (body, scale) = match t.strip_suffix("ms") {
            Some(b) => (b.trim_end(), 1000u128),
            None => (t.strip_suffix('s').unwrap_or(&t).trim_end(), 1),
        };
        let fields: Vec<&str> = body.split(':').collect();
        if fields.len() > 3 || (scale != 1 && fields.len() > 1) {
            return Err(bad());
        }
        let (last, whole) = fields.split_last().ok_or_else(bad)?;
        let (num, den) = parse_decimal(last).ok_or_else(bad)?;
        // Leading clock fields are whole hours/minutes; fold them into whole minutes.
        let mut minutes: u128 = 0;
        for (i, f) in whole.iter().enumerate() {
            let v = parse_uint(f).ok_or_else(bad)?;
            if i > 0 && v >= 60 {
                return Err(bad());
            }
            minutes = minutes
                .checked_mul(60)
                .and_then(|m| m.checked_add(v))
                .ok_or_else(bad)?;
        }
        if !whole.is_empty() && num >= 60 * den {
            return Err(bad());
        }
        let numerator = minutes
            .checked_mul(60 * den)
            .and_then(|m| m.checked_add(num))
            .ok_or_else(bad)?;
        let denominator = den.checked_mul(scale).ok_or_else(bad)?;
        Ok(Self::Seconds {
            numerator,
            denominator,
        })
    }

    /// The duration in frames at `sample_rate`, rounded to the nearest frame (ties up).
    /// Fails when it rounds to zero or does not fit in `u64`.
    pub fn frames(&self, sample_rate: u32) -> Result<u64, StretchError> {
        let n = match *self {
            Self::Frames(n) => n as u128,
            Self::Seconds {
                numerator,
                denominator,
            } => {
                let a = numerator
                    .checked_mul(sample_rate as u128)
                    .ok_or(StretchError::Overflow)?;
                let (k, rem) = (a / denominator, a % denominator);
                k + (rem.checked_mul(2).ok_or(StretchError::Overflow)? >= denominator) as u128
            }
        };
        if n == 0 {
            return Err(StretchError::MalformedDuration(format!(
                "{self} rounds to zero frames"
            )));
        }
        u64::try_from(n).map_err(|_| StretchError::Overflow)
    }

    /// Approximate seconds at `sample_rate`, for display only.
    pub fn seconds_f64(&self, sample_rate: u32) -> f64 {
        match *self {
            Self::Frames(n) => n as f64 / sample_rate as f64,
            Self::Seconds {
                numerator,
                denominator,
            } => numerator as f64 / denominator as f64,
        }
    }

    /// The stretch ratio that turns `n_in` input frames into this duration at
    /// `sample_rate`. With [`TargetMode::AtMost`], inputs already within the target
    /// get ratio 1.
    pub fn ratio(
        &self,
        n_in: u64,
        sample_rate: u32,
        mode: TargetMode,
    ) -> Result<StretchRatio, StretchError> {
        let n_target = self.frames(sample_rate)?;
        if mode == TargetMode::AtMost && n_in <= n_target {
            return StretchRatio::new(1, 1);
        }
        StretchRatio::for_frames(n_in, n_target)
    }
}

impl fmt::Display for TargetDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::Frames(n) => write!(f, "{n} frames"),
            Self::Seconds {
                numerator,
                denominator,
            } => write!(f, "{} s", numerator as f64 / denominator as f64),
        }
    }
}

fn parse_uint(s: &str) -> Option<u128> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

/// Exact `"12.345"` → `(12345, 1000)`.
fn parse_decimal(s: &str) -> Option<(u128, u128)> {
    let (i, f) = s.split_once('.').unwrap_or((s, ""));
    if (i.is_empty() && f.is_empty()) || f.len() > 18 {
        return None;
    }
    let ip = if i.is_empty() { 0 } else { parse_uint(i)? };
    let fp = if f.is_empty() { 0 } else { parse_uint(f)? };
    let den = 10u128.pow(f.len() as u32);
    Some((ip.checked_mul(den)?.checked_add(fp)?, den))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frames(s: &str, fs: u32) -> u64 {
        TargetDuration::parse(s).unwrap().frames(fs).unwrap()
    }

    #[test]
    fn parses_all_forms() {
        assert_eq!(frames("29.5", 48_000), 1_416_000);
        assert_eq!(frames("29.5s", 48_000), 1_416_000);
        assert_eq!(frames(" 29.5 S ", 48_000), 1_416_000);
        assert_eq!(frames(".75", 48_000), 36_000);
        assert_eq!(frames("1500ms", 44_100), 66_150);
        assert_eq!(frames("0:29.5", 48_000), 1_416_000);
        assert_eq!(frames("1:02.250", 1000), 62_250);
        assert_eq!(frames("1:00:00", 8000), 3600 * 8000);
        assert_eq!(frames("720000f", 48_000), 720_000);
        assert_eq!(frames("720000 frames", 44_100), 720_000);
    }

    #[test]
    fn rounds_to_nearest_frame_ties_up() {
        // 0.00001 s at 44.1 kHz = 0.441 frame -> 0 -> error
        assert!(
            TargetDuration::parse("0.00001")
                .unwrap()
                .frames(44_100)
                .is_err()
        );
        assert!(
            TargetDuration::parse("0.4ms")
                .unwrap()
                .frames(1000)
                .is_err()
        );
        assert_eq!(frames("0.5ms", 1000), 1);
        assert_eq!(frames("1.4ms", 1000), 1);
        assert_eq!(frames("1.5ms", 1000), 2);
    }

    #[test]
    fn rejects_bad_input() {
        for s in [
            "", "s", "ms", ".", "-1", "1e3", "1:60", "1:2:60", "1:60:00", "1:2:3:4", "1:30ms",
            "abc", "1.5f", "f", "0f", "0", "0:00", "+3",
        ] {
            let ok = TargetDuration::parse(s).and_then(|d| d.frames(48_000));
            assert!(ok.is_err(), "{s:?} should fail, got {ok:?}");
        }
    }

    #[test]
    fn ratio_hits_target_exactly() {
        for n_in in [1u64, 7, 44_100, 1_234_567] {
            for t in ["0.5", "1", "2.345", "17.9", "3000f"] {
                let d = TargetDuration::parse(t).unwrap();
                let want = d.frames(44_100).unwrap();
                let r = d.ratio(n_in, 44_100, TargetMode::Exact).unwrap();
                assert_eq!(r.target_frames(n_in).unwrap(), want, "{n_in} -> {t}");
            }
        }
    }

    #[test]
    fn at_most_never_lengthens() {
        let d = TargetDuration::parse("1").unwrap();
        let r = d.ratio(40_000, 48_000, TargetMode::AtMost).unwrap();
        assert!(r.is_identity());
        let r = d.ratio(48_000, 48_000, TargetMode::AtMost).unwrap();
        assert!(r.is_identity());
        let r = d.ratio(60_000, 48_000, TargetMode::AtMost).unwrap();
        assert_eq!(r, StretchRatio::new(4, 5).unwrap());
        assert_eq!(r.target_frames(60_000).unwrap(), 48_000);
    }
}

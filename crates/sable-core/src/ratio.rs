//! Exact rational stretch factors and the authoritative target-frame computation.
//!
//! `r = duration_out / duration_in`; `r > 1` lengthens. The output frame count is
//! derived from integers only: `N_out = floor(N_in * p / q + 1/2)`.

use std::fmt;
use std::num::NonZeroU64;

use serde::Serialize;

use crate::error::StretchError;

/// Positive rational stretch factor `numerator / denominator`, always stored reduced.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub struct StretchRatio {
    pub numerator: u64,
    #[serde(serialize_with = "ser_nonzero")]
    pub denominator: NonZeroU64,
}

fn ser_nonzero<S: serde::Serializer>(v: &NonZeroU64, s: S) -> Result<S::Ok, S::Error> {
    s.serialize_u64(v.get())
}

fn gcd(mut a: u128, mut b: u128) -> u128 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

impl StretchRatio {
    /// Builds a reduced ratio; rejects a zero numerator or denominator.
    pub fn new(numerator: u64, denominator: u64) -> Result<Self, StretchError> {
        Self::from_u128(numerator as u128, denominator as u128)
    }

    fn from_u128(p: u128, q: u128) -> Result<Self, StretchError> {
        if p == 0 || q == 0 {
            return Err(StretchError::ZeroRatio);
        }
        let g = gcd(p, q);
        let (p, q) = (p / g, q / g);
        let numerator = u64::try_from(p).map_err(|_| StretchError::Overflow)?;
        let denominator = u64::try_from(q).map_err(|_| StretchError::Overflow)?;
        Ok(Self {
            numerator,
            denominator: NonZeroU64::new(denominator).ok_or(StretchError::ZeroRatio)?,
        })
    }

    /// Parses `"p/q"` with positive integer parts.
    pub fn from_fraction_str(s: &str) -> Result<Self, StretchError> {
        let bad = || StretchError::MalformedRatio(s.to_string());
        let (a, b) = s.trim().split_once('/').ok_or_else(bad)?;
        let p = parse_uint(a.trim()).ok_or_else(bad)?;
        let q = parse_uint(b.trim()).ok_or_else(bad)?;
        Self::from_u128(p, q)
    }

    /// Parses a plain decimal such as `"1.25"` or `"0.75"` exactly (no binary floats).
    pub fn from_decimal_str(s: &str) -> Result<Self, StretchError> {
        let bad = || StretchError::MalformedRatio(s.to_string());
        let t = s.trim();
        let (int_part, frac_part) = match t.split_once('.') {
            Some((i, f)) => (i, f),
            None => (t, ""),
        };
        if int_part.is_empty() && frac_part.is_empty() {
            return Err(bad());
        }
        if frac_part.len() > 18 {
            return Err(bad());
        }
        let ip = if int_part.is_empty() {
            0
        } else {
            parse_uint(int_part).ok_or_else(bad)?
        };
        let fp = if frac_part.is_empty() {
            0
        } else {
            parse_uint(frac_part).ok_or_else(bad)?
        };
        let scale = 10u128.pow(frac_part.len() as u32);
        let p = ip
            .checked_mul(scale)
            .and_then(|v| v.checked_add(fp))
            .ok_or(StretchError::Overflow)?;
        Self::from_u128(p, scale)
    }

    /// Accepts either `p/q` or a decimal.
    pub fn parse(s: &str) -> Result<Self, StretchError> {
        if s.contains('/') {
            Self::from_fraction_str(s)
        } else {
            Self::from_decimal_str(s)
        }
    }

    pub fn is_identity(&self) -> bool {
        self.numerator == self.denominator.get()
    }

    /// Approximate value, used only for *preferences* and diagnostics, never for lengths.
    pub fn as_f64(&self) -> f64 {
        self.numerator as f64 / self.denominator.get() as f64
    }

    /// `floor(n_in * p / q + 1/2)` in checked `u128` (ties round up).
    ///
    /// `n_in * p < 2^128` always holds for `u64` operands; the rounding uses the
    /// remainder (`2 * rem >= q`) so no intermediate can overflow.
    pub fn target_frames_u128(&self, n_in: u64) -> Result<u128, StretchError> {
        let p = self.numerator as u128;
        let q = self.denominator.get() as u128;
        let a = (n_in as u128)
            .checked_mul(p)
            .ok_or(StretchError::Overflow)?;
        let (k, rem) = (a / q, a % q);
        let round_up = rem.checked_mul(2).ok_or(StretchError::Overflow)? >= q;
        k.checked_add(round_up as u128)
            .ok_or(StretchError::Overflow)
    }

    /// Target frame count as `u64`, or `Overflow` if it does not fit.
    pub fn target_frames(&self, n_in: u64) -> Result<u64, StretchError> {
        u64::try_from(self.target_frames_u128(n_in)?).map_err(|_| StretchError::Overflow)
    }

    /// True when the factor is outside the v0.1 audio-quality target `[1/2, 2]`.
    pub fn is_extreme(&self) -> bool {
        let p = self.numerator as u128;
        let q = self.denominator.get() as u128;
        2 * p < q || p > 2 * q
    }
}

fn parse_uint(s: &str) -> Option<u128> {
    if s.is_empty() || !s.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    s.parse().ok()
}

impl fmt::Display for StretchRatio {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}", self.numerator, self.denominator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_and_reduces() {
        let r = StretchRatio::from_fraction_str("10/4").unwrap();
        assert_eq!((r.numerator, r.denominator.get()), (5, 2));
        let r = StretchRatio::from_decimal_str("1.25").unwrap();
        assert_eq!((r.numerator, r.denominator.get()), (5, 4));
        let r = StretchRatio::from_decimal_str("0.75").unwrap();
        assert_eq!((r.numerator, r.denominator.get()), (3, 4));
        let r = StretchRatio::from_decimal_str(".5").unwrap();
        assert_eq!((r.numerator, r.denominator.get()), (1, 2));
        let r = StretchRatio::from_decimal_str("2").unwrap();
        assert_eq!((r.numerator, r.denominator.get()), (2, 1));
        let r = StretchRatio::from_decimal_str("1.").unwrap();
        assert!(r.is_identity());
    }

    #[test]
    fn rejects_bad_input() {
        for s in [
            "", ".", "0", "0.0", "0/3", "3/0", "-1", "1/-2", "a/b", "1.2.3", "1e3", " / ", "+1",
            "1/2/3",
        ] {
            assert!(StretchRatio::parse(s).is_err(), "{s:?} should fail");
        }
    }

    #[test]
    fn target_rounding_ties_up() {
        let r = StretchRatio::new(3, 2).unwrap();
        assert_eq!(r.target_frames(48_000).unwrap(), 72_000);
        // 3 * 1/2 = 1.5 -> 2 (tie rounds up)
        let half = StretchRatio::new(1, 2).unwrap();
        assert_eq!(half.target_frames(3).unwrap(), 2);
        assert_eq!(half.target_frames(1).unwrap(), 1);
        let third = StretchRatio::new(1, 3).unwrap();
        assert_eq!(third.target_frames(1).unwrap(), 0);
        assert_eq!(third.target_frames(2).unwrap(), 1);
    }

    #[test]
    fn rounding_matches_reference_definition() {
        for p in 1..25u64 {
            for q in 1..25u64 {
                let r = StretchRatio::new(p, q).unwrap();
                for n in 0..300u64 {
                    let reference = (2 * n * p + q) / (2 * q);
                    assert_eq!(r.target_frames(n).unwrap(), reference, "{n}*{p}/{q}");
                }
            }
        }
    }

    #[test]
    fn target_overflow_is_typed() {
        let r = StretchRatio::new(u64::MAX, 1).unwrap();
        assert_eq!(r.target_frames(u64::MAX), Err(StretchError::Overflow));
        assert!(r.target_frames_u128(u64::MAX).is_ok());
    }

    #[test]
    fn extreme_detection() {
        assert!(!StretchRatio::new(1, 2).unwrap().is_extreme());
        assert!(!StretchRatio::new(2, 1).unwrap().is_extreme());
        assert!(StretchRatio::new(1, 4).unwrap().is_extreme());
        assert!(StretchRatio::new(201, 100).unwrap().is_extreme());
    }
}

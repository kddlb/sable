//! User-facing processing options and quality presets.

use serde::Serialize;

/// How stereo input is rendered.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum StereoMode {
    /// Both channels get every renderer; sibilant noise uses the stereo covariance model.
    #[default]
    LeftRight,
    /// Mid = (L+R)/2 gets the full SABLE treatment; Side = (L−R)/2 is WSOLA only (same
    /// time map and grain positions as the mid) and is ducked by `side_duck_db` inside
    /// the mid's special events. Output is recombined to L = M+S, R = M−S.
    MidSide,
}

/// What happens to the spectral envelope when the pitch is shifted.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum FormantMode {
    /// Keep formants (and sibilant colour) where they were: same voice, new pitch.
    #[default]
    Preserve,
    /// Move formants with the pitch, like a tape-speed change ("chipmunk" upwards).
    Shift,
}

/// Trades CPU time for search breadth and analysis resolution.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub enum QualityPreset {
    Fast,
    #[default]
    Standard,
    High,
}

#[derive(Clone, Debug, Serialize)]
pub struct Options {
    /// Seed for the deterministic noise-resynthesis streams.
    pub seed: u64,
    pub quality: QualityPreset,
    /// When true the report also carries per-frame analysis features.
    pub diagnostics: bool,
    /// Hard limit on output frames per channel (memory guard).
    pub max_output_frames: u64,
    /// Enable anchored dry-copy of detected plosive bursts (and the burst-aware budget).
    pub enable_plosive: bool,
    /// Enable stochastic resynthesis of sustained sibilants.
    pub enable_sibilant: bool,
    /// Also return intermediate layers (baseline, overlays, masks) for inspection.
    pub keep_layers: bool,
    /// Skip event detection and use one uniform time map (generic WSOLA baseline).
    pub uniform_timing: bool,
    /// Stereo rendering strategy (ignored for mono).
    pub stereo_mode: StereoMode,
    /// Side-channel gain (dB, <= 0) inside resynthesized sibilants and protected burst
    /// cores in [`StereoMode::MidSide`]; follows the renderer masks smoothly.
    pub side_duck_db: f64,
    /// Pitch shift in semitones (`0` = none, the pure time stretch). Range ±24.
    /// Duration is still set by the stretch ratio alone.
    pub pitch_semitones: f64,
    /// Envelope handling for a nonzero `pitch_semitones`.
    pub formants: FormantMode,
    /// Experimental: where the local slow-down is strong (≥ 1.8×), render voiced grains
    /// as a blend of the two aligned pitch periods around the predicted source position
    /// instead of repeating one period and jumping. Off by default.
    pub period_blend: bool,
    /// Worker threads for the parallel stages (`0` = all cores, `1` = calling thread
    /// only). The output is bit-identical for every setting.
    #[serde(skip)]
    pub threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            seed: 0x5AB1_E000,
            quality: QualityPreset::Standard,
            diagnostics: false,
            max_output_frames: 1 << 32,
            enable_plosive: true,
            enable_sibilant: true,
            keep_layers: false,
            uniform_timing: false,
            stereo_mode: StereoMode::LeftRight,
            side_duck_db: -6.0,
            pitch_semitones: 0.0,
            formants: FormantMode::Preserve,
            period_blend: false,
            threads: 0,
        }
    }
}

/// Largest accepted `|pitch_semitones|` (two octaves).
pub const MAX_PITCH_SEMITONES: f64 = 24.0;

impl Options {
    /// Pitch factor `2^(semitones / 12)`, or `None` when no shift is requested.
    pub fn pitch_factor(&self) -> Option<f64> {
        (self.pitch_semitones != 0.0).then(|| (self.pitch_semitones / 12.0).exp2())
    }

    /// Generic WSOLA with every speech-specific policy disabled (comparison baseline).
    pub fn generic_wsola() -> Self {
        Self {
            enable_plosive: false,
            enable_sibilant: false,
            uniform_timing: true,
            ..Self::default()
        }
    }
}

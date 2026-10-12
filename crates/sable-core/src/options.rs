//! User-facing processing options and quality presets.

use serde::Serialize;

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
        }
    }
}

impl Options {
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

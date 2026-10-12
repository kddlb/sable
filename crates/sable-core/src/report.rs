//! Serializable diagnostics describing every decision made for one run.

use serde::Serialize;

use crate::budget::Stage;
use crate::options::StereoMode;
use crate::ratio::StretchRatio;
use crate::segments::{EventKind, SegmentScores};

/// Which top-level code path rendered the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum RenderPath {
    Empty,
    Identity,
    Silence,
    /// Analysis + allocator + WSOLA + special renderers.
    Full,
    /// Single uniform time map with the generic renderer (stage D or generic mode).
    Uniform,
}

/// Non-fatal conditions worth surfacing to the user.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub enum Warning {
    /// Factor outside the v0.1 quality target `[1/2, 2]`.
    ExtremeStretchFactor { ratio: f64 },
    /// Stage B: non-burst bounds were widened beyond the class table.
    RelaxedNonBurstBudget,
    /// Stage C: these burst segments did not keep their 1x duration.
    BurstProtectionWaived { segment_ids: Vec<usize> },
    /// Stage D: whole-clip uniform map.
    UniformFallback { reason: String },
    /// Short regions were merged so the segment count fits the output length.
    SegmentsCoalesced { from: usize, to: usize },
    /// A plosive-phase warp could not reach a 1x slope at the burst edge.
    WarpSlopeRelaxed {
        segment_id: usize,
        k_ideal: f64,
        k_used: f64,
    },
    /// Output samples not covered by any grain were filled by the safe fallback.
    UncoveredSamplesFilled { count: u64 },
    /// A special renderer was disabled for a segment; see the overlay list.
    SpecialRendererDisabled { segment_id: usize, reason: String },
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct SegmentReport {
    pub id: usize,
    pub kind: EventKind,
    pub confidence: f64,
    pub in_start: u64,
    pub in_len: u64,
    pub out_start: u64,
    pub out_len: u64,
    pub preferred_factor: f64,
    pub local_ratio: f64,
    pub warp_k: f64,
    pub anchor: Option<u64>,
    pub scores: SegmentScores,
    pub reasons: Vec<String>,
}

/// A plosive or sibilant overlay decision.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct OverlayReport {
    pub segment_id: usize,
    pub applied: bool,
    pub out_start: u64,
    pub out_end: u64,
    pub detail: String,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct WsolaStats {
    pub grains: u64,
    pub transient_grains: u64,
    /// Grains that used the predicted source position (no safe/energetic candidate).
    pub predicted_only: u64,
    pub mean_abs_offset: f64,
    pub uncovered_filled: u64,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChannelLevels {
    pub channel: usize,
    pub peak_in: f32,
    pub rms_in: f32,
    pub peak_out: f32,
    pub rms_out: f32,
}

/// Per-frame analysis features (only with `Options::diagnostics`).
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct FeatureDump {
    pub hop: usize,
    pub log_rms: Vec<f32>,
    pub flux: Vec<f32>,
    pub hf_ratio: Vec<f32>,
    pub flatness: Vec<f32>,
    pub centroid_hz: Vec<f32>,
    pub periodicity: Vec<f32>,
    pub f0_hz: Vec<f32>,
    pub zcr: Vec<f32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AnalysisReport {
    pub version: &'static str,
    pub sample_rate: u32,
    pub channels: usize,
    pub ratio: StretchRatio,
    pub ratio_f64: f64,
    pub n_in: u64,
    pub n_out: u64,
    pub actual_frames: u64,
    pub achieved_factor: f64,
    pub path: RenderPath,
    pub seed: u64,
    /// Effective stereo mode (always `LeftRight` for mono input).
    pub stereo_mode: StereoMode,
    /// Side duck depth actually applied (mid/side mode only).
    pub side_duck_db: Option<f64>,
    pub allocator_stage: Option<Stage>,
    pub warnings: Vec<Warning>,
    pub segments: Vec<SegmentReport>,
    pub plosive_overlays: Vec<OverlayReport>,
    pub sibilant_overlays: Vec<OverlayReport>,
    pub wsola: Option<WsolaStats>,
    pub levels: Vec<ChannelLevels>,
    pub features: Option<FeatureDump>,
}

impl AnalysisReport {
    pub fn new(
        sample_rate: u32,
        channels: usize,
        ratio: StretchRatio,
        n_in: u64,
        n_out: u64,
        seed: u64,
    ) -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION"),
            sample_rate,
            channels,
            ratio,
            ratio_f64: ratio.as_f64(),
            n_in,
            n_out,
            actual_frames: 0,
            achieved_factor: 0.0,
            path: RenderPath::Empty,
            seed,
            stereo_mode: StereoMode::LeftRight,
            side_duck_db: None,
            allocator_stage: None,
            warnings: Vec::new(),
            segments: Vec::new(),
            plosive_overlays: Vec::new(),
            sibilant_overlays: Vec::new(),
            wsola: None,
            levels: Vec::new(),
            features: None,
        }
    }
}

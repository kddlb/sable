//! Anchored dry-copy plosive renderer (spec §7). Implemented in M3.

use crate::audio::AudioView;
use crate::options::Options;
use crate::pipeline::Plan;
use crate::report::AnalysisReport;

/// Result of the plosive pass.
pub struct PlosiveResult {
    /// Event weight per output frame (1 inside protected cores, 0 outside wings).
    pub mask: Vec<f32>,
}

pub fn apply(
    _input: &AudioView<'_>,
    plan: &Plan,
    _opts: &Options,
    _out: &mut [Vec<f32>],
    _report: &mut AnalysisReport,
) -> PlosiveResult {
    PlosiveResult {
        mask: vec![0.0; plan.map.n_out as usize],
    }
}

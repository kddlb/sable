//! Stochastic sibilant resynthesis (spec §8). Implemented in M4.

use crate::audio::AudioView;
use crate::features::Features;
use crate::options::Options;
use crate::pipeline::Plan;
use crate::report::AnalysisReport;

/// Result of the sibilant pass.
pub struct SibilantResult {
    /// Synthetic weight per output frame.
    pub mask: Vec<f32>,
    /// Raw synthesized noise per channel (zero outside resynthesized events).
    pub noise: Vec<Vec<f32>>,
}

pub fn apply(
    input: &AudioView<'_>,
    _features: &Features,
    plan: &Plan,
    _opts: &Options,
    _plosive_mask: &[f32],
    _out: &mut [Vec<f32>],
    _report: &mut AnalysisReport,
) -> SibilantResult {
    let n = plan.map.n_out as usize;
    SibilantResult {
        mask: vec![0.0; n],
        noise: vec![vec![0.0; n]; input.channels.len()],
    }
}

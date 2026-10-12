//! # SABLE: Speech-Adaptive Budgeted Lengthening Engine
//!
//! Offline, pitch-preserving time-scale modification for speech. Different acoustic
//! events get different policies (dry-copied plosive bursts, stochastic sibilant
//! resynthesis, variable-rate WSOLA elsewhere), all reconciled against one
//! **sample-exact** output length:
//!
//! `N_out = floor(N_in * p / q + 1/2)` for stretch factor `r = p/q = duration_out / duration_in`.
//!
//! ```
//! use sable_core::{stretch, AudioView, Options, StretchRatio};
//! let tone: Vec<f32> = (0..4800).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
//! let chans = [tone.as_slice()];
//! let input = AudioView { sample_rate: 48_000, channels: &chans };
//! let ratio = StretchRatio::from_fraction_str("3/2").unwrap();
//! let out = stretch(&input, ratio, &Options::default()).unwrap();
//! assert_eq!(out.audio.channels[0].len(), 7200);
//! ```

#![forbid(unsafe_code)]

pub mod audio;
pub mod budget;
pub mod detect;
pub mod error;
pub mod features;
pub mod fft;
pub mod metrics;
pub mod options;
pub mod pipeline;
pub mod plosives;
pub mod ratio;
pub mod report;
pub mod segments;
pub mod sibilants;
pub mod synth;
pub mod timemap;
pub mod windows;
pub mod wsola;

pub use audio::{AudioBuffer, AudioView};
pub use error::StretchError;
pub use options::{Options, QualityPreset};
pub use ratio::StretchRatio;
pub use report::{AnalysisReport, RenderPath, Warning};

/// Output audio plus the diagnostic report.
#[derive(Clone, Debug)]
pub struct ProcessResult {
    pub audio: AudioBuffer,
    pub report: AnalysisReport,
    /// Intermediate layers, present when `Options::keep_layers` is set on the full path.
    pub layers: Option<Layers>,
}

/// Intermediate renders, all exactly `N_out` frames, for inspection.
#[derive(Clone, Debug)]
pub struct Layers {
    /// Variable-rate WSOLA output before any special renderer.
    pub baseline: AudioBuffer,
    /// Output after plosive overlays (before sibilants).
    pub after_plosive: AudioBuffer,
    /// Mono masks: plosive event weight and sibilant synthetic weight per output frame.
    pub plosive_mask: AudioBuffer,
    pub sibilant_mask: AudioBuffer,
    /// Raw synthesized sibilant noise (zero outside resynthesized events).
    pub sibilant_noise: AudioBuffer,
}

impl Layers {
    pub fn named(&self) -> [(&'static str, &AudioBuffer); 5] {
        [
            ("baseline", &self.baseline),
            ("after_plosive", &self.after_plosive),
            ("plosive_mask", &self.plosive_mask),
            ("sibilant_mask", &self.sibilant_mask),
            ("sibilant_noise", &self.sibilant_noise),
        ]
    }
}

/// Runs the shared analysis and event segmentation only (no rendering).
pub fn analyze_events(
    input: &AudioView<'_>,
) -> Result<(features::Features, detect::Detection), StretchError> {
    use detect::EventDetector;
    input.validate()?;
    let feats = features::analyze(
        input.channels,
        input.sample_rate,
        &features::FeatureConfig::default(),
    );
    let det = detect::HeuristicDetector::default().detect(&feats);
    Ok((feats, det))
}

/// Time-stretches `input` by `ratio` (= duration_out / duration_in).
///
/// Every successful return has exactly `ratio.target_frames(N_in)` frames per channel.
pub fn stretch(
    input: &AudioView<'_>,
    ratio: StretchRatio,
    options: &Options,
) -> Result<ProcessResult, StretchError> {
    input.validate()?;
    let nch = input.channels.len();
    let n_in = input.frames() as u64;
    let n_out_wide = ratio.target_frames_u128(n_in)?;
    if n_out_wide > options.max_output_frames as u128 {
        return Err(StretchError::OutputTooLarge {
            n_out: n_out_wide,
            limit: options.max_output_frames,
        });
    }
    let n_out = u64::try_from(n_out_wide).map_err(|_| StretchError::Overflow)?;
    if usize::try_from(n_out).is_err() {
        return Err(StretchError::OutputTooLarge {
            n_out: n_out_wide,
            limit: options.max_output_frames,
        });
    }
    let mut report = AnalysisReport::new(input.sample_rate, nch, ratio, n_in, n_out, options.seed);
    if ratio.is_extreme() {
        report.warnings.push(Warning::ExtremeStretchFactor {
            ratio: ratio.as_f64(),
        });
    }

    let mut layers = None;
    let channels: Vec<Vec<f32>> = if n_in == 0 {
        report.path = RenderPath::Empty;
        vec![Vec::new(); nch]
    } else if ratio.is_identity() {
        report.path = RenderPath::Identity;
        input.channels.iter().map(|c| c.to_vec()).collect()
    } else if n_out == 0 {
        return Err(StretchError::OutputTooShort { n_in });
    } else if input.is_all_zero() {
        report.path = RenderPath::Silence;
        vec![vec![0.0; n_out as usize]; nch]
    } else {
        let (out, l) = pipeline::render(input, ratio, n_out, options, &mut report)?;
        layers = l;
        out
    };

    let mut result = finish(input, channels, report)?;
    result.layers = layers;
    Ok(result)
}

fn finish(
    input: &AudioView<'_>,
    channels: Vec<Vec<f32>>,
    mut report: AnalysisReport,
) -> Result<ProcessResult, StretchError> {
    for (i, c) in channels.iter().enumerate() {
        if c.len() as u64 != report.n_out {
            return Err(StretchError::Internal(format!(
                "channel {i} rendered {} frames, expected {}",
                c.len(),
                report.n_out
            )));
        }
        if let Some(index) = c.iter().position(|v| !v.is_finite()) {
            return Err(StretchError::Internal(format!(
                "non-finite output sample in channel {i} at {index}"
            )));
        }
    }
    report.actual_frames = report.n_out;
    report.achieved_factor = if report.n_in == 0 {
        report.ratio_f64
    } else {
        report.n_out as f64 / report.n_in as f64
    };
    report.levels = channels
        .iter()
        .enumerate()
        .map(|(ch, c)| {
            let (peak_in, rms_in) = audio::peak_rms(input.channels[ch]);
            let (peak_out, rms_out) = audio::peak_rms(c);
            report::ChannelLevels {
                channel: ch,
                peak_in,
                rms_in,
                peak_out,
                rms_out,
            }
        })
        .collect();
    Ok(ProcessResult {
        audio: AudioBuffer {
            sample_rate: input.sample_rate,
            channels,
        },
        report,
        layers: None,
    })
}

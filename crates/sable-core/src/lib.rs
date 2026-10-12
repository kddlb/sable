//! # SABLE: Speech-Adaptive Budgeted Lengthening Engine
//!
//! Offline, pitch-preserving time-scale modification for speech. Different acoustic
//! events get different policies (dry-copied plosive bursts, stochastic sibilant
//! resynthesis, variable-rate WSOLA elsewhere), all reconciled against one
//! **sample-exact** output length:
//!
//! `N_out = floor(N_in * p / q + 1/2)` for stretch factor `r = p/q = duration_out / duration_in`.
//!
//! Optionally the pitch can be shifted at the same time ([`Options::pitch_semitones`]),
//! with or without formant preservation ([`FormantMode`]); the length rule is unchanged.
//! Use [`stretch_with_progress`] to follow a long render phase by phase.
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
pub mod filters;
pub mod formant;
pub mod metrics;
pub mod options;
pub mod par;
pub mod pipeline;
pub mod plosives;
pub mod progress;
pub mod ratio;
pub mod report;
pub mod resample;
pub mod rng;
pub mod segments;
pub mod sibilants;
pub mod stereo;
pub mod synth;
pub mod target;
pub mod timemap;
pub mod windows;
pub mod wsola;

pub use audio::{AudioBuffer, AudioView};
pub use error::StretchError;
pub use options::{FormantMode, MAX_PITCH_SEMITONES, Options, QualityPreset, StereoMode};
pub use progress::{Phase, ProgressEvent};
pub use ratio::StretchRatio;
pub use report::{AnalysisReport, PitchReport, RenderPath, Warning};
pub use target::{TargetDuration, TargetMode};

use progress::Progress;

/// Output audio plus the diagnostic report.
#[derive(Clone, Debug)]
pub struct ProcessResult {
    pub audio: AudioBuffer,
    pub report: AnalysisReport,
    /// Intermediate layers, present when `Options::keep_layers` is set on the full path.
    pub layers: Option<Layers>,
}

/// Intermediate renders, all exactly `N_out` frames, for inspection. With a pitch shift
/// they are resampled to `N_out` like the output but not formant-corrected.
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

/// Time-stretches `input` by `ratio` (= duration_out / duration_in), and shifts its
/// pitch when [`Options::pitch_semitones`] is nonzero.
///
/// Every successful return has exactly `ratio.target_frames(N_in)` frames per channel.
pub fn stretch(
    input: &AudioView<'_>,
    ratio: StretchRatio,
    options: &Options,
) -> Result<ProcessResult, StretchError> {
    run(input, ratio, options, &mut Progress::none())
}

/// [`stretch`], calling `on_progress` from the rendering thread as each [`Phase`] starts,
/// advances and finishes. The output is identical to [`stretch`].
pub fn stretch_with_progress(
    input: &AudioView<'_>,
    ratio: StretchRatio,
    options: &Options,
    on_progress: &mut dyn FnMut(ProgressEvent),
) -> Result<ProcessResult, StretchError> {
    run(input, ratio, options, &mut Progress::new(on_progress))
}

fn validated_pitch(options: &Options) -> Result<Option<f64>, StretchError> {
    let st = options.pitch_semitones;
    if !st.is_finite() || st.abs() > MAX_PITCH_SEMITONES {
        return Err(StretchError::InvalidPitch(st.to_string()));
    }
    Ok(options.pitch_factor())
}

fn run(
    input: &AudioView<'_>,
    ratio: StretchRatio,
    options: &Options,
    progress: &mut Progress<'_>,
) -> Result<ProcessResult, StretchError> {
    input.validate()?;
    let pitch = validated_pitch(options)?;
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
    } else if ratio.is_identity() && pitch.is_none() {
        report.path = RenderPath::Identity;
        input.channels.iter().map(|c| c.to_vec()).collect()
    } else if n_out == 0 {
        return Err(StretchError::OutputTooShort { n_in });
    } else if input.is_all_zero() {
        report.path = RenderPath::Silence;
        vec![vec![0.0; n_out as usize]; nch]
    } else if let Some(s) = pitch {
        let (out, l) = render_pitched(input, n_out, s, options, &mut report, progress)?;
        layers = l;
        out
    } else {
        let r = pipeline::render(input, ratio, n_out, options, &mut report, progress)?;
        layers = r.layers;
        r.channels
    };

    let mut result = finish(input, channels, report)?;
    result.layers = layers;
    Ok(result)
}

/// Pitch shift by `s`: SABLE stretch to `M = round(N_out · s)` frames (pitch kept),
/// optional envelope pre-warp, then band-limited resampling `M → N_out`, which scales
/// every frequency by `s` and restores the requested duration.
fn render_pitched(
    input: &AudioView<'_>,
    n_out: u64,
    s: f64,
    options: &Options,
    report: &mut AnalysisReport,
    progress: &mut Progress<'_>,
) -> Result<(Vec<Vec<f32>>, Option<Layers>), StretchError> {
    let n_in = input.frames() as u64;
    let m = ((n_out as f64 * s).round() as u64).max(1);
    if m > options.max_output_frames {
        return Err(StretchError::OutputTooLarge {
            n_out: m as u128,
            limit: options.max_output_frames,
        });
    }
    let internal = StretchRatio::new(m, n_in)?;
    if internal.is_extreme() {
        report.warnings.push(Warning::ExtremeInternalStretch {
            internal_ratio: internal.as_f64(),
        });
    }
    let rendered = pipeline::render(input, internal, m, options, report, progress)?;
    let mut channels = rendered.channels;
    let fs = input.sample_rate;

    let formant = (options.formants == FormantMode::Preserve).then(|| {
        progress.start(Phase::Formants);
        let threads = par::threads(options.threads);
        let st = formant::prewarp(
            &mut channels,
            fs,
            s,
            rendered.f0_ref_hz,
            threads,
            &mut |d, t| progress.step(Phase::Formants, d, t),
        );
        progress.finish(
            Phase::Formants,
            format!(
                "{} frames, lifter {:.2} ms (F0 ref {:.0} Hz), {:.1} iterations",
                st.frames, st.lifter_ms, st.f0_ref_hz, st.mean_iterations
            ),
        );
        st
    });

    progress.start(Phase::Resample);
    let rs = resample::Resampler::new(m as usize, n_out as usize).with_threads(options.threads);
    let n = n_out as usize;
    let mut layers = rendered.layers;
    let mut jobs: Vec<&mut Vec<f32>> = channels.iter_mut().collect();
    if let Some(l) = layers.as_mut() {
        for buf in [&mut l.plosive_mask, &mut l.sibilant_mask] {
            for c in &mut buf.channels {
                *c = resample::resample_linear(c, n);
            }
        }
        for buf in [&mut l.baseline, &mut l.after_plosive, &mut l.sibilant_noise] {
            jobs.extend(buf.channels.iter_mut());
        }
    }
    let total = jobs.len() * n;
    for (i, c) in jobs.into_iter().enumerate() {
        *c = rs.process(c, &mut |d, _| {
            progress.step(Phase::Resample, i * n + d, total)
        });
    }
    progress.finish(
        Phase::Resample,
        format!(
            "{m} → {n_out} frames (frequencies ×{:.4})",
            m as f64 / n_out as f64
        ),
    );

    // Report positions in final output frames.
    let to_out = |y: u64| ((2 * y as u128 * n_out as u128 + m as u128) / (2 * m as u128)) as u64;
    for seg in &mut report.segments {
        let (a, b) = (to_out(seg.out_start), to_out(seg.out_start + seg.out_len));
        seg.out_start = a;
        seg.out_len = b - a;
        seg.local_ratio = seg.out_len as f64 / seg.in_len as f64;
    }
    for o in report
        .plosive_overlays
        .iter_mut()
        .chain(report.sibilant_overlays.iter_mut())
    {
        o.out_start = to_out(o.out_start);
        o.out_end = to_out(o.out_end);
    }
    report.pitch = Some(PitchReport {
        semitones: options.pitch_semitones,
        factor: s,
        formants: options.formants,
        internal_frames: m,
        internal_ratio: internal,
        formant,
    });
    Ok((channels, layers))
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

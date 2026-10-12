//! Live progress reporting for long renders.
//!
//! [`crate::stretch_with_progress`] calls a user closure with [`ProgressEvent`]s as the
//! pipeline moves through its phases. Fractional updates are throttled to about one per
//! percent, so the closure can be cheap (e.g. send on a channel) without slowing the
//! render. Progress never influences the output.

use serde::Serialize;

/// One step of the pipeline, in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum Phase {
    /// Frame features: RMS, flux, spectral shape, periodicity, F0.
    Analysis,
    /// Event segmentation (bursts, sibilants, vowels, ...).
    Detection,
    /// Frame budget allocation and time-map construction.
    Allocation,
    /// Variable-rate WSOLA baseline.
    Wsola,
    /// Dry-copied plosive bursts.
    Plosives,
    /// Sibilant noise resynthesis.
    Sibilants,
    /// Spectral-envelope correction (formant-preserving pitch shift only).
    Formants,
    /// Band-limited resampling to the final length (pitch shift only).
    Resample,
}

impl Phase {
    pub const ALL: [Phase; 8] = [
        Phase::Analysis,
        Phase::Detection,
        Phase::Allocation,
        Phase::Wsola,
        Phase::Plosives,
        Phase::Sibilants,
        Phase::Formants,
        Phase::Resample,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Phase::Analysis => "Analysis",
            Phase::Detection => "Event detection",
            Phase::Allocation => "Budget allocation",
            Phase::Wsola => "WSOLA",
            Phase::Plosives => "Plosive overlays",
            Phase::Sibilants => "Sibilant resynthesis",
            Phase::Formants => "Formant correction",
            Phase::Resample => "Resampling",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ProgressEvent {
    Started(Phase),
    /// Fraction of the phase completed, in `[0, 1]`.
    Progress {
        phase: Phase,
        fraction: f32,
    },
    /// The phase finished; `detail` is a short human-readable summary.
    Finished {
        phase: Phase,
        detail: String,
    },
}

/// Throttling wrapper around an optional progress closure.
pub struct Progress<'a> {
    sink: Option<&'a mut dyn FnMut(ProgressEvent)>,
    last: f32,
}

impl<'a> Progress<'a> {
    pub fn new(sink: &'a mut dyn FnMut(ProgressEvent)) -> Self {
        Self {
            sink: Some(sink),
            last: -1.0,
        }
    }

    /// A sink that drops every event.
    pub fn none() -> Self {
        Self {
            sink: None,
            last: -1.0,
        }
    }

    pub fn start(&mut self, phase: Phase) {
        self.last = -1.0;
        if let Some(s) = self.sink.as_mut() {
            s(ProgressEvent::Started(phase));
        }
    }

    /// Reports `done / total` of `phase`, at most once per percent.
    pub fn step(&mut self, phase: Phase, done: usize, total: usize) {
        let Some(s) = self.sink.as_mut() else { return };
        let f = if total == 0 {
            1.0
        } else {
            (done as f32 / total as f32).clamp(0.0, 1.0)
        };
        if f - self.last >= 0.01 || (f >= 1.0 && self.last < 1.0) {
            self.last = f;
            s(ProgressEvent::Progress { phase, fraction: f });
        }
    }

    pub fn finish(&mut self, phase: Phase, detail: impl Into<String>) {
        if let Some(s) = self.sink.as_mut() {
            s(ProgressEvent::Finished {
                phase,
                detail: detail.into(),
            });
        }
    }

    /// True when a closure is attached (lets callers skip building detail strings).
    pub fn active(&self) -> bool {
        self.sink.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steps_are_throttled() {
        let mut events = Vec::new();
        let mut sink = |e| events.push(e);
        let mut p = Progress::new(&mut sink);
        p.start(Phase::Wsola);
        for i in 0..=10_000 {
            p.step(Phase::Wsola, i, 10_000);
        }
        p.finish(Phase::Wsola, "ok");
        let steps = events
            .iter()
            .filter(|e| matches!(e, ProgressEvent::Progress { .. }))
            .count();
        assert!((90..=102).contains(&steps), "{steps} progress events");
        assert_eq!(
            events.last(),
            Some(&ProgressEvent::Finished {
                phase: Phase::Wsola,
                detail: "ok".into()
            })
        );
    }
}

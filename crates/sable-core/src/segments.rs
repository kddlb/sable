//! Event classes and the nonoverlapping input partition.

use serde::Serialize;

/// Acoustic event class (not a phoneme label).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize)]
pub enum EventKind {
    Voiced,
    SibilantUnvoiced,
    SibilantVoiced,
    PlosiveClosure,
    PlosiveBurst,
    Aspiration,
    Transition,
    Silence,
    Other,
}

impl EventKind {
    pub fn is_sibilant(self) -> bool {
        matches!(self, Self::SibilantUnvoiced | Self::SibilantVoiced)
    }
}

/// Detector evidence carried with a segment for diagnostics.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct SegmentScores {
    pub mean_log_rms: f64,
    pub hf_ratio: f64,
    pub flatness: f64,
    pub periodicity: f64,
    pub peak_flux: f64,
    pub valley_depth_db: f64,
}

/// Half-open input interval `[start, end)` with a class and confidence.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Segment {
    pub id: usize,
    pub start: u64,
    pub end: u64,
    pub kind: EventKind,
    /// Detector confidence in `[0, 1]`.
    pub confidence: f64,
    /// Exact sample-level onset for bursts (equals `start` for a burst segment).
    pub anchor: Option<u64>,
    pub scores: SegmentScores,
    /// Human-readable reasons for the class and for enabling/disabling renderers.
    pub reasons: Vec<String>,
}

impl Segment {
    pub fn new(start: u64, end: u64, kind: EventKind, confidence: f64) -> Self {
        Self {
            id: 0,
            start,
            end,
            kind,
            confidence,
            anchor: None,
            scores: SegmentScores::default(),
            reasons: Vec::new(),
        }
    }

    pub fn len(&self) -> u64 {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// Checks that `segs` are sorted, contiguous, nonempty and cover `[0, n_in)` exactly.
pub fn validate_partition(segs: &[Segment], n_in: u64) -> Result<(), String> {
    if n_in == 0 {
        return if segs.is_empty() {
            Ok(())
        } else {
            Err("nonempty partition of empty input".into())
        };
    }
    let mut pos = 0;
    for (i, s) in segs.iter().enumerate() {
        if s.start != pos {
            return Err(format!(
                "segment {i} starts at {} but expected {pos}",
                s.start
            ));
        }
        if s.end <= s.start {
            return Err(format!("segment {i} is empty or reversed"));
        }
        pos = s.end;
    }
    if pos != n_in {
        return Err(format!("partition ends at {pos}, expected {n_in}"));
    }
    Ok(())
}

/// Renumbers segment IDs to their index.
pub fn renumber(segs: &mut [Segment]) {
    for (i, s) in segs.iter_mut().enumerate() {
        s.id = i;
    }
}

/// Merges adjacent non-burst segments until there are at most `max_count`, always
/// absorbing the shortest eligible segment into its shorter neighbour (lowest index
/// wins ties). Burst anchors are never merged away; returns `false` if the target
/// count cannot be met without doing so.
pub fn coalesce_to_count(segs: &mut Vec<Segment>, max_count: usize) -> bool {
    while segs.len() > max_count {
        let mut best: Option<(u64, usize)> = None;
        for (i, s) in segs.iter().enumerate() {
            if s.kind == EventKind::PlosiveBurst {
                continue;
            }
            let has_partner = (i > 0 && segs[i - 1].kind != EventKind::PlosiveBurst)
                || (i + 1 < segs.len() && segs[i + 1].kind != EventKind::PlosiveBurst);
            if !has_partner {
                continue;
            }
            if best.is_none_or(|(l, _)| s.len() < l) {
                best = Some((s.len(), i));
            }
        }
        let Some((_, i)) = best else {
            return false;
        };
        let left_ok = i > 0 && segs[i - 1].kind != EventKind::PlosiveBurst;
        let right_ok = i + 1 < segs.len() && segs[i + 1].kind != EventKind::PlosiveBurst;
        let into_left = match (left_ok, right_ok) {
            (true, true) => segs[i - 1].len() <= segs[i + 1].len(),
            (l, _) => l,
        };
        let victim = segs.remove(i);
        let target = if into_left {
            &mut segs[i - 1]
        } else {
            &mut segs[i]
        };
        absorb(target, &victim);
    }
    renumber(segs);
    true
}

/// Extends `target` to cover the adjacent `other`; keeps the longer segment's class.
pub fn absorb(target: &mut Segment, other: &Segment) {
    let keep_other_kind = other.len() > target.len() && target.kind != EventKind::PlosiveBurst;
    target.start = target.start.min(other.start);
    target.end = target.end.max(other.end);
    if keep_other_kind {
        target.kind = other.kind;
        target.confidence = other.confidence;
        target.scores = other.scores.clone();
    }
    target.reasons.push(format!(
        "absorbed {:?} [{}, {})",
        other.kind, other.start, other.end
    ));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn part(lens: &[(u64, EventKind)]) -> Vec<Segment> {
        let mut pos = 0;
        let mut v: Vec<Segment> = lens
            .iter()
            .map(|&(l, k)| {
                let s = Segment::new(pos, pos + l, k, 1.0);
                pos += l;
                s
            })
            .collect();
        renumber(&mut v);
        v
    }

    #[test]
    fn coalesce_keeps_bursts_and_partition() {
        use EventKind::*;
        let mut v = part(&[
            (5, Silence),
            (1, PlosiveBurst),
            (3, Aspiration),
            (2, Voiced),
            (1, PlosiveBurst),
            (4, Voiced),
        ]);
        // V2 merges into the aspiration; afterwards every non-burst is isolated by bursts.
        assert!(coalesce_to_count(&mut v.clone(), 5));
        assert!(!coalesce_to_count(&mut v.clone(), 4));
        assert!(coalesce_to_count(&mut v, 5));
        assert_eq!(v.len(), 5);
        validate_partition(&v, 16).unwrap();
        assert_eq!(v.iter().filter(|s| s.kind == PlosiveBurst).count(), 2);
        // Only bursts and 1 separator possible -> cannot go below 3.
        let mut w = part(&[(1, PlosiveBurst), (1, Voiced), (1, PlosiveBurst)]);
        assert!(!coalesce_to_count(&mut w, 2));
    }

    #[test]
    fn validate_rejects_gaps() {
        let mut v = part(&[(3, EventKind::Voiced), (3, EventKind::Voiced)]);
        v[1].start = 4;
        assert!(validate_partition(&v, 6).is_err());
    }
}

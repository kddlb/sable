//! Conservative heuristic acoustic-event segmentation (spec §4.2–4.3).
//!
//! This is *acoustic event recognition*, not phoneme recognition. Frame labels come
//! from a hysteresis state machine; plosive bursts come from a 1-ms pre-emphasised
//! energy envelope and are only protected when several cues corroborate. Anything
//! uncertain becomes `Transition`/`Other`, which receives generic WSOLA.

use serde::Serialize;

use crate::features::Features;
use crate::fft::RealFft;
use crate::segments::{EventKind, Segment, SegmentScores, renumber, validate_partition};
use crate::windows::{hann_periodic, ms_to_samples, pow2_near_ms};

#[derive(Clone, Debug, Serialize)]
pub struct DetectorConfig {
    /// Minimum 1-ms energy jump (dB) over the preceding ~20 ms for a burst candidate.
    pub burst_jump_db: f64,
    /// Confidence required to protect a burst.
    pub burst_confidence: f64,
    pub burst_min_ms: f64,
    pub burst_max_ms: f64,
    pub closure_min_ms: f64,
    pub closure_max_ms: f64,
    pub aspiration_max_ms: f64,
    pub sib_enter_hf: f32,
    pub sib_stay_hf: f32,
    pub sib_enter_flat: f32,
    pub sib_stay_flat: f32,
    pub sib_min_ms: f64,
    pub sib_confidence: f64,
    /// Interior 1-ms energy excess (dB over the run median) that marks an impulse.
    pub sib_impulse_db: f32,
    pub voiced_enter: f32,
    pub voiced_stay: f32,
    pub voiced_min_ms: f64,
    pub voiced_sib_low_periodicity: f32,
    pub voiced_sib_low_frac: f32,
    pub min_segment_ms: f64,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            burst_jump_db: 15.0,
            burst_confidence: 0.5,
            burst_min_ms: 2.0,
            burst_max_ms: 12.0,
            closure_min_ms: 10.0,
            closure_max_ms: 120.0,
            aspiration_max_ms: 60.0,
            sib_enter_hf: 0.55,
            sib_stay_hf: 0.42,
            sib_enter_flat: 0.30,
            sib_stay_flat: 0.20,
            sib_min_ms: 40.0,
            sib_confidence: 0.5,
            sib_impulse_db: 12.0,
            voiced_enter: 0.60,
            voiced_stay: 0.45,
            voiced_min_ms: 20.0,
            voiced_sib_low_periodicity: 0.55,
            voiced_sib_low_frac: 0.08,
            min_segment_ms: 10.0,
        }
    }
}

/// Evidence for one burst candidate (accepted or rejected), for diagnostics.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct BurstCandidate {
    pub onset: u64,
    pub end: u64,
    /// 1-ms pre-emphasised energy jump over the preceding ~20 ms.
    pub jump_db: f64,
    /// Time to come within 3 dB of the local maximum (diagnostic only).
    pub rise_ms: f64,
    /// Fraction (≥ 2 kHz) of the energy *added* at the onset relative to the 4 ms just
    /// before it; robust to a steady low-frequency background.
    pub hf_added: f64,
    /// Frame periodicity 5 ms after the onset (low = aperiodic release/aspiration).
    pub periodicity_after: f64,
    /// Peak 1-ms level in the first 3 ms minus the minimum 2-ms level 4–15 ms later.
    /// Negative = energy keeps growing (fricative/whisper onset).
    pub dip_db: f64,
    /// Peak 0.25-ms energy around the onset over the median 3–10 ms later.
    pub spike_db: f64,
    /// Highest frame periodicity 20–60 ms after the onset: voicing follows a stop release
    /// quickly, but not an /s/ or a whispered vowel.
    pub periodicity_follow: f64,
    pub valley_depth_db: Option<f64>,
    /// Closure level above the recording's noise floor (dB; diagnostic only).
    pub closure_above_floor_db: Option<f64>,
    pub c_jump: f64,
    pub c_aperiodic: f64,
    pub c_valley: f64,
    pub c_dip: f64,
    /// `max(dip, voicing-follows)` factor actually used.
    pub c_growth: f64,
    pub c_evidence: f64,
    pub confidence: f64,
    pub accepted: bool,
}

#[derive(Clone, Debug)]
pub struct Detection {
    pub segments: Vec<Segment>,
    pub burst_candidates: Vec<BurstCandidate>,
}

/// Replaceable detector interface.
pub trait EventDetector {
    fn detect(&self, features: &Features) -> Detection;
}

#[derive(Clone, Debug, Default)]
pub struct HeuristicDetector {
    pub cfg: DetectorConfig,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Lbl {
    Silence,
    Voiced,
    Sib,
    Unk,
}

/// High-band evidence for frication. For frames with a periodic, energetic low band
/// (voiced fricatives) the ratio is taken relative to the band above `low_band_hz`, so
/// that voicing energy does not mask the aperiodic high band.
fn sib_hf(f: &Features, j: usize, c: &DetectorConfig) -> f32 {
    let voiced_low = f.periodicity_low[j] > c.voiced_sib_low_periodicity
        && f.low_frac[j] > c.voiced_sib_low_frac;
    if voiced_low && f.hf_ratio[j] > 0.25 {
        f.hf_ratio[j].max(f.hf_ratio_upper[j])
    } else {
        f.hf_ratio[j]
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn percentile(v: &[f32], p: f64) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mut s = v.to_vec();
    s.sort_by(f32::total_cmp);
    s[((s.len() - 1) as f64 * p).round() as usize]
}

impl EventDetector for HeuristicDetector {
    fn detect(&self, f: &Features) -> Detection {
        let n = f.n as u64;
        if n == 0 {
            return Detection {
                segments: vec![],
                burst_candidates: vec![],
            };
        }
        let labels = self.frame_labels(f);
        let mut segs = self.runs_to_segments(f, &labels);
        let burst_candidates = self.find_bursts(f);
        for b in burst_candidates.iter().filter(|b| b.accepted) {
            self.paint_stop(f, &labels, &mut segs, b);
        }
        self.cleanup(f, &mut segs);
        for s in &mut segs {
            if s.kind != EventKind::PlosiveBurst {
                s.scores = scores_for(f, s.start as usize, s.end as usize);
            }
        }
        renumber(&mut segs);
        debug_assert!(validate_partition(&segs, n).is_ok());
        Detection {
            segments: segs,
            burst_candidates,
        }
    }
}

fn scores_for(f: &Features, a: usize, b: usize) -> SegmentScores {
    let ja = f.frame_of(a);
    let jb = f.frame_of(b.saturating_sub(1)).max(ja);
    let k = (jb - ja + 1) as f64;
    let mean = |v: &[f32]| v[ja..=jb].iter().map(|&x| x as f64).sum::<f64>() / k;
    SegmentScores {
        mean_log_rms: mean(&f.log_rms),
        hf_ratio: mean(&f.hf_ratio),
        flatness: mean(&f.flatness),
        periodicity: mean(&f.periodicity),
        peak_flux: f.flux[ja..=jb].iter().fold(0.0f32, |m, &x| m.max(x)) as f64,
        valley_depth_db: 0.0,
    }
}

/// Replaces `[start, end)` in a sorted contiguous partition by `seg`.
fn paint(segs: &mut Vec<Segment>, seg: Segment) {
    let (a, b) = (seg.start, seg.end);
    if a >= b {
        return;
    }
    let mut out = Vec::with_capacity(segs.len() + 2);
    let mut inserted = false;
    for s in segs.drain(..) {
        if s.end <= a || s.start >= b {
            if s.start >= b && !inserted {
                out.push(seg.clone());
                inserted = true;
            }
            out.push(s);
            continue;
        }
        if s.start < a {
            let mut left = s.clone();
            left.end = a;
            out.push(left);
        }
        if !inserted {
            out.push(seg.clone());
            inserted = true;
        }
        if s.end > b {
            let mut right = s;
            right.start = b;
            out.push(right);
        }
    }
    if !inserted {
        out.push(seg);
    }
    *segs = out;
}

impl HeuristicDetector {
    fn frame_labels(&self, f: &Features) -> Vec<Lbl> {
        let c = &self.cfg;
        let mut out = Vec::with_capacity(f.frames());
        let mut prev = Lbl::Silence;
        for j in 0..f.frames() {
            let l = if f.is_silent(j) {
                Lbl::Silence
            } else {
                let hf = sib_hf(f, j, c);
                let fl = f.flatness[j];
                let sib = if prev == Lbl::Sib {
                    hf > c.sib_stay_hf && fl > c.sib_stay_flat
                } else {
                    hf > c.sib_enter_hf && fl > c.sib_enter_flat
                };
                let p = f.periodicity[j];
                let voiced = if prev == Lbl::Voiced {
                    p > c.voiced_stay
                } else {
                    p > c.voiced_enter
                };
                if sib {
                    Lbl::Sib
                } else if voiced {
                    Lbl::Voiced
                } else {
                    Lbl::Unk
                }
            };
            out.push(l);
            prev = l;
        }
        out
    }

    fn runs_to_segments(&self, f: &Features, labels: &[Lbl]) -> Vec<Segment> {
        let c = &self.cfg;
        let n = f.n as u64;
        let hop = f.hop as u64;
        let sib_min = ms_to_samples(c.sib_min_ms, f.fs) as u64;
        let voiced_min = ms_to_samples(c.voiced_min_ms, f.fs) as u64;
        let long_unk = ms_to_samples(50.0, f.fs) as u64;
        let mut segs = Vec::new();
        let mut j = 0;
        while j < labels.len() {
            let mut k = j;
            while k < labels.len() && labels[k] == labels[j] {
                k += 1;
            }
            let start = (j as u64 * hop).min(n);
            let end = (k as u64 * hop).min(n);
            if end > start {
                let len = end - start;
                let mut seg = match labels[j] {
                    Lbl::Silence => {
                        let mut s = Segment::new(start, end, EventKind::Silence, 0.9);
                        s.reasons
                            .push("frame energy below adaptive silence threshold".into());
                        s
                    }
                    Lbl::Voiced => {
                        let conf = mean(&f.periodicity[j..k]);
                        if len >= voiced_min {
                            Segment::new(start, end, EventKind::Voiced, conf)
                        } else {
                            let mut s = Segment::new(start, end, EventKind::Transition, conf);
                            s.reasons.push("periodic run shorter than minimum".into());
                            s
                        }
                    }
                    Lbl::Sib => self.classify_sib(f, j, k, start, end, sib_min),
                    Lbl::Unk => {
                        let kind = if len < long_unk {
                            EventKind::Transition
                        } else {
                            EventKind::Other
                        };
                        let mut s = Segment::new(start, end, kind, 0.5);
                        s.reasons
                            .push("neither periodic, sibilant nor silent".into());
                        s
                    }
                };
                seg.scores = scores_for(f, start as usize, end as usize);
                segs.push(seg);
            }
            j = k;
        }
        if let Some(last) = segs.last_mut() {
            last.end = n;
        }
        segs
    }

    fn classify_sib(
        &self,
        f: &Features,
        j: usize,
        k: usize,
        start: u64,
        end: u64,
        sib_min: u64,
    ) -> Segment {
        let c = &self.cfg;
        let len = end - start;
        let conf_frames: Vec<f64> = (j..k)
            .map(|i| {
                let a = ((sib_hf(f, i, c) as f64 - 0.35) / 0.35).clamp(0.0, 1.0);
                let b = ((f.flatness[i] as f64 - 0.15) / 0.35).clamp(0.0, 1.0);
                a.min(b)
            })
            .collect();
        let conf = conf_frames.iter().sum::<f64>() / conf_frames.len() as f64;
        let voiced_frames = (j..k)
            .filter(|&i| {
                f.periodicity_low[i] > c.voiced_sib_low_periodicity
                    && f.low_frac[i] > c.voiced_sib_low_frac
            })
            .count();
        let voiced = voiced_frames * 2 >= k - j;
        let kind = if voiced {
            EventKind::SibilantVoiced
        } else {
            EventKind::SibilantUnvoiced
        };
        let mut s = Segment::new(start, end, kind, conf);
        if len < sib_min {
            s.kind = EventKind::Transition;
            s.reasons.push(format!(
                "sibilant-like run of {len} samples is shorter than minimum core"
            ));
            return s;
        }
        if conf < c.sib_confidence {
            s.kind = EventKind::Other;
            s.reasons
                .push(format!("sibilant confidence {conf:.2} below threshold"));
            return s;
        }
        // Interior impulse check (excluding 10-ms guards).
        let guard = ms_to_samples(10.0, f.fs);
        let (a, b) = (start as usize + guard, (end as usize).saturating_sub(guard));
        if b > a {
            let (ma, mb) = (a / f.fine_block, (b / f.fine_block).min(f.fine_db.len()));
            if mb > ma {
                let run = &f.fine_db[ma..mb];
                let med = percentile(run, 0.5);
                let mx = run.iter().fold(f32::MIN, |m, &v| m.max(v));
                if mx - med > c.sib_impulse_db {
                    s.kind = EventKind::Other;
                    s.reasons
                        .push(format!("interior impulse {:.1} dB above median", mx - med));
                    return s;
                }
            }
        }
        s.reasons.push(format!(
            "sustained high-band noise: hf={:.2} flat={:.2}, {} of {} frames low-band periodic",
            mean(&f.hf_ratio[j..k]),
            mean(&f.flatness[j..k]),
            voiced_frames,
            k - j
        ));
        s
    }

    /// 1-ms envelope transient search with corroborating cues.
    fn find_bursts(&self, f: &Features) -> Vec<BurstCandidate> {
        let c = &self.cfg;
        let e = &f.fine_db;
        let nb = e.len();
        let bl = f.fine_block;
        if nb < 4 {
            return vec![];
        }
        let floor = percentile(e, 0.10);
        let level = percentile(e, 0.95);
        let pre_len = 20usize.min(nb);
        let lin = |db: f32| 10f64.powf(db as f64 / 10.0);
        let mut out = Vec::new();
        let refractory = ms_to_samples(20.0, f.fs) / bl;
        let short_fft = pow2_near_ms(4.0, f.fs).max(64);
        let mut sfft = RealFft::new(short_fft);
        let swin = hann_periodic(short_fft);
        let mut m = 0;
        while m < nb {
            let pre = if m >= 2 {
                let a = m.saturating_sub(2 + pre_len);
                let b = m - 2;
                if b > a {
                    10.0 * ((a..b).map(|i| lin(e[i])).sum::<f64>() / (b - a) as f64).log10()
                } else {
                    floor as f64
                }
            } else {
                floor as f64
            };
            let pre = pre.max(floor as f64);
            let post = e[m..(m + 3).min(nb)]
                .iter()
                .fold(f32::MIN, |a, &v| a.max(v)) as f64;
            let jump = post - pre;
            if jump < c.burst_jump_db || post < level as f64 - 35.0 {
                m += 1;
                continue;
            }
            // Peak sample in the candidate neighbourhood of the pre-emphasised signal.
            let x = &f.analysis;
            let d = |i: usize| x[i] as f64 - if i > 0 { 0.97 * x[i - 1] as f64 } else { 0.0 };
            let lo = (m * bl).saturating_sub(bl);
            let hi = ((m + 3) * bl).min(f.n);
            let mut p = lo;
            for i in lo..hi {
                if d(i).abs() > d(p).abs() {
                    p = i;
                }
            }
            let thr = 0.2 * d(p).abs();
            let scan = p.saturating_sub(2 * bl);
            let mut onset = p;
            for i in scan..=p {
                if d(i).abs() >= thr {
                    onset = i;
                    break;
                }
            }
            onset = onset.saturating_sub(ms_to_samples(0.25, f.fs).min(onset));
            // Rise: time to come within 3 dB of the max over the next 6 ms.
            let mb = onset / bl;
            let w6 = ms_to_samples(6.0, f.fs).div_ceil(bl).max(1);
            let span_end = (mb + w6).min(nb);
            let local_max = e[mb..span_end].iter().fold(f32::MIN, |a, &v| a.max(v));
            let reach = (mb..span_end)
                .find(|&i| e[i] >= local_max - 3.0)
                .unwrap_or(mb);
            let rise_ms = (reach - mb) as f64 * bl as f64 * 1000.0 / f.fs as f64;
            // End of burst core.
            let peak_blk = (mb..span_end).find(|&i| e[i] >= local_max).unwrap_or(mb);
            let min_len = ms_to_samples(c.burst_min_ms, f.fs);
            let max_len = ms_to_samples(c.burst_max_ms, f.fs);
            let end_blk = (peak_blk..nb)
                .find(|&i| e[i] < local_max - 10.0)
                .unwrap_or(nb);
            let end = (end_blk * bl)
                .clamp(onset + min_len, onset + max_len)
                .min(f.n);
            // High-band share of the energy *added* at the onset (≈4 ms window vs the 4 ms
            // before it, ignoring < 300 Hz so a steady low-frequency bed does not dominate).
            let k2 =
                ((2000.0 * short_fft as f64 / f.fs as f64).round() as usize).min(short_fft / 2);
            let k0 = ((300.0 * short_fft as f64 / f.fs as f64).round() as usize).min(k2);
            let mut bands = |start: isize| {
                let mut buf = vec![0f32; short_fft];
                for (k, v) in buf.iter_mut().enumerate() {
                    let i = start + k as isize;
                    *v = if i >= 0 {
                        *x.get(i as usize).unwrap_or(&0.0)
                    } else {
                        0.0
                    };
                }
                let spec = sfft.forward_windowed(&buf, &swin);
                let lo: f64 = spec[k0..k2].iter().map(|c| c.norm_sqr() as f64).sum();
                let hi: f64 = spec[k2..].iter().map(|c| c.norm_sqr() as f64).sum();
                (lo, hi)
            };
            let (lo_post, hi_post) = bands(onset as isize);
            let (lo_pre, hi_pre) =
                bands(onset as isize - short_fft as isize - ms_to_samples(1.0, f.fs) as isize);
            let add_hi = (hi_post - hi_pre).max(0.0);
            let add_lo = (lo_post - lo_pre).max(0.0);
            let hf_added = add_hi / (add_hi + add_lo + 1e-30);
            // Aperiodicity just after the release (voiceless release / aspiration).
            let periodicity_after =
                f.periodicity[f.frame_of((onset + ms_to_samples(5.0, f.fs)).min(f.n - 1))] as f64;
            // Voicing within 20–60 ms after the release.
            let periodicity_follow = {
                let ja = f.frame_of((onset + ms_to_samples(20.0, f.fs)).min(f.n - 1));
                let jb = f.frame_of((onset + ms_to_samples(60.0, f.fs)).min(f.n - 1));
                f.periodicity[ja..=jb].iter().fold(0f32, |a, &v| a.max(v)) as f64
            };
            // Burst-then-dip on the 1-ms envelope.
            let ob = onset / bl;
            let peak3 = e[ob..(ob + 3).min(nb)]
                .iter()
                .fold(f32::MIN, |a, &v| a.max(v));
            let dmin = ((ob + 4)..(ob + 15).min(nb.saturating_sub(1)))
                .map(|m| 0.5 * (e[m] + e[m + 1]))
                .fold(f32::MAX, f32::min);
            let dip_db = if dmin == f32::MAX {
                0.0
            } else {
                (peak3 - dmin) as f64
            };
            // Impulsiveness on 0.25-ms blocks.
            let q = ms_to_samples(0.25, f.fs).max(1);
            let qe = |start: isize| {
                let mut acc = 0f64;
                for k in 0..q as isize {
                    let i = start + k;
                    if i > 0 && (i as usize) < x.len() {
                        let v = x[i as usize] as f64 - 0.97 * x[i as usize - 1] as f64;
                        acc += v * v;
                    }
                }
                acc / q as f64 + 1e-20
            };
            let one = ms_to_samples(1.0, f.fs) as isize;
            let on = onset as isize;
            let mut pk = 0f64;
            let mut t = on - one;
            while t < on + 3 * one {
                pk = pk.max(qe(t));
                t += q as isize;
            }
            let mut tail = Vec::new();
            let mut t = on + 3 * one;
            while t < on + 10 * one {
                tail.push(qe(t));
                t += q as isize;
            }
            tail.sort_by(f64::total_cmp);
            let spike_db = 10.0 * (pk / tail[tail.len() / 2]).log10();
            // Closure valley before the onset.
            let gap = ms_to_samples(10.0, f.fs);
            let back = ms_to_samples(c.closure_max_ms, f.fs);
            let mut closure_above_floor = None;
            let valley_depth = if onset >= gap + f.hop {
                let ja = f.frame_of(onset.saturating_sub(back));
                let jb = f.frame_of(onset - gap);
                let vmin = f.log_rms[ja..=jb].iter().fold(f32::MAX, |a, &v| a.min(v));
                let jc = f.frame_of(onset);
                let jd = f.frame_of((onset + gap).min(f.n - 1));
                let vmax = f.log_rms[jc..=jd].iter().fold(f32::MIN, |a, &v| a.max(v));
                closure_above_floor = Some((vmin - f.noise_floor_db) as f64);
                Some((vmax - vmin) as f64)
            } else {
                None
            };
            // Evidence model (tuned on labelled Spanish/English stops, see docs):
            // a stop release is aperiodic right after the onset, follows a closure valley,
            // does not keep growing like a fricative or whispered-vowel onset, and is either
            // impulsive or adds high-band energy.
            let c_jump = sigmoid((jump - (c.burst_jump_db - 1.0)) / 3.0).sqrt();
            let c_aperiodic = sigmoid((0.62 - periodicity_after) / 0.05);
            // Closure depth. (A "near the noise floor" alternative was tried for noisy phone
            // recordings and did not separate stops from vowel onsets; see COMPROMISES §5.)
            let c_valley = valley_depth.map_or(0.6, |d| sigmoid((d - 7.0) / 2.0));
            let c_dip = sigmoid((dip_db - 2.0) / 1.5);
            // Fricative and whispered-vowel onsets keep growing smoothly and stay aperiodic.
            // A stop shows a dip after the release, OR quick voicing (weak unaspirated bursts
            // run straight into a louder vowel), OR a clearly impulsive release (affricates).
            let c_growth = c_dip
                .max(sigmoid((periodicity_follow - 0.6) / 0.05))
                .max(sigmoid((spike_db - 3.0) / 1.5));
            let c_evidence = sigmoid((spike_db - 1.0) / 1.5).max(sigmoid((hf_added - 0.3) / 0.07));
            let confidence = c_jump * c_aperiodic * c_valley * c_growth * c_evidence;
            out.push(BurstCandidate {
                onset: onset as u64,
                end: end as u64,
                jump_db: jump,
                rise_ms,
                hf_added,
                periodicity_after,
                dip_db,
                spike_db,
                periodicity_follow,
                valley_depth_db: valley_depth,
                closure_above_floor_db: closure_above_floor,
                c_jump,
                c_aperiodic,
                c_valley,
                c_dip,
                c_growth,
                c_evidence,
                confidence,
                accepted: confidence >= c.burst_confidence,
            });
            m = (end / bl).max(m + 1) + refractory;
        }
        // Accepted bursts must not overlap each other.
        let mut last_end = 0;
        for b in &mut out {
            if b.accepted && b.onset < last_end {
                b.accepted = false;
            }
            if b.accepted {
                last_end = b.end;
            }
        }
        out
    }

    /// Paints closure, burst and aspiration around an accepted burst.
    fn paint_stop(
        &self,
        f: &Features,
        labels: &[Lbl],
        segs: &mut Vec<Segment>,
        b: &BurstCandidate,
    ) {
        let c = &self.cfg;
        let (onset, end) = (b.onset, b.end);
        // Closure: low-energy frames right before the onset.
        if let Some(depth) = b.valley_depth_db.filter(|&d| d > 6.0) {
            let jo = f.frame_of(onset as usize);
            let max_back = ms_to_samples(c.closure_max_ms, f.fs) as u64;
            let level_after = f.log_rms
                [jo..=f.frame_of((onset as usize + ms_to_samples(10.0, f.fs)).min(f.n - 1))]
                .iter()
                .fold(f32::MIN, |a, &v| a.max(v));
            let thr = level_after - (depth as f32) * 0.5;
            // Only frames whose RMS window ends before the onset may vote for a closure.
            let rms_half = ms_to_samples(10.0, f.fs) as u64;
            let hop = f.hop as u64;
            let mut start = onset;
            if let Some(last) = onset.checked_sub(rms_half + hop / 2) {
                let mut j = (last / hop) as usize;
                loop {
                    let cand = j as u64 * hop;
                    if onset - cand > max_back || f.log_rms[j] > thr {
                        break;
                    }
                    start = cand;
                    if j == 0 {
                        break;
                    }
                    j -= 1;
                }
            }
            if onset - start >= ms_to_samples(c.closure_min_ms, f.fs) as u64 {
                let mut s = Segment::new(start, onset, EventKind::PlosiveClosure, b.c_valley);
                s.reasons.push(format!(
                    "closure valley {depth:.1} dB before burst at {onset}"
                ));
                paint(segs, s);
            }
        }
        let mut s = Segment::new(onset, end, EventKind::PlosiveBurst, b.confidence);
        s.anchor = Some(onset);
        s.scores = SegmentScores {
            peak_flux: b.jump_db,
            valley_depth_db: b.valley_depth_db.unwrap_or(0.0),
            hf_ratio: b.hf_added,
            periodicity: b.periodicity_after,
            ..SegmentScores::default()
        };
        s.reasons.push(format!(
            "burst: jump {:.1} dB, periodicity after {:.2}, dip {:.1} dB, spike {:.1} dB,              added-HF {:.2}, valley {}, confidence {:.2}",
            b.jump_db,
            b.periodicity_after,
            b.dip_db,
            b.spike_db,
            b.hf_added,
            b.valley_depth_db
                .map_or("-".into(), |v| format!("{v:.1} dB")),
            b.confidence
        ));
        paint(segs, s);
        // Aspiration: following aperiodic, non-silent frames. Short sibilant-like runs count
        // as aspiration; a sustained (>= minimum core) sibilant run stops the search.
        let max_asp = ms_to_samples(c.aspiration_max_ms, f.fs) as u64;
        let sib_min_frames = ms_to_samples(c.sib_min_ms, f.fs).div_ceil(f.hop);
        let mut j = (end as usize).div_ceil(f.hop);
        let mut asp_end = end;
        let sib_run_from = |j: usize| labels[j..].iter().take_while(|&&l| l == Lbl::Sib).count();
        while j < labels.len()
            && (labels[j] == Lbl::Unk
                || (labels[j] == Lbl::Sib && sib_run_from(j) < sib_min_frames))
        {
            let cand = ((j as u64 + 1) * f.hop as u64).min(f.n as u64);
            if cand - end > max_asp {
                break;
            }
            asp_end = cand;
            j += 1;
        }
        if asp_end - end >= ms_to_samples(5.0, f.fs) as u64 {
            let mut s = Segment::new(end, asp_end, EventKind::Aspiration, 0.6);
            s.reasons.push("aperiodic noise following burst".into());
            paint(segs, s);
        }
    }

    /// Merges tiny regions and adjacent equal classes; never merges away a burst.
    fn cleanup(&self, f: &Features, segs: &mut Vec<Segment>) {
        let min_len = ms_to_samples(self.cfg.min_segment_ms, f.fs) as u64;
        loop {
            // Merge equal adjacent non-burst classes.
            let mut i = 0;
            while i + 1 < segs.len() {
                if segs[i].kind == segs[i + 1].kind && segs[i].kind != EventKind::PlosiveBurst {
                    let next = segs.remove(i + 1);
                    let total = (segs[i].len() + next.len()) as f64;
                    let ci = segs[i].confidence * segs[i].len() as f64;
                    segs[i].confidence = (ci + next.confidence * next.len() as f64) / total;
                    segs[i].end = next.end;
                    segs[i].reasons.extend(next.reasons);
                } else {
                    i += 1;
                }
            }
            // Shortest tiny non-burst segment that has a non-burst neighbour.
            let mut best: Option<usize> = None;
            for (i, s) in segs.iter().enumerate() {
                if s.kind == EventKind::PlosiveBurst || s.len() >= min_len {
                    continue;
                }
                let left = i > 0 && segs[i - 1].kind != EventKind::PlosiveBurst;
                let right = i + 1 < segs.len() && segs[i + 1].kind != EventKind::PlosiveBurst;
                if (left || right) && best.is_none_or(|b| s.len() < segs[b].len()) {
                    best = Some(i);
                }
            }
            let Some(i) = best else { break };
            let left = i > 0 && segs[i - 1].kind != EventKind::PlosiveBurst;
            let right = i + 1 < segs.len() && segs[i + 1].kind != EventKind::PlosiveBurst;
            let into_left = match (left, right) {
                (true, true) => {
                    // Prefer a neighbour of the same class, else the longer one.
                    if segs[i - 1].kind == segs[i].kind {
                        true
                    } else if segs[i + 1].kind == segs[i].kind {
                        false
                    } else {
                        segs[i - 1].len() >= segs[i + 1].len()
                    }
                }
                (l, _) => l,
            };
            let victim = segs.remove(i);
            let t = if into_left { i - 1 } else { i };
            crate::segments::absorb(&mut segs[t], &victim);
        }
    }
}

fn mean(v: &[f32]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64
    }
}

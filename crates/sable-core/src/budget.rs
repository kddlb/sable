//! Exact-frame constrained duration allocator (spec §5).
//!
//! Pure integer/real arithmetic, no DSP. Given per-interval input lengths, class
//! preferences, admissible bounds and compliances, it returns integer output
//! durations `D_i` with `sum(D_i) == N_out` and `lo_i <= D_i <= hi_i`.

use serde::Serialize;

use crate::segments::EventKind;

/// One interval presented to the allocator.
#[derive(Clone, Debug, PartialEq)]
pub struct AllocItem {
    /// Input length `L_i` in frames (> 0).
    pub len: u64,
    /// Preferred local factor `a_i` (already clamped to `[lo_factor, hi_factor]`).
    pub preferred: f64,
    pub lo_factor: f64,
    pub hi_factor: f64,
    /// Willingness to absorb extra duration (`c_i`).
    pub compliance: f64,
    /// Protected burst: stage A/B keep `D_i == L_i`.
    pub fixed: bool,
}

/// Relaxation stage that produced the allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub enum Stage {
    /// Normal class bounds.
    A,
    /// Non-burst bounds widened, bursts fixed (`RelaxedNonBurstBudget`).
    B,
    /// Burst protection waived as well (`BurstProtectionWaived`).
    C,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Allocation {
    pub durations: Vec<u64>,
    pub stage: Stage,
    /// Final integer bounds that were enforced.
    pub lo: Vec<u64>,
    pub hi: Vec<u64>,
    /// Indices of `fixed` items whose duration differs from their input length.
    pub relaxed_fixed: Vec<usize>,
    /// Lagrange multiplier found by the solver (diagnostic).
    pub lambda: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AllocError {
    /// More segments than output frames; caller must coalesce or use stage D.
    TooManySegments { segments: usize, n_out: u64 },
    /// No stage is feasible (should not happen when `segments <= n_out`).
    Infeasible,
    /// An item has zero length.
    EmptyItem(usize),
}

/// Starting class policies (spec §5.1 table). Returns `(preferred, lo, hi, compliance)`.
pub fn class_policy(kind: EventKind, r: f64) -> (f64, f64, f64, f64) {
    let (pref, lo, hi, c) = match kind {
        EventKind::PlosiveBurst => (1.0, 1.0, 1.0, 0.0),
        EventKind::PlosiveClosure => (1.0 + 0.12 * (2.0 * (r - 1.0)).tanh(), 0.85, 1.15, 0.05),
        EventKind::Aspiration => (1.0 + 0.50 * (r - 1.0), 0.65, 1.65, 0.30),
        EventKind::SibilantUnvoiced | EventKind::SibilantVoiced => {
            (1.0 + 0.65 * (r - 1.0), 0.55, 2.50, 0.70)
        }
        EventKind::Voiced => (r, 0.40, 3.00, 1.00),
        EventKind::Transition | EventKind::Other => (1.0 + 0.80 * (r - 1.0), 0.55, 2.40, 0.60),
        EventKind::Silence => (1.0 + 1.25 * (r - 1.0), 0.25, 3.00, 1.50),
    };
    (pref.clamp(lo, hi), lo, hi, c)
}

/// Builds an [`AllocItem`] from the class policy table.
pub fn item_for(kind: EventKind, len: u64, r: f64, protect_burst: bool) -> AllocItem {
    let effective = if kind == EventKind::PlosiveBurst && !protect_burst {
        EventKind::Transition
    } else {
        kind
    };
    let (preferred, lo_factor, hi_factor, compliance) = class_policy(effective, r);
    AllocItem {
        len,
        preferred,
        lo_factor,
        hi_factor,
        compliance,
        fixed: effective == EventKind::PlosiveBurst,
    }
}

const BURST_STAGE_C_COMPLIANCE: f64 = 0.05;

/// Runs stages A → B → C and returns the first feasible exact allocation.
pub fn allocate(items: &[AllocItem], n_out: u64) -> Result<Allocation, AllocError> {
    if let Some(i) = items.iter().position(|it| it.len == 0) {
        return Err(AllocError::EmptyItem(i));
    }
    if items.is_empty() {
        return if n_out == 0 {
            Ok(Allocation {
                durations: vec![],
                stage: Stage::A,
                lo: vec![],
                hi: vec![],
                relaxed_fixed: vec![],
                lambda: 0.0,
            })
        } else {
            Err(AllocError::Infeasible)
        };
    }
    if items.len() as u128 > n_out as u128 {
        return Err(AllocError::TooManySegments {
            segments: items.len(),
            n_out,
        });
    }
    for stage in [Stage::A, Stage::B, Stage::C] {
        let (lo, hi) = bounds(items, n_out, stage);
        let slo: u128 = lo.iter().map(|&v| v as u128).sum();
        let shi: u128 = hi.iter().map(|&v| v as u128).sum();
        if slo <= n_out as u128 && n_out as u128 <= shi {
            let (real, lambda) = solve_real(items, &lo, &hi, n_out, stage);
            let durations = integerize(&real, &lo, &hi, n_out);
            let relaxed_fixed = items
                .iter()
                .enumerate()
                .filter(|(i, it)| it.fixed && durations[*i] != it.len)
                .map(|(i, _)| i)
                .collect();
            let alloc = Allocation {
                durations,
                stage,
                lo,
                hi,
                relaxed_fixed,
                lambda,
            };
            check(&alloc, n_out).map_err(|_| AllocError::Infeasible)?;
            return Ok(alloc);
        }
    }
    Err(AllocError::Infeasible)
}

/// Verifies the hard allocator contract.
pub fn check(a: &Allocation, n_out: u64) -> Result<(), String> {
    let sum: u128 = a.durations.iter().map(|&d| d as u128).sum();
    if sum != n_out as u128 {
        return Err(format!("sum {sum} != n_out {n_out}"));
    }
    for (i, &d) in a.durations.iter().enumerate() {
        if d < a.lo[i] || d > a.hi[i] {
            return Err(format!("D[{i}]={d} outside [{}, {}]", a.lo[i], a.hi[i]));
        }
    }
    Ok(())
}

fn bounds(items: &[AllocItem], n_out: u64, stage: Stage) -> (Vec<u64>, Vec<u64>) {
    let mut lo = Vec::with_capacity(items.len());
    let mut hi = Vec::with_capacity(items.len());
    for it in items {
        let (l, h) = match (stage, it.fixed) {
            (Stage::A, true) | (Stage::B, true) => (it.len, it.len),
            (Stage::A, false) => {
                let lf = it.len as f64 * it.lo_factor;
                let hf = it.len as f64 * it.hi_factor;
                let l = ((lf - 1e-9).ceil().max(1.0)) as u64;
                let h = ((hf + 1e-9).floor() as u64).max(l);
                (l, h)
            }
            (Stage::B, false) | (Stage::C, _) => (1, n_out.max(1)),
        };
        lo.push(l);
        hi.push(h);
    }
    (lo, hi)
}

fn affordance(it: &AllocItem, stage: Stage) -> f64 {
    let c = if it.fixed && stage == Stage::C {
        BURST_STAGE_C_COMPLIANCE
    } else {
        it.compliance
    };
    // epsilon scaled by length keeps every non-fixed item strictly monotone in lambda.
    (it.len as f64 * c).max(1e-9 * it.len as f64)
}

fn eval(
    items: &[AllocItem],
    lo: &[u64],
    hi: &[u64],
    lambda: f64,
    stage: Stage,
    out: &mut [f64],
) -> f64 {
    let mut sum = 0.0;
    for (i, it) in items.iter().enumerate() {
        let p = it.len as f64 * it.preferred;
        let d = (p + lambda * affordance(it, stage)).clamp(lo[i] as f64, hi[i] as f64);
        out[i] = d;
        sum += d;
    }
    sum
}

/// Bisection on the single Lagrange multiplier.
fn solve_real(
    items: &[AllocItem],
    lo: &[u64],
    hi: &[u64],
    n_out: u64,
    stage: Stage,
) -> (Vec<f64>, f64) {
    let target = n_out as f64;
    let mut buf = vec![0.0; items.len()];
    let mut l = -1.0f64;
    let mut h = 1.0f64;
    let mut guard = 0;
    while eval(items, lo, hi, l, stage, &mut buf) > target && guard < 2100 {
        l *= 2.0;
        guard += 1;
    }
    guard = 0;
    while eval(items, lo, hi, h, stage, &mut buf) < target && guard < 2100 {
        h *= 2.0;
        guard += 1;
    }
    for _ in 0..400 {
        let m = 0.5 * (l + h);
        if m <= l || m >= h {
            break;
        }
        let s = eval(items, lo, hi, m, stage, &mut buf);
        if (s - target).abs() <= 1e-7 * target.max(1.0) {
            l = m;
            h = m;
            break;
        }
        if s < target {
            l = m;
        } else {
            h = m;
        }
    }
    let lambda = 0.5 * (l + h);
    eval(items, lo, hi, lambda, stage, &mut buf);
    (buf, lambda)
}

/// Floors real durations and repairs the integer sum deterministically.
fn integerize(real: &[f64], lo: &[u64], hi: &[u64], n_out: u64) -> Vec<u64> {
    let n = real.len();
    let mut d: Vec<u64> = (0..n)
        .map(|i| (real[i].floor().max(0.0) as u64).clamp(lo[i], hi[i]))
        .collect();
    let frac: Vec<f64> = (0..n).map(|i| real[i] - real[i].floor()).collect();
    let sum: u128 = d.iter().map(|&v| v as u128).sum();
    let target = n_out as u128;
    if sum < target {
        let mut rem = target - sum;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| frac[b].total_cmp(&frac[a]).then(a.cmp(&b)));
        // First pass: one frame each by largest remainder; later passes fill headroom.
        let mut first = true;
        while rem > 0 {
            let mut progressed = false;
            for &i in &order {
                if rem == 0 {
                    break;
                }
                let room = (hi[i] - d[i]) as u128;
                if room == 0 {
                    continue;
                }
                let add = if first { 1 } else { room.min(rem) };
                d[i] += add as u64;
                rem -= add;
                progressed = true;
            }
            first = false;
            if !progressed {
                break;
            }
        }
    } else if sum > target {
        let mut rem = sum - target;
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| frac[a].total_cmp(&frac[b]).then(a.cmp(&b)));
        let mut first = true;
        while rem > 0 {
            let mut progressed = false;
            for &i in &order {
                if rem == 0 {
                    break;
                }
                let room = (d[i] - lo[i]) as u128;
                if room == 0 {
                    continue;
                }
                let sub = if first { 1 } else { room.min(rem) };
                d[i] -= sub as u64;
                rem -= sub;
                progressed = true;
            }
            first = false;
            if !progressed {
                break;
            }
        }
    }
    d
}

#[cfg(test)]
mod tests {
    use super::*;

    fn items(spec: &[(u64, EventKind)], r: f64) -> Vec<AllocItem> {
        spec.iter().map(|&(l, k)| item_for(k, l, r, true)).collect()
    }

    #[test]
    fn simple_uniform_voiced() {
        let it = items(&[(48_000, EventKind::Voiced)], 1.5);
        let a = allocate(&it, 72_000).unwrap();
        assert_eq!(a.durations, vec![72_000]);
        assert_eq!(a.stage, Stage::A);
    }

    #[test]
    fn burst_fixed_others_pay() {
        use EventKind::*;
        let it = items(
            &[
                (4800, Silence),
                (960, PlosiveClosure),
                (240, PlosiveBurst),
                (960, Aspiration),
                (9600, Voiced),
            ],
            1.5,
        );
        let n_in: u64 = it.iter().map(|i| i.len).sum();
        let n_out = n_in * 3 / 2;
        let a = allocate(&it, n_out).unwrap();
        assert_eq!(a.stage, Stage::A);
        assert_eq!(a.durations[2], 240);
        assert!(a.durations[1] as f64 <= 960.0 * 1.15 + 1.0);
        assert!(
            a.durations[4] > 9600 * 3 / 2,
            "voiced pays back: {:?}",
            a.durations
        );
    }

    #[test]
    fn all_burst_infeasible_goes_to_c() {
        let it = items(
            &[
                (100, EventKind::PlosiveBurst),
                (100, EventKind::PlosiveBurst),
            ],
            2.0,
        );
        let a = allocate(&it, 400).unwrap();
        assert_eq!(a.stage, Stage::C);
        assert_eq!(a.durations.iter().sum::<u64>(), 400);
        assert_eq!(a.relaxed_fixed, vec![0, 1]);
    }

    #[test]
    fn stage_b_when_class_bounds_too_tight() {
        use EventKind::*;
        // Closures alone cannot reach 2x with their [0.85, 1.15] bounds.
        let it = items(
            &[
                (1000, PlosiveClosure),
                (100, PlosiveBurst),
                (1000, PlosiveClosure),
            ],
            2.0,
        );
        let a = allocate(&it, 4200).unwrap();
        assert_eq!(a.stage, Stage::B);
        assert_eq!(a.durations[1], 100);
        assert!(a.relaxed_fixed.is_empty());
    }

    #[test]
    fn too_many_segments() {
        let it = items(&[(1, EventKind::Voiced); 5], 0.5);
        assert!(matches!(
            allocate(&it, 3),
            Err(AllocError::TooManySegments { .. })
        ));
    }

    #[test]
    fn deterministic_ties() {
        let it = items(&[(3, EventKind::Voiced); 4], 0.5);
        let a = allocate(&it, 6).unwrap();
        let b = allocate(&it, 6).unwrap();
        assert_eq!(a, b);
        assert_eq!(a.durations.iter().sum::<u64>(), 6);
    }
}

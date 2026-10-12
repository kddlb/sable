//! Exhaustive/property tests for the exact-frame allocator (spec §5, §11.1).

use proptest::prelude::*;
use sable_core::budget::{AllocError, Stage, allocate, check, item_for};
use sable_core::ratio::StretchRatio;
use sable_core::segments::EventKind;

const KINDS: [EventKind; 9] = [
    EventKind::Voiced,
    EventKind::SibilantUnvoiced,
    EventKind::SibilantVoiced,
    EventKind::PlosiveClosure,
    EventKind::PlosiveBurst,
    EventKind::Aspiration,
    EventKind::Transition,
    EventKind::Silence,
    EventKind::Other,
];

fn kind_strategy() -> impl Strategy<Value = EventKind> {
    (0usize..KINDS.len()).prop_map(|i| KINDS[i])
}

fn ratio_strategy() -> impl Strategy<Value = (u64, u64)> {
    prop_oneof![
        (1u64..=40, 1u64..=40).prop_filter("in quality range", |(p, q)| 2 * p >= *q && *p <= 2 * q),
        Just((1, 4)),
        Just((4, 1)),
        Just((1, 2)),
        Just((2, 1)),
        (1u64..=1000, 1u64..=1000),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 2000, ..ProptestConfig::default() })]

    /// Every feasible partition yields an exact, in-bounds integer allocation.
    #[test]
    fn exact_sum_and_bounds(
        segs in prop::collection::vec((1u64..5000, kind_strategy()), 1..40),
        (p, q) in ratio_strategy(),
    ) {
        let ratio = StretchRatio::new(p, q).unwrap();
        let n_in: u64 = segs.iter().map(|s| s.0).sum();
        let n_out = ratio.target_frames(n_in).unwrap();
        let items: Vec<_> = segs.iter().map(|&(l, k)| item_for(k, l, ratio.as_f64(), true)).collect();
        match allocate(&items, n_out) {
            Ok(a) => {
                check(&a, n_out).unwrap();
                prop_assert_eq!(a.durations.iter().sum::<u64>(), n_out);
                prop_assert!(a.durations.iter().all(|&d| d >= 1));
                if a.stage < Stage::C {
                    for (i, it) in items.iter().enumerate() {
                        if it.fixed {
                            prop_assert_eq!(a.durations[i], it.len, "burst {} changed in stage {:?}", i, a.stage);
                        }
                    }
                    prop_assert!(a.relaxed_fixed.is_empty());
                }
                for &i in &a.relaxed_fixed {
                    prop_assert!(items[i].fixed && a.durations[i] != items[i].len);
                }
                // Determinism.
                prop_assert_eq!(allocate(&items, n_out).unwrap(), a);
            }
            Err(AllocError::TooManySegments { segments, n_out: n }) => {
                prop_assert!(segments as u64 > n);
            }
            Err(e) => prop_assert!(false, "unexpected {:?}", e),
        }
    }

    /// Many sub-20-ms segments at 8 kHz (< 160 frames each).
    #[test]
    fn many_tiny_segments(
        segs in prop::collection::vec((1u64..160, kind_strategy()), 50..400),
        (p, q) in ratio_strategy(),
    ) {
        let ratio = StretchRatio::new(p, q).unwrap();
        let n_in: u64 = segs.iter().map(|s| s.0).sum();
        let n_out = ratio.target_frames(n_in).unwrap();
        let items: Vec<_> = segs.iter().map(|&(l, k)| item_for(k, l, ratio.as_f64(), true)).collect();
        if let Ok(a) = allocate(&items, n_out) {
            check(&a, n_out).unwrap();
        } else {
            prop_assert!(items.len() as u64 > n_out);
        }
    }

    /// Stage A durations stay near preferences: no segment is pushed past its bounds and
    /// voiced (compliant) material absorbs the residual.
    #[test]
    fn stage_a_respects_class_bounds(
        segs in prop::collection::vec((200u64..5000, kind_strategy()), 1..20),
        (p, q) in (1u64..=20, 1u64..=20).prop_filter("range", |(p, q)| 2 * p >= *q && *p <= 2 * q),
    ) {
        let ratio = StretchRatio::new(p, q).unwrap();
        let n_in: u64 = segs.iter().map(|s| s.0).sum();
        let n_out = ratio.target_frames(n_in).unwrap();
        let items: Vec<_> = segs.iter().map(|&(l, k)| item_for(k, l, ratio.as_f64(), true)).collect();
        let a = allocate(&items, n_out).unwrap();
        if a.stage == Stage::A {
            for (i, it) in items.iter().enumerate() {
                let d = a.durations[i] as f64;
                prop_assert!(d >= (it.len as f64 * it.lo_factor - 1.0).max(1.0));
                prop_assert!(d <= it.len as f64 * it.hi_factor + 1.0);
            }
        }
    }
}

#[test]
fn all_burst_clip_double_length() {
    let items: Vec<_> = (0..10)
        .map(|_| item_for(EventKind::PlosiveBurst, 48, 2.0, true))
        .collect();
    let a = allocate(&items, 960).unwrap();
    assert_eq!(a.stage, Stage::C);
    assert_eq!(a.durations.iter().sum::<u64>(), 960);
    assert_eq!(a.relaxed_fixed.len(), 10);
}

#[test]
fn single_frame_segments_exact_count() {
    let items: Vec<_> = (0..7)
        .map(|i| item_for(KINDS[i % KINDS.len()], 1, 1.0, true))
        .collect();
    let a = allocate(&items, 7).unwrap();
    assert_eq!(a.durations, vec![1; 7]);
}

#[test]
fn huge_lengths_do_not_overflow() {
    let big = 1u64 << 40;
    let items = vec![
        item_for(EventKind::Voiced, big, 1.5, true),
        item_for(EventKind::PlosiveBurst, 100, 1.5, true),
        item_for(EventKind::Silence, big, 1.5, true),
    ];
    let ratio = StretchRatio::new(3, 2).unwrap();
    let n_out = ratio.target_frames(2 * big + 100).unwrap();
    let a = allocate(&items, n_out).unwrap();
    check(&a, n_out).unwrap();
    assert_eq!(a.durations[1], 100);
}

#[test]
fn unprotected_burst_is_not_fixed() {
    let it = item_for(EventKind::PlosiveBurst, 100, 2.0, false);
    assert!(!it.fixed);
}

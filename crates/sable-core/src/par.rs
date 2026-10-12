//! Minimal deterministic data parallelism on scoped `std` threads.
//!
//! Work is split into independent items whose results are reassembled in index order,
//! so the output never depends on the thread count or on scheduling.

use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Worker count for a `requested` setting: `0` means all available cores, `1` runs
/// inline on the calling thread.
pub fn threads(requested: usize) -> usize {
    match requested {
        0 => std::thread::available_parallelism().map_or(1, |n| n.get()),
        n => n,
    }
}

/// Splits `0..n` into at most `threads` contiguous ranges of at least `min_chunk`
/// items, runs `f` on each (in parallel when there is more than one) and returns the
/// results in range order.
pub fn chunks<T: Send>(
    n: usize,
    threads: usize,
    min_chunk: usize,
    f: impl Fn(Range<usize>) -> T + Sync,
) -> Vec<T> {
    let parts = threads.min(n / min_chunk.max(1)).max(1);
    let ranges: Vec<Range<usize>> = (0..parts)
        .map(|k| k * n / parts..(k + 1) * n / parts)
        .collect();
    if parts == 1 {
        return ranges.into_iter().map(f).collect();
    }
    std::thread::scope(|s| {
        let f = &f;
        let handles: Vec<_> = ranges.into_iter().map(|r| s.spawn(move || f(r))).collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("worker thread panicked"))
            .collect()
    })
}

/// Runs `f(i)` for every `i` in `0..n` on up to `threads` workers that take items
/// dynamically (for items of uneven cost), and returns the results in index order.
pub fn map<T: Send>(n: usize, threads: usize, f: impl Fn(usize) -> T + Sync) -> Vec<T> {
    let workers = threads.min(n).max(1);
    if workers == 1 {
        return (0..n).map(f).collect();
    }
    let next = AtomicUsize::new(0);
    let mut tagged: Vec<(usize, T)> = std::thread::scope(|s| {
        let (f, next) = (&f, &next);
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                s.spawn(move || {
                    let mut out = Vec::new();
                    loop {
                        let i = next.fetch_add(1, Ordering::Relaxed);
                        if i >= n {
                            break out;
                        }
                        out.push((i, f(i)));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|h| h.join().expect("worker thread panicked"))
            .collect()
    });
    tagged.sort_unstable_by_key(|&(i, _)| i);
    tagged.into_iter().map(|(_, t)| t).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_range_in_order() {
        for t in [1, 2, 3, 8] {
            let parts = chunks(103, t, 10, |r| r.collect::<Vec<_>>());
            assert!(parts.len() <= t.max(1));
            let flat: Vec<usize> = parts.into_iter().flatten().collect();
            assert_eq!(flat, (0..103).collect::<Vec<_>>());
        }
        assert_eq!(chunks(0, 4, 1, |r| r.len()), vec![0]);
    }

    #[test]
    fn map_is_ordered_for_any_thread_count() {
        for t in [1, 2, 5, 16] {
            assert_eq!(
                map(50, t, |i| i * i),
                (0..50).map(|i| i * i).collect::<Vec<_>>()
            );
        }
        assert!(map(0, 4, |i| i).is_empty());
    }
}

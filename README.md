# SABLE: Speech-Adaptive Budgeted Lengthening Engine

Pure-Rust, offline, pitch-preserving time stretching for speech. Different acoustic events
get different treatment, and everything is reconciled against **one sample-exact output
length**:

* **Plosives:** confident release bursts are copied 1:1 at their exact output onset. The
  preceding closure and following aspiration get nonlinear warps whose time-map slope is 1
  at the release.
* **Sibilants:** sustained /s/, /ʃ/, /z/ are re-synthesized from their time-varying
  smoothed spectrum with fresh, seeded noise instead of repeated noise grains. Voiced
  sibilants keep their harmonic low band.
* **Everything else:** variable-rate WSOLA driven by a shared monotonic time map.
* **Timing:** a constrained Lagrange allocator distributes the exact frame budget across
  event classes. Infeasible budgets relax in documented, reported stages.

Specification: [SABLE_SPEC.md](SABLE_SPEC.md). Measurements: [QUALITY.md](QUALITY.md).
Trade-offs: [docs/COMPROMISES.md](docs/COMPROMISES.md).

## Guarantees

For stretch factor `r = p/q = duration_out / duration_in` (so `r > 1` makes the audio longer):

* Output has exactly `N_out = floor(N_in · p / q + 1/2)` frames on every channel,
  computed in checked `u128` and never from floating-point seconds. This holds on every
  path, including the relaxation and fallback paths. Length is never fixed afterwards by
  padding or truncation.
* `r = 1` returns sample-identical audio, including signed zeros.
* All-zero input yields exact zeros at any factor.
* Same input, options and seed produce bit-identical output and report.
* Invalid input (NaN/Inf, unequal channel lengths, sample rate outside 8–96 kHz, more
  than 2 channels, zero ratio, zero rounded output length, size limits) returns a typed
  `StretchError` and never panics.
* Factors outside `[1/2, 2]` work but raise `ExtremeStretchFactor`, since they have no
  quality target.

## CLI

```sh
cargo build --release
# 1.5x longer
target/release/sable --input speech.wav --output stretched.wav --stretch 3/2
# faster, with diagnostics and intermediate layers
target/release/sable -i speech.wav -o faster.wav -s 0.75 --seed 42 \
    --diagnostics events.json --dump-dir layers/
```

| option | meaning |
|---|---|
| `--stretch` | `p/q` or decimal (`1.25`), parsed exactly |
| `--seed` | seed for sibilant noise (default fixed) |
| `--diagnostics FILE` | JSON report: frame counts, segment table with local ratios, detector scores, allocator stage, overlay decisions, WSOLA stats, peak/RMS before and after |
| `--features` | also include per-frame analysis features in the report |
| `--dump-dir DIR` | write `baseline`, `after_plosive`, `plosive_mask`, `sibilant_mask`, `sibilant_noise` WAVs (all exactly `N_out` frames) |
| `--quality fast\|standard\|high` | WSOLA grain/search presets |
| `--format float32\|pcm16\|pcm24` | output format. The default float32 neither clips nor quantizes. Integer formats round and clip to `[-1, 1)` with **no dither**. |
| `--no-plosive`, `--no-sibilant` | disable a special renderer |
| `--generic` | uniform generic WSOLA with all speech handling off (comparison baseline) |

The CLI prints source, target and actual frame counts, the achieved factor, the
allocator stage, and every warning. Stereo is never downmixed; more than 2 channels is an
error.

## Library

```rust
use sable_core::{stretch, AudioView, Options, StretchRatio};

let left: Vec<f32> = vec![0.0; 48_000];
let right = left.clone();
let chans = [left.as_slice(), right.as_slice()];
let input = AudioView { sample_rate: 48_000, channels: &chans };
let ratio = StretchRatio::from_fraction_str("3/2").unwrap(); // or from_decimal_str("1.5")
let result = stretch(&input, ratio, &Options::default()).unwrap();
assert_eq!(result.audio.channels[0].len(), 72_000);
println!("{:?}", result.report.warnings);
```

`Options::generic_wsola()` gives the comparison baseline. `analyze_events()` runs only
the analyzer and detector.

## Layout

```
crates/sable-core/src/
  ratio.rs      exact rational factors, checked target frames
  budget.rs     constrained exact-frame allocator (stages A/B/C)
  features.rs   shared mono analysis (RMS, flux, HF ratio, flatness, pitch, 1-ms envelope)
  detect.rs     heuristic event detector (EventDetector trait)
  segments.rs   event classes, partition validation, coalescing
  timemap.rs    monotonic time map with plosive-phase warps and stable inverse
  wsola.rs      variable-rate WSOLA
  plosives.rs   anchored dry-copy burst overlay
  sibilants.rs  spectral template + stochastic noise resynthesis + compositing
  stereo.rs     noise covariance model
  filters.rs    complementary zero-phase FIR split
  rng.rs        ChaCha8 Gaussian streams keyed by (seed, segment)
  pipeline.rs   two-pass orchestration
  report.rs     diagnostics
  metrics.rs    objective measurements used by tests and the quality report
  synth.rs      deterministic synthetic fixtures
crates/sable-core/tests/     allocator, safety, detector, duration, quality, plosive,
                             sibilant, determinism, real-speech integration tests
crates/sable-core/examples/  inspect_events, quality_report, ab_render
crates/sable-core/benches/   speech (realtime factor, no external harness)
crates/sable-cli/            `sable` binary + CLI tests
```

The spec suggests a top-level `tests/` directory. Integration tests live in
`crates/sable-core/tests/` instead, because a virtual workspace root cannot own test
targets.

## Development

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
cargo bench -p sable-core --bench speech
cargo run --release -p sable-core --example quality_report -- --write
cargo run --release -p sable-core --example inspect_events -- fixture:stop 48000
cargo run --release -p sable-core --example ab_render
```

Real-speech tests and examples read local recordings from `SABLE_SPEECH_WAV` (a
`;`-separated list of paths). They skip silently when no file is present. No recording is
committed: `*.wav`, `ab/` and `out/` are gitignored.

Dependencies: `realfft`/`rustfft`, `rand_chacha` (with `rand_core` 0.10), `hound`, `serde`,
`thiserror`, `clap`, `serde_json`, and `proptest` for tests only. All are pure Rust. The
crate uses `#![forbid(unsafe_code)]`.

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

**How it works: [docs/ALGORITHM.md](docs/ALGORITHM.md).** Specification: [SABLE_SPEC.md](SABLE_SPEC.md). Measurements: [QUALITY.md](QUALITY.md).
Trade-offs: [docs/COMPROMISES.md](docs/COMPROMISES.md).

## Guarantees

For stretch factor `r = p/q = duration_out / duration_in` (so `r > 1` makes the audio longer):

* Output has exactly `N_out = floor(N_in · p / q + 1/2)` frames on every channel,
  computed in checked `u128` and never from floating-point seconds. This holds on every
  path, including the relaxation and fallback paths. Length is never fixed afterwards by
  padding or truncation.
* `r = 1` returns sample-identical audio, including signed zeros.
* All-zero input yields exact zeros at any factor.
* Same input, options and seed produce bit-identical output and report, for any thread
  count.
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
# fit a disclaimer read into exactly 15 s of speech
target/release/sable -i disclaimer.wav -o fitted.wav --duration 15s     --skip-leading-silence --skip-trailing-silence
# faster, with diagnostics and intermediate layers
target/release/sable -i speech.wav -o faster.wav -s 0.75 --seed 42 \
    --diagnostics events.json --dump-dir layers/
```

| option | meaning |
|---|---|
| `--stretch` | `p/q` or decimal (`1.25`), parsed exactly |
| `--duration D` | instead of `--stretch`: hit an exact output duration, e.g. to fit a disclaimer read into a slot. `D` is seconds (`29.5`, `29.5s`), `1500ms`, clock time (`0:29.5`, `1:02.250`) or a frame count (`720000f`), parsed exactly and rounded to the nearest frame. The factor becomes `N_target / N_in`, so the output has exactly `N_target` frames. Applies to the input after any silence trimming. |
| `--max-duration D` | like `--duration`, but only shortens: an input already within `D` keeps its length (factor 1) |
| `--seed` | seed for sibilant noise (default fixed) |
| `--threads N` | worker threads for analysis, sibilant synthesis, formant correction and resampling (default 0 = all cores). The output is bit-identical for every value. See [ALGORITHM §11a](docs/ALGORITHM.md). |
| `--diagnostics FILE` | JSON report: frame counts, segment table with local ratios, detector scores, allocator stage, overlay decisions, WSOLA stats, peak/RMS before and after |
| `--features` | also include per-frame analysis features in the report |
| `--dump-dir DIR` | write `baseline`, `after_plosive`, `plosive_mask`, `sibilant_mask`, `sibilant_noise` WAVs (all exactly `N_out` frames) |
| `--quality fast\|standard\|high` | WSOLA grain/search presets |
| `--format float32\|pcm16\|pcm24` | output format. The default float32 neither clips nor quantizes. Integer formats round and clip to `[-1, 1)` with **no dither**. |
| `--no-plosive`, `--no-sibilant` | disable a special renderer |
| `--stereo-mode lr\|ms` | stereo strategy. `lr` (default) renders every event on both channels, with a stereo noise covariance model for sibilants. `ms` gives the mid the full SABLE treatment and the side WSOLA on the same time map and grain positions, ducked by `--side-duck-db` (default −6) inside the mid's resynthesized sibilants and burst cores. See [COMPROMISES §8](docs/COMPROMISES.md) before using `ms` on panned or spaced-microphone material. |
| `--skip-leading-silence` | drop silence before the first sound, keeping a 20 ms pre-roll (`--silence-threshold-db`, default −50 dBFS RMS on the loudest channel; `--silence-preroll-ms`). The exact output length then applies to the **trimmed** input, and the skipped frame count is printed and stored under `input` in the diagnostics JSON. |
| `--skip-trailing-silence` | the same after the last sound, keeping a 50 ms post-roll (`--silence-postroll-ms`). Combine both with `--duration` so the target applies to the speech, not the room tone around it. |
| `--generic` | uniform generic WSOLA with all speech handling off (comparison baseline) |
| `--pitch ST` | shift the pitch by `ST` semitones (−24…24, fractional allowed) without changing the duration set by `--stretch`. Use `--stretch 1` for a pure pitch shift. See [ALGORITHM §10a](docs/ALGORITHM.md). |
| `--formants preserve\|shift` | with `--pitch`: keep the spectral envelope (default, same voice) or move it with the pitch (tape-speed character) |
| `--progress` | print each processing phase with its time and a short summary to stderr |

### Input formats

WAV is read with `hound`. With the default `decode` feature, other formats are decoded in
pure Rust by `symphonia` 0.6. Output is always WAV.

| container / codec | status |
|---|---|
| WAV (PCM 8–32-bit, float 32/64, ADPCM) | ✔ (hound; symphonia fallback for encodings hound cannot read) |
| FLAC, ALAC (`.m4a`), AIFF, CAF, MKV/WebM (`.mka`) | ✔ lossless; verified bit-exact against the source WAV |
| MP3 (MP1/MP2), Ogg Vorbis, AAC-LC (`.m4a`/`.mp4`/`.aac`) | ✔ lossy, see below |
| Opus | ✘ no pure-Rust decoder in symphonia; returns an error |

Inputs with more than two channels are rejected rather than downmixed.
`cargo build -p sable-cli --no-default-features` gives a WAV-only build.

> **Lossy MP3/AAC and exact duration.** Encoders add delay and padding (often 1000–2000
> samples). symphonia removes MP3 padding only when the file records it (LAME/Xing
> header), and it does not remove AAC padding. The decoded input can therefore be longer than
> the original recording, and SABLE's exact frame count applies to the *decoded* signal.
> In a test, an AAC `.m4a` decoded 1,381 frames longer than its source. The MP3 (with a
> LAME header) and Ogg Vorbis decoded to the exact length. The CLI prints a note for
> MP3/AAC inputs, and the GUI shows one. Use WAV or FLAC when exact duration matters.

The CLI prints source, target and actual frame counts, the achieved factor, the
allocator stage, and every warning. Stereo is never downmixed; more than 2 channels is an
error.

## GUI (optional feature)

A desktop front end built with egui/eframe ships behind the `gui` feature, so the default
build stays headless and dependency-light:

```sh
cargo run --release -p sable-cli --features gui --bin sable-gui
# optional: open a file, and optionally process it immediately
cargo run --release -p sable-cli --features gui --bin sable-gui -- speech.wav 3/2
```

* Open any supported audio file (button or drag-and-drop). MP3/AAC files show a
  padding note. "Skip leading silence" and "Skip trailing silence" (with a threshold)
  trim the ends before processing, and the skipped regions are shaded in the original lane. The analyzer runs in the background, and event
  classes are shown as coloured bands over the waveform.
* Set the factor (`p/q` or decimal, with presets), or switch to **Target duration**
  and type the output length (`15`, `0:29.5`, `1500ms`, `720000f`) as **Exact** or
  **At most**; the panel shows the (trimmed) input length and the resulting factor. Also set the seed, quality, and which renderers are
  enabled, or pick generic WSOLA for comparison. Processing runs on a worker thread.
  With **Auto** on (the default), any settings change reprocesses after a 300 ms pause in
  editing. If the result is playing, playback continues in the new result at the same
  source position. A configuration that failed is not retried until it changes.
* Original and result are shown stacked, with views synchronized through the segment
  time map. Marks above the result show where bursts were dry-copied (red) and where
  sibilant noise was synthesized (orange). Use the wheel to zoom, drag to pan, click to
  set the cursor, and double-click for the full view.
* Play the original or the result through the default output device (cpal).
  Switching between them keeps the position, for A/B comparison. **Seek** by clicking a
  waveform (also while playing), by dragging the transport slider, with ←/→ (±1 s,
  Shift ±5 s), or with Home / ⏮ to go back to the start. Space plays or stops, and
  stopping leaves the cursor where playback was. If the device rate differs, playback alone is
  resampled; saved audio never is.
* Tabs show the summary (exact frame check, stage, levels, warnings), the segment table
  (click a row to zoom), renderer decisions with reasons, and burst candidates with
  their scores.
* Save the result (float32/pcm16/pcm24), the diagnostics JSON, or all intermediate
  layers.

On Linux the `gui` feature needs ALSA headers (`libasound2-dev`).

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
crates/sable-cli/            `sable` binary, shared WAV I/O (lib), CLI tests,
                             `sable-gui` (src/gui/, feature `gui`)
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
cargo run --release -p sable-core --example stop_eval -- --verbose   # needs stop_labels.txt
```

Real-speech tests and examples read local recordings from `SABLE_SPEECH_WAV` (a
`;`-separated list of paths) or, if that is unset, from a gitignored `speech_files.txt` at
the workspace root (one path per line, `#` comments). They skip silently when no file is
present. No recording is
committed: `*.wav`, `ab/` and `out/` are gitignored.

Dependencies: `realfft`/`rustfft`, `rand_chacha` (with `rand_core` 0.10), `hound`, `serde`,
`thiserror`, `clap`, `serde_json`, and `proptest` for tests only. The optional `gui` feature
adds `eframe`/`egui` 0.33 with `egui_extras` (SVG loader) and `egui-twemoji` for colour emoji,
`rfd` (file dialogs) and `cpal` (playback). The egui stack is pinned to 0.33 because
`egui-twemoji` 0.8 (the latest release) targets egui 0.33. All are pure Rust. The
crate uses `#![forbid(unsafe_code)]`.

## License

Licensed under either of [Apache License, Version 2.0](LICENSE-APACHE) or
[MIT license](LICENSE-MIT) at your option.

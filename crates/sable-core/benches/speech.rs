//! Throughput benchmark (no external harness): realtime factor = audio seconds / wall
//! seconds, on one thread and on all cores.
//!
//! ```text
//! cargo bench -p sable-core --bench speech
//! ```

use std::time::Instant;

use sable_core::synth::{self, StereoLayout};
use sable_core::{AudioBuffer, AudioView, Options, QualityPreset, StretchRatio, stretch};

fn tile(a: &AudioBuffer, seconds: f64) -> AudioBuffer {
    let n = (seconds * a.sample_rate as f64) as usize;
    AudioBuffer {
        sample_rate: a.sample_rate,
        channels: a
            .channels
            .iter()
            .map(|c| c.iter().cycle().take(n).copied().collect())
            .collect(),
    }
}

/// Best of 3 (after a warm-up) in seconds.
fn best_time(view: &AudioView<'_>, r: StretchRatio, opts: &Options) -> f64 {
    let _ = stretch(view, r, opts).unwrap();
    let mut best = f64::MAX;
    for _ in 0..3 {
        let t = Instant::now();
        let out = stretch(view, r, opts).unwrap();
        best = best.min(t.elapsed().as_secs_f64());
        std::hint::black_box(out);
    }
    best
}

/// Realtime factor on one thread and on all cores (the output is identical).
fn bench(label: &str, audio: &AudioBuffer, f: &str, opts: &Options) {
    let refs = audio.view_refs();
    let view = AudioView {
        sample_rate: audio.sample_rate,
        channels: &refs,
    };
    let r = StretchRatio::parse(f).unwrap();
    let secs = audio.frames() as f64 / audio.sample_rate as f64;
    let one = best_time(
        &view,
        r,
        &Options {
            threads: 1,
            ..opts.clone()
        },
    );
    let all = best_time(
        &view,
        r,
        &Options {
            threads: 0,
            ..opts.clone()
        },
    );
    println!(
        "{label:<34} r={f:<5} {:>6.2} s audio  1 thread {:>7.1} ms {:>6.1}x  all cores {:>7.1} ms {:>6.1}x realtime",
        secs,
        one * 1000.0,
        secs / one,
        all * 1000.0,
        secs / all
    );
}

fn main() {
    let utt16 = tile(&synth::utterance_fixture(16_000, 1).audio, 10.0);
    let utt48 = tile(&synth::utterance_fixture(48_000, 1).audio, 10.0);
    let st48 = tile(
        &synth::stereo_s_fixture(48_000, StereoLayout::PartiallyDecorrelated, 2).audio,
        10.0,
    );
    for f in ["0.75", "1.5"] {
        bench("utterance 16k mono (SABLE)", &utt16, f, &Options::default());
        bench("utterance 48k mono (SABLE)", &utt48, f, &Options::default());
        bench(
            "utterance 48k mono (generic WSOLA)",
            &utt48,
            f,
            &Options::generic_wsola(),
        );
        bench(
            "utterance 48k mono (SABLE fast)",
            &utt48,
            f,
            &Options {
                quality: QualityPreset::Fast,
                ..Options::default()
            },
        );
        bench("stereo /s/ 48k (SABLE)", &st48, f, &Options::default());
    }
}

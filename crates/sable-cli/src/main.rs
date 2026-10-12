//! `sable`: command-line front end for the SABLE speech time stretcher.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{ArgGroup, Parser, ValueEnum};
use sable_cli::trim::SilenceTrim;
use sable_cli::wav::{Format, write_wav};
use sable_cli::{load_input, may_have_encoder_padding, supported_extensions};
use sable_core::{
    AudioView, FormantMode, Options, ProgressEvent, QualityPreset, StereoMode, StretchRatio,
    TargetDuration, TargetMode, stretch_with_progress,
};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Stereo {
    /// Left/right: every renderer on both channels (default).
    Lr,
    /// Mid/side: mid gets SABLE, side gets WSOLA ducked inside mid events.
    Ms,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Formants {
    /// Keep the spectral envelope: same voice, new pitch.
    Preserve,
    /// Move formants with the pitch (tape-speed character).
    Shift,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Quality {
    Fast,
    Standard,
    High,
}

/// Speech time stretcher and pitch shifter with sample-exact output length.
#[derive(Parser, Debug)]
#[command(name = "sable", version)]
#[command(group(ArgGroup::new("length").required(true).args(["stretch", "duration", "max_duration"])))]
struct Cli {
    /// Input audio file, mono or stereo: WAV (integer/float PCM) and, with the default
    /// `decode` feature, FLAC, MP3, Ogg Vorbis, M4A/MP4 (AAC, ALAC), AIFF, CAF, MKV/WebM.
    #[arg(long, short)]
    input: PathBuf,
    /// Output WAV file.
    #[arg(long, short)]
    output: PathBuf,
    /// Stretch factor = duration_out / duration_in, as "p/q" or a decimal ("1.25").
    /// Use 1 to only shift the pitch.
    #[arg(long, short)]
    stretch: Option<String>,
    /// Exact output duration instead of a factor: seconds ("29.5", "29.5s"), "1500ms",
    /// clock time ("0:29.5", "1:02.250") or frames ("720000f"). Applies to the input
    /// after any silence trimming.
    #[arg(long, short)]
    duration: Option<String>,
    /// Like --duration, but only shortens: an input already within this duration is
    /// left at its own length (factor 1).
    #[arg(long)]
    max_duration: Option<String>,
    /// Pitch shift in semitones (-24..=24, fractional allowed). Duration is still set by
    /// --stretch alone.
    #[arg(long, default_value_t = 0.0, allow_negative_numbers = true)]
    pitch: f64,
    /// Experimental: for strong local slow-downs (>= 1.8x), blend the two aligned pitch
    /// periods around each grain's source position instead of repeating one.
    #[arg(long)]
    period_blend: bool,
    /// Formant handling for --pitch.
    #[arg(long, value_enum, default_value_t = Formants::Preserve)]
    formants: Formants,
    /// Print each processing phase with its timing to stderr as it finishes.
    #[arg(long)]
    progress: bool,
    /// Worker threads for the parallel stages (0 = all cores). The output is identical
    /// for every value.
    #[arg(long, default_value_t = 0)]
    threads: usize,
    /// Seed for deterministic noise resynthesis.
    #[arg(long, default_value_t = Options::default().seed)]
    seed: u64,
    /// Write the JSON diagnostic report to this path.
    #[arg(long)]
    diagnostics: Option<PathBuf>,
    /// Include per-frame analysis features in the diagnostic report.
    #[arg(long)]
    features: bool,
    #[arg(long, value_enum, default_value_t = Quality::Standard)]
    quality: Quality,
    /// Output sample format.
    #[arg(long, value_enum, default_value_t = Format::Float32)]
    format: Format,
    /// Disable plosive burst protection (generic timing for stops).
    #[arg(long)]
    no_plosive: bool,
    /// Disable sibilant noise resynthesis.
    #[arg(long)]
    no_sibilant: bool,
    /// Generic uniform WSOLA (all speech-specific handling off): comparison baseline.
    #[arg(long)]
    generic: bool,
    /// Write intermediate layers (baseline, overlays, masks) as WAVs into this directory.
    #[arg(long)]
    dump_dir: Option<PathBuf>,
    /// Stereo rendering strategy (ignored for mono input).
    #[arg(long, value_enum, default_value_t = Stereo::Lr)]
    stereo_mode: Stereo,
    /// Side gain in dB inside resynthesized sibilants and burst cores (--stereo-mode ms).
    #[arg(long, default_value_t = Options::default().side_duck_db, allow_negative_numbers = true)]
    side_duck_db: f64,
    /// Drop leading silence from the input before stretching (keeps a short pre-roll).
    /// The exact output length then applies to the trimmed input.
    #[arg(long)]
    skip_leading_silence: bool,
    /// Drop trailing silence from the input before stretching (keeps a short post-roll).
    /// The exact output length then applies to the trimmed input.
    #[arg(long)]
    skip_trailing_silence: bool,
    /// Level (dBFS RMS, loudest channel) treated as sound by --skip-*-silence.
    #[arg(long, default_value_t = SilenceTrim::default().threshold_db, allow_negative_numbers = true)]
    silence_threshold_db: f64,
    /// Audio kept before the first sound by --skip-leading-silence, in milliseconds.
    #[arg(long, default_value_t = SilenceTrim::default().preroll_ms)]
    silence_preroll_ms: f64,
    /// Audio kept after the last sound by --skip-trailing-silence, in milliseconds.
    #[arg(long, default_value_t = SilenceTrim::default().postroll_ms)]
    silence_postroll_ms: f64,
}

fn main() -> ExitCode {
    match run(Cli::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(cli: Cli) -> Result<(), String> {
    // Parse the length request before decoding, so typos fail fast.
    let target = match (&cli.duration, &cli.max_duration) {
        (Some(d), _) => Some((TargetDuration::parse(d), TargetMode::Exact)),
        (_, Some(d)) => Some((TargetDuration::parse(d), TargetMode::AtMost)),
        _ => None,
    }
    .map(|(d, m)| d.map(|d| (d, m)))
    .transpose()
    .map_err(|e| e.to_string())?;
    let fixed_ratio = cli
        .stretch
        .as_deref()
        .map(StretchRatio::parse)
        .transpose()
        .map_err(|e| e.to_string())?;
    let trimming = cli.skip_leading_silence || cli.skip_trailing_silence;
    let trim = trimming.then(|| SilenceTrim {
        leading: cli.skip_leading_silence,
        trailing: cli.skip_trailing_silence,
        threshold_db: cli.silence_threshold_db,
        preroll_ms: cli.silence_preroll_ms.max(0.0),
        postroll_ms: cli.silence_postroll_ms.max(0.0),
        ..SilenceTrim::default()
    });
    let (input, trimmed) = load_input(&cli.input, trim.as_ref()).map_err(|e| {
        format!(
            "{e}
(supported extensions: {})",
            supported_extensions().join(", ")
        )
    })?;
    if may_have_encoder_padding(&cli.input) {
        eprintln!(
            "note: lossy MP3/AAC input may include encoder padding (silence) that the decoder cannot remove; the exact frame count applies to the decoded signal"
        );
    }
    let mut opts = if cli.generic {
        Options::generic_wsola()
    } else {
        Options::default()
    };
    opts.seed = cli.seed;
    opts.threads = cli.threads;
    opts.period_blend = cli.period_blend;
    opts.diagnostics = cli.features;
    opts.quality = match cli.quality {
        Quality::Fast => QualityPreset::Fast,
        Quality::Standard => QualityPreset::Standard,
        Quality::High => QualityPreset::High,
    };
    opts.enable_plosive &= !cli.no_plosive;
    opts.enable_sibilant &= !cli.no_sibilant;
    opts.keep_layers = cli.dump_dir.is_some();
    opts.stereo_mode = match cli.stereo_mode {
        Stereo::Lr => StereoMode::LeftRight,
        Stereo::Ms => StereoMode::MidSide,
    };
    opts.side_duck_db = cli.side_duck_db.min(0.0);
    opts.pitch_semitones = cli.pitch;
    opts.formants = match cli.formants {
        Formants::Preserve => FormantMode::Preserve,
        Formants::Shift => FormantMode::Shift,
    };

    let ratio = match (fixed_ratio, target) {
        (Some(r), _) => r,
        (None, Some((d, mode))) => d
            .ratio(input.frames() as u64, input.sample_rate, mode)
            .map_err(|e| e.to_string())?,
        (None, None) => unreachable!("clap requires one of --stretch/--duration/--max-duration"),
    };

    let refs = input.view_refs();
    let view = AudioView {
        sample_rate: input.sample_rate,
        channels: &refs,
    };
    let mut phase_start = std::time::Instant::now();
    let mut on_progress = |e: ProgressEvent| match e {
        ProgressEvent::Started(_) => phase_start = std::time::Instant::now(),
        ProgressEvent::Finished { phase, detail } if cli.progress => eprintln!(
            "{:<22}{:>8.0} ms  {detail}",
            phase.label(),
            phase_start.elapsed().as_secs_f64() * 1000.0
        ),
        _ => {}
    };
    let result =
        stretch_with_progress(&view, ratio, &opts, &mut on_progress).map_err(|e| e.to_string())?;
    let r = &result.report;
    let actual = result.audio.frames() as u64;
    println!("ratio          {} ({:.6})", r.ratio, r.ratio_f64);
    println!(
        "sample rate    {} Hz, {} channel(s)",
        r.sample_rate, r.channels
    );
    let ms = |f: usize| f as f64 * 1000.0 / r.sample_rate as f64;
    if cli.skip_leading_silence {
        println!(
            "skipped        {} leading frames ({:.1} ms of silence)",
            trimmed.leading,
            ms(trimmed.leading)
        );
    }
    if cli.skip_trailing_silence {
        println!(
            "skipped        {} trailing frames ({:.1} ms of silence)",
            trimmed.trailing,
            ms(trimmed.trailing)
        );
    }
    if let Some((d, mode)) = &target {
        let secs = |f: u64| f as f64 / r.sample_rate as f64;
        let frames = match d {
            TargetDuration::Frames(_) => String::new(),
            TargetDuration::Seconds { .. } => format!(
                " = {} frames",
                d.frames(r.sample_rate).map_err(|e| e.to_string())?
            ),
        };
        println!(
            "target         {}{d}{frames}; input {:.3} s -> output {:.3} s",
            match mode {
                TargetMode::Exact => "",
                TargetMode::AtMost => "at most ",
            },
            secs(r.n_in),
            secs(r.n_out),
        );
    }
    if let Some(p) = &r.pitch {
        println!(
            "pitch          {:+} semitones (×{:.4}), formants {:?}, internal stretch {} frames",
            p.semitones, p.factor, p.formants, p.internal_frames
        );
    }
    if let Some(d) = r.side_duck_db {
        println!("stereo         mid/side (side duck {d:.1} dB inside mid events)");
    }
    println!("source frames  {}", r.n_in);
    println!("target frames  {}", r.n_out);
    println!("actual frames  {actual}");
    println!("achieved       {:.6}", r.achieved_factor);
    println!(
        "path           {:?}, allocator stage {:?}",
        r.path, r.allocator_stage
    );
    let bursts = r.plosive_overlays.iter().filter(|o| o.applied).count();
    let sibs = r.sibilant_overlays.iter().filter(|o| o.applied).count();
    println!(
        "segments       {} ({} protected bursts, {} resynthesized sibilants)",
        r.segments.len(),
        bursts,
        sibs
    );
    for w in &r.warnings {
        eprintln!("warning: {w:?}");
    }
    if actual != r.n_out {
        return Err(format!("internal length mismatch: {actual} != {}", r.n_out));
    }
    write_wav(&cli.output, &result.audio, cli.format)?;
    if let Some(p) = &cli.diagnostics {
        let mut value = serde_json::to_value(&result.report).map_err(|e| e.to_string())?;
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "input".into(),
                serde_json::json!({
                    "path": cli.input.display().to_string(),
                    "skip_leading_silence": cli.skip_leading_silence,
                    "skipped_leading_frames": trimmed.leading,
                    "skip_trailing_silence": cli.skip_trailing_silence,
                    "skipped_trailing_frames": trimmed.trailing,
                    "silence_threshold_db": cli.silence_threshold_db,
                    "silence_preroll_ms": cli.silence_preroll_ms,
                    "silence_postroll_ms": cli.silence_postroll_ms,
                    "target_duration": target.map(|(d, mode)| serde_json::json!({
                        "duration": d,
                        "mode": mode,
                        "frames": d.frames(r.sample_rate).ok(),
                    })),
                }),
            );
        }
        let json = serde_json::to_string_pretty(&value).map_err(|e| e.to_string())?;
        std::fs::write(p, json).map_err(|e| format!("{}: {e}", p.display()))?;
    }
    if let (Some(dir), Some(layers)) = (&cli.dump_dir, &result.layers) {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for (name, buf) in layers.named() {
            write_wav(&dir.join(format!("{name}.wav")), buf, Format::Float32)?;
        }
        println!("layers written to {}", dir.display());
    }
    Ok(())
}

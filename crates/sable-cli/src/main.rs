//! `sable`: command-line front end for the SABLE speech time stretcher.

#![forbid(unsafe_code)]

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use sable_cli::trim::SilenceTrim;
use sable_cli::wav::{Format, write_wav};
use sable_cli::{load_input, may_have_encoder_padding, supported_extensions};
use sable_core::{AudioView, Options, QualityPreset, StereoMode, StretchRatio, stretch};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Stereo {
    /// Left/right: every renderer on both channels (default).
    Lr,
    /// Mid/side: mid gets SABLE, side gets WSOLA ducked inside mid events.
    Ms,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Quality {
    Fast,
    Standard,
    High,
}

/// Pitch-preserving speech time stretcher with sample-exact output length.
#[derive(Parser, Debug)]
#[command(name = "sable", version)]
struct Cli {
    /// Input audio file, mono or stereo: WAV (integer/float PCM) and, with the default
    /// `decode` feature, FLAC, MP3, Ogg Vorbis, M4A/MP4 (AAC, ALAC), AIFF, CAF, MKV/WebM.
    #[arg(long, short)]
    input: PathBuf,
    /// Output WAV file.
    #[arg(long, short)]
    output: PathBuf,
    /// Stretch factor = duration_out / duration_in, as "p/q" or a decimal ("1.25").
    #[arg(long, short)]
    stretch: String,
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
    /// Level (dBFS RMS, loudest channel) treated as sound by --skip-leading-silence.
    #[arg(long, default_value_t = SilenceTrim::default().threshold_db, allow_negative_numbers = true)]
    silence_threshold_db: f64,
    /// Audio kept before the first sound by --skip-leading-silence, in milliseconds.
    #[arg(long, default_value_t = SilenceTrim::default().preroll_ms)]
    silence_preroll_ms: f64,
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
    let ratio = StretchRatio::parse(&cli.stretch).map_err(|e| e.to_string())?;
    let trim = cli.skip_leading_silence.then(|| SilenceTrim {
        threshold_db: cli.silence_threshold_db,
        preroll_ms: cli.silence_preroll_ms.max(0.0),
        ..SilenceTrim::default()
    });
    let (input, skipped) = load_input(&cli.input, trim.as_ref()).map_err(|e| {
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

    let refs = input.view_refs();
    let view = AudioView {
        sample_rate: input.sample_rate,
        channels: &refs,
    };
    let result = stretch(&view, ratio, &opts).map_err(|e| e.to_string())?;
    let r = &result.report;
    let actual = result.audio.frames() as u64;
    println!("ratio          {} ({:.6})", r.ratio, r.ratio_f64);
    println!(
        "sample rate    {} Hz, {} channel(s)",
        r.sample_rate, r.channels
    );
    if cli.skip_leading_silence {
        println!(
            "skipped        {skipped} leading frames ({:.1} ms of silence)",
            skipped as f64 * 1000.0 / r.sample_rate as f64
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
                    "skipped_leading_frames": skipped,
                    "silence_threshold_db": cli.silence_threshold_db,
                    "silence_preroll_ms": cli.silence_preroll_ms,
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

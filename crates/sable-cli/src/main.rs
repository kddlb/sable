//! `sable`: WAV front end for the SABLE speech time stretcher.

#![forbid(unsafe_code)]

use std::path::{Path, PathBuf};
use std::process::ExitCode;

use clap::{Parser, ValueEnum};
use sable_core::{AudioBuffer, AudioView, Options, QualityPreset, StretchRatio, stretch};

#[derive(Clone, Copy, Debug, ValueEnum)]
enum Format {
    /// 32-bit IEEE float (default; no clipping or quantization).
    Float32,
    /// 16-bit integer PCM (clips to [-1, 1), no dither).
    Pcm16,
    /// 24-bit integer PCM (clips to [-1, 1), no dither).
    Pcm24,
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
    /// Input WAV file (mono or stereo; integer or float PCM).
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
    let input = read_wav(&cli.input)?;
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
        let json = serde_json::to_string_pretty(&result.report).map_err(|e| e.to_string())?;
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

fn read_wav(path: &Path) -> Result<AudioBuffer, String> {
    let ctx = |e: hound::Error| format!("{}: {e}", path.display());
    let mut reader = hound::WavReader::open(path).map_err(ctx)?;
    let spec = reader.spec();
    let nch = spec.channels as usize;
    if nch == 0 || nch > 2 {
        return Err(format!(
            "{}: {nch} channels; only mono and stereo are supported (no automatic downmix)",
            path.display()
        ));
    }
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => {
            if spec.bits_per_sample != 32 {
                return Err(format!("{}: unsupported float bit depth", path.display()));
            }
            reader
                .samples::<f32>()
                .collect::<Result<_, _>>()
                .map_err(ctx)?
        }
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if !(1..=32).contains(&bits) {
                return Err(format!("{}: unsupported bit depth {bits}", path.display()));
            }
            let scale = 1.0 / (1u64 << (bits - 1)) as f64;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (v as f64 * scale) as f32))
                .collect::<Result<_, _>>()
                .map_err(ctx)?
        }
    };
    let frames = interleaved.len() / nch;
    let mut channels = vec![Vec::with_capacity(frames); nch];
    for f in interleaved.chunks_exact(nch) {
        for (c, &v) in f.iter().enumerate() {
            channels[c].push(v);
        }
    }
    Ok(AudioBuffer {
        sample_rate: spec.sample_rate,
        channels,
    })
}

fn write_wav(path: &Path, audio: &AudioBuffer, format: Format) -> Result<(), String> {
    let ctx = |e: hound::Error| format!("{}: {e}", path.display());
    let (bits, sample_format) = match format {
        Format::Float32 => (32, hound::SampleFormat::Float),
        Format::Pcm16 => (16, hound::SampleFormat::Int),
        Format::Pcm24 => (24, hound::SampleFormat::Int),
    };
    let spec = hound::WavSpec {
        channels: audio.channels.len() as u16,
        sample_rate: audio.sample_rate,
        bits_per_sample: bits,
        sample_format,
    };
    let mut w = hound::WavWriter::create(path, spec).map_err(ctx)?;
    for i in 0..audio.frames() {
        for c in &audio.channels {
            match format {
                Format::Float32 => w.write_sample(c[i]).map_err(ctx)?,
                Format::Pcm16 | Format::Pcm24 => {
                    let full = (1i64 << (bits - 1)) as f64;
                    let v = (c[i] as f64 * full).round().clamp(-full, full - 1.0) as i32;
                    w.write_sample(v).map_err(ctx)?
                }
            }
        }
    }
    w.finalize().map_err(ctx)
}

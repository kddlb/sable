//! End-to-end CLI checks on generated WAV files.

use std::path::{Path, PathBuf};
use std::process::Command;

fn tmp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("sable-cli-test-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write_wav(path: &Path, channels: u16, fs: u32, frames: usize, bits: u16) {
    let spec = hound::WavSpec {
        channels,
        sample_rate: fs,
        bits_per_sample: bits,
        sample_format: if bits == 32 {
            hound::SampleFormat::Float
        } else {
            hound::SampleFormat::Int
        },
    };
    let mut w = hound::WavWriter::create(path, spec).unwrap();
    for i in 0..frames {
        for c in 0..channels {
            let v = ((i as f64 * 0.031 * (c + 1) as f64).sin() * 0.3) as f32;
            if bits == 32 {
                w.write_sample(v).unwrap();
            } else {
                w.write_sample((v * 32767.0) as i32).unwrap();
            }
        }
    }
    w.finalize().unwrap();
}

fn sable(args: &[&str]) -> std::process::Output {
    Command::new(env!("CARGO_BIN_EXE_sable"))
        .args(args)
        .output()
        .unwrap()
}

#[test]
fn stretches_with_exact_frames_and_writes_diagnostics() {
    let d = tmp_dir("ok");
    let input = d.join("in.wav");
    let output = d.join("out.wav");
    let diag = d.join("events.json");
    let layers = d.join("layers");
    write_wav(&input, 2, 44_100, 44_100, 16);
    let out = sable(&[
        "--input",
        input.to_str().unwrap(),
        "--output",
        output.to_str().unwrap(),
        "--stretch",
        "3/2",
        "--seed",
        "42",
        "--diagnostics",
        diag.to_str().unwrap(),
        "--dump-dir",
        layers.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(stdout.contains("target frames  66150"), "{stdout}");
    assert!(stdout.contains("actual frames  66150"), "{stdout}");
    let r = hound::WavReader::open(&output).unwrap();
    assert_eq!(r.spec().channels, 2);
    assert_eq!(r.spec().sample_rate, 44_100);
    assert_eq!(r.spec().sample_format, hound::SampleFormat::Float);
    assert_eq!(r.duration(), 66_150);
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&diag).unwrap()).unwrap();
    assert_eq!(json["n_out"], 66_150);
    assert_eq!(json["seed"], 42);
    for name in [
        "baseline",
        "after_plosive",
        "plosive_mask",
        "sibilant_mask",
        "sibilant_noise",
    ] {
        assert_eq!(
            hound::WavReader::open(layers.join(format!("{name}.wav")))
                .unwrap()
                .duration(),
            66_150
        );
    }
}

#[test]
fn decimal_ratio_and_integer_output() {
    let d = tmp_dir("dec");
    let input = d.join("in.wav");
    let output = d.join("out.wav");
    write_wav(&input, 1, 16_000, 16_001, 32);
    let out = sable(&[
        "-i",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-s",
        "0.75",
        "--format",
        "pcm16",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let r = hound::WavReader::open(&output).unwrap();
    assert_eq!(r.spec().bits_per_sample, 16);
    // floor(16001 * 3 / 4 + 1/2) = 12001
    assert_eq!(r.duration(), 12_001);
}

#[test]
fn rejects_bad_ratio_and_multichannel() {
    let d = tmp_dir("bad");
    let input = d.join("in.wav");
    write_wav(&input, 1, 16_000, 1000, 16);
    let out = sable(&[
        "-i",
        input.to_str().unwrap(),
        "-o",
        d.join("o.wav").to_str().unwrap(),
        "-s",
        "0/3",
    ]);
    assert!(!out.status.success());
    let quad = d.join("quad.wav");
    write_wav(&quad, 4, 16_000, 1000, 16);
    let out = sable(&[
        "-i",
        quad.to_str().unwrap(),
        "-o",
        d.join("o.wav").to_str().unwrap(),
        "-s",
        "2",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("no automatic downmix"));
}

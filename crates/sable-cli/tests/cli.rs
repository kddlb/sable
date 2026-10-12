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

/// Hand-written 16-bit big-endian PCM AIFF (exercises the symphonia decode path).
#[cfg(feature = "decode")]
fn write_aiff(path: &Path, channels: u16, frames: u32, samples: impl Fn(u32, u16) -> i16) {
    let mut ssnd = Vec::new();
    ssnd.extend_from_slice(&0u32.to_be_bytes()); // offset
    ssnd.extend_from_slice(&0u32.to_be_bytes()); // block size
    for i in 0..frames {
        for c in 0..channels {
            ssnd.extend_from_slice(&samples(i, c).to_be_bytes());
        }
    }
    let mut comm = Vec::new();
    comm.extend_from_slice(&channels.to_be_bytes());
    comm.extend_from_slice(&frames.to_be_bytes());
    comm.extend_from_slice(&16u16.to_be_bytes());
    // 48000 Hz as an 80-bit IEEE extended float.
    comm.extend_from_slice(&[0x40, 0x0E, 0xBB, 0x80, 0, 0, 0, 0, 0, 0]);
    let mut body = b"AIFF".to_vec();
    for (id, data) in [(b"COMM", &comm), (b"SSND", &ssnd)] {
        body.extend_from_slice(id);
        body.extend_from_slice(&(data.len() as u32).to_be_bytes());
        body.extend_from_slice(data);
    }
    let mut file = b"FORM".to_vec();
    file.extend_from_slice(&(body.len() as u32).to_be_bytes());
    file.extend_from_slice(&body);
    std::fs::write(path, file).unwrap();
}

#[cfg(feature = "decode")]
#[test]
fn decodes_aiff_bit_exactly() {
    let d = tmp_dir("aiff");
    let input = d.join("in.aiff");
    let output = d.join("out.wav");
    let sample = |i: u32, c: u16| ((i as i32 * 37 + c as i32 * 1000) % 20000 - 10000) as i16;
    write_aiff(&input, 2, 4800, sample);
    let out = sable(&[
        "-i",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-s",
        "1",
        "--format",
        "pcm16",
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let mut r = hound::WavReader::open(&output).unwrap();
    assert_eq!(r.spec().channels, 2);
    assert_eq!(r.spec().sample_rate, 48_000);
    let got: Vec<i16> = r.samples::<i16>().map(Result::unwrap).collect();
    let want: Vec<i16> = (0..4800)
        .flat_map(|i| [sample(i, 0), sample(i, 1)])
        .collect();
    assert_eq!(got, want);
}

#[test]
fn unsupported_extension_is_a_clear_error() {
    let d = tmp_dir("ext");
    let input = d.join("in.xyz");
    std::fs::write(&input, b"not audio").unwrap();
    let out = sable(&[
        "-i",
        input.to_str().unwrap(),
        "-o",
        d.join("o.wav").to_str().unwrap(),
        "-s",
        "2",
    ]);
    assert!(!out.status.success());
    assert!(String::from_utf8_lossy(&out.stderr).contains("supported extensions"));
}

#[test]
fn skips_leading_silence_before_stretching() {
    let d = tmp_dir("skip");
    let input = d.join("in.wav");
    let output = d.join("out.wav");
    let diag = d.join("d.json");
    // 0.5 s of digital silence, then 0.5 s of tone at 16 kHz.
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 32,
        sample_format: hound::SampleFormat::Float,
    };
    let mut w = hound::WavWriter::create(&input, spec).unwrap();
    for i in 0..16_000 {
        let v = if i < 8_000 {
            0.0
        } else {
            ((i as f32) * 0.05).sin() * 0.3
        };
        w.write_sample(v).unwrap();
    }
    w.finalize().unwrap();
    let out = sable(&[
        "-i",
        input.to_str().unwrap(),
        "-o",
        output.to_str().unwrap(),
        "-s",
        "2",
        "--skip-leading-silence",
        "--diagnostics",
        diag.to_str().unwrap(),
    ]);
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    // Onset at 8000 minus 20 ms pre-roll (320 frames) -> 7680 skipped, 8320 remain -> 16640 out.
    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&diag).unwrap()).unwrap();
    assert_eq!(json["input"]["skipped_leading_frames"], 7_680);
    assert_eq!(json["n_in"], 8_320);
    assert_eq!(hound::WavReader::open(&output).unwrap().duration(), 16_640);
}

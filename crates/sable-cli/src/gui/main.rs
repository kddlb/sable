//! `sable-gui`: desktop front end for SABLE (build with `--features gui`).
//!
//! Open a WAV, inspect the detected acoustic events, stretch it with any exact ratio,
//! compare original and result (waveforms, event bands, renderer masks, playback),
//! and save audio, diagnostics and intermediate layers.

#![forbid(unsafe_code)]
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod emoji;
mod player;
mod wave;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, RichText};
use sable_cli::trim::{SilenceTrim, leading_silence_frames, skip_frames};
use sable_cli::wav::{Format, write_wav};
use sable_cli::{may_have_encoder_padding, read_audio, supported_extensions};
use sable_core::detect::BurstCandidate;
use sable_core::segments::{EventKind, Segment};
use sable_core::{
    AudioBuffer, AudioView, Options, ProcessResult, QualityPreset, StretchRatio, analyze_events,
    stretch,
};

use emoji::{emoji_button, emoji_label};
use player::{Player, Source};
use wave::{Band, Lane, kind_color};

fn main() -> eframe::Result {
    // Usage: sable-gui [file.wav [ratio]]  (a ratio processes immediately)
    let initial = std::env::args_os().nth(1).map(PathBuf::from);
    let auto_ratio = std::env::args().nth(2);
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("SABLE")
            .with_inner_size([1280.0, 820.0])
            .with_min_inner_size([900.0, 600.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "SABLE",
        options,
        Box::new(move |cc| Ok(Box::new(App::new(cc, initial, auto_ratio)))),
    )
}

/// Analyzer output sent back from the background thread.
type AnalysisMsg = Result<(Vec<Segment>, Vec<BurstCandidate>), String>;

/// Segment row: `(start, end, kind, confidence, reasons, id)` in input frames.
type SegRow = (u64, u64, EventKind, f64, Vec<String>, usize);

struct Loaded {
    path: PathBuf,
    audio: Arc<AudioBuffer>,
    segments: Option<Vec<Segment>>,
    bursts: Vec<BurstCandidate>,
}

#[derive(Clone, PartialEq)]
struct Settings {
    ratio_text: String,
    seed: u64,
    quality: QualityPreset,
    plosive: bool,
    sibilant: bool,
    generic: bool,
    skip_silence: bool,
    silence_db: f64,
}

impl Settings {
    fn options(&self) -> Options {
        let mut o = if self.generic {
            Options::generic_wsola()
        } else {
            Options::default()
        };
        o.seed = self.seed;
        o.quality = self.quality;
        o.enable_plosive &= self.plosive;
        o.enable_sibilant &= self.sibilant;
        o.keep_layers = true;
        o
    }
}

/// A running processing job.
struct Job {
    rx: Receiver<Result<Done, String>>,
    started: Instant,
    settings: Settings,
}

/// Delay after the last settings edit before auto-processing starts.
const AUTO_DEBOUNCE: Duration = Duration::from_millis(300);

struct Done {
    result: Arc<ProcessResult>,
    /// Result audio shared with the player (avoids copying on every play).
    audio: Arc<AudioBuffer>,
    settings: Settings,
    elapsed: Duration,
    /// Leading input frames dropped before stretching (skip-silence option).
    skipped: usize,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    Summary,
    Segments,
    Overlays,
    Bursts,
}

struct App {
    input: Option<Loaded>,
    settings: Settings,
    format: Format,
    analyze_rx: Option<Receiver<AnalysisMsg>>,
    process_rx: Option<Job>,
    done: Option<Done>,
    status: String,
    error: Option<String>,
    player: Result<Player, String>,
    /// Visible input frame range.
    view: (f64, f64),
    /// Cursor in input frames (play/inspect position).
    cursor: f64,
    selected: Option<usize>,
    tab: Tab,
    show_masks: bool,
    /// Re-run processing automatically when settings change.
    auto_process: bool,
    /// Settings seen last frame (change detection) and when they last changed.
    last_settings: Settings,
    changed_at: Option<Instant>,
    /// Settings whose processing failed (not retried automatically).
    failed_settings: Option<Settings>,
}

impl App {
    fn new(
        cc: &eframe::CreationContext<'_>,
        initial: Option<PathBuf>,
        auto_ratio: Option<String>,
    ) -> Self {
        egui_extras::install_image_loaders(&cc.egui_ctx);
        let mut app = Self {
            input: None,
            settings: Settings {
                ratio_text: "1.5".into(),
                seed: Options::default().seed,
                quality: QualityPreset::Standard,
                plosive: true,
                sibilant: true,
                generic: false,
                skip_silence: false,
                silence_db: SilenceTrim::default().threshold_db,
            },
            format: Format::Float32,
            analyze_rx: None,
            process_rx: None,
            done: None,
            status: "Open a mono or stereo audio file (WAV, FLAC, MP3, Ogg, M4A, AIFF, …) or drop one onto the window.".into(),
            error: None,
            player: Player::new(),
            view: (0.0, 1.0),
            cursor: 0.0,
            selected: None,
            tab: Tab::Summary,
            show_masks: true,
            auto_process: true,
            last_settings: Settings {
                ratio_text: String::new(),
                seed: 0,
                quality: QualityPreset::Standard,
                plosive: true,
                sibilant: true,
                generic: false,
                skip_silence: false,
                silence_db: 0.0,
            },
            changed_at: None,
            failed_settings: None,
        };
        app.last_settings = app.settings.clone();
        if let Some(p) = initial {
            app.open(&p, &cc.egui_ctx);
            if let Some(r) = auto_ratio {
                app.settings.ratio_text = r;
                app.process(&cc.egui_ctx);
            }
        }
        app
    }

    fn open(&mut self, path: &Path, ctx: &egui::Context) {
        match read_audio(path) {
            Ok(audio) => {
                if let Ok(p) = &self.player {
                    p.stop();
                }
                let audio = Arc::new(audio);
                self.view = (0.0, audio.frames().max(1) as f64);
                self.cursor = 0.0;
                self.selected = None;
                self.done = None;
                self.error = None;
                self.status = format!("Loaded {}", path.display());
                let (tx, rx) = channel();
                let a = audio.clone();
                let ctx2 = ctx.clone();
                std::thread::spawn(move || {
                    let refs = a.view_refs();
                    let view = AudioView {
                        sample_rate: a.sample_rate,
                        channels: &refs,
                    };
                    let r = analyze_events(&view)
                        .map(|(_, d)| (d.segments, d.burst_candidates))
                        .map_err(|e| e.to_string());
                    let _ = tx.send(r);
                    ctx2.request_repaint();
                });
                self.analyze_rx = Some(rx);
                self.input = Some(Loaded {
                    path: path.to_path_buf(),
                    audio,
                    segments: None,
                    bursts: Vec::new(),
                });
            }
            Err(e) => self.error = Some(e),
        }
    }

    fn ratio(&self) -> Result<StretchRatio, String> {
        StretchRatio::parse(self.settings.ratio_text.trim()).map_err(|e| e.to_string())
    }

    fn process(&mut self, ctx: &egui::Context) {
        let (Some(input), Ok(ratio)) = (&self.input, self.ratio()) else {
            return;
        };
        let full = input.audio.clone();
        let settings = self.settings.clone();
        let (tx, rx) = channel();
        let ctx2 = ctx.clone();
        std::thread::spawn(move || {
            let t = Instant::now();
            let skipped = if settings.skip_silence {
                let cfg = SilenceTrim {
                    threshold_db: settings.silence_db,
                    ..SilenceTrim::default()
                };
                leading_silence_frames(&full, &cfg)
            } else {
                0
            };
            let audio = if skipped > 0 {
                Arc::new(skip_frames(&full, skipped))
            } else {
                full
            };
            let refs = audio.view_refs();
            let view = AudioView {
                sample_rate: audio.sample_rate,
                channels: &refs,
            };
            let r = stretch(&view, ratio, &settings.options())
                .map(|result| Done {
                    audio: Arc::new(result.audio.clone()),
                    result: Arc::new(result),
                    settings,
                    elapsed: t.elapsed(),
                    skipped,
                })
                .map_err(|e| e.to_string());
            let _ = tx.send(r);
            ctx2.request_repaint();
        });
        self.process_rx = Some(Job {
            rx,
            started: Instant::now(),
            settings: self.settings.clone(),
        });
        self.status = "Processing…".into();
    }

    fn poll_jobs(&mut self) {
        if let Some(rx) = &self.analyze_rx
            && let Ok(r) = rx.try_recv()
        {
            self.analyze_rx = None;
            match (r, &mut self.input) {
                (Ok((segs, bursts)), Some(inp)) => {
                    inp.segments = Some(segs);
                    inp.bursts = bursts;
                }
                (Err(e), _) => self.error = Some(e),
                _ => {}
            }
        }
        if let Some(job) = &self.process_rx
            && let Ok(r) = job.rx.try_recv()
        {
            let job_settings = job.settings.clone();
            self.process_rx = None;
            match r {
                Ok(done) => {
                    // Keep playing the result across reprocessing: same source position.
                    let resume = match self.playing() {
                        Some((Source::Result, pos, _)) => Some(self.out_to_in(pos)),
                        _ => None,
                    };
                    let rep = &done.result.report;
                    self.status = format!(
                        "Done in {:.0} ms: {} frames in, {} frames out (exact), {} warning(s)",
                        done.elapsed.as_secs_f64() * 1000.0,
                        rep.n_in,
                        rep.actual_frames,
                        rep.warnings.len()
                    );
                    self.error = None;
                    self.failed_settings = None;
                    self.done = Some(done);
                    if let Some(x) = resume {
                        self.play_result(x);
                    }
                }
                Err(e) => {
                    self.error = Some(e);
                    self.failed_settings = Some(job_settings);
                    self.status = "Processing failed".into();
                }
            }
        }
    }

    fn playing(&self) -> Option<(Source, f64, f64)> {
        self.player.as_ref().ok().and_then(|p| p.status())
    }

    /// Current listening position in input frames: the playhead if playing, else the cursor.
    fn position_in(&self) -> f64 {
        match self.playing() {
            Some((Source::Original, p, _)) => p,
            Some((Source::Result, p, _)) => self.out_to_in(p),
            None => self.cursor,
        }
    }

    fn input_frames(&self) -> f64 {
        self.input.as_ref().map_or(0.0, |i| i.audio.frames() as f64)
    }

    fn play_original(&self, from_in: f64) {
        if let (Ok(p), Some(inp)) = (&self.player, &self.input) {
            p.play(
                inp.audio.clone(),
                Source::Original,
                from_in.max(0.0) as usize,
            );
        }
    }

    fn play_result(&self, from_in: f64) {
        if let (Ok(p), Some(d)) = (&self.player, &self.done) {
            p.play(
                d.audio.clone(),
                Source::Result,
                self.in_to_out(from_in).max(0.0) as usize,
            );
        }
    }

    /// Moves the cursor to `x_in` (input frames) and, if playing, seeks the playing
    /// buffer to the equivalent position.
    fn seek_in(&mut self, x_in: f64) {
        let x = x_in.clamp(0.0, (self.input_frames() - 1.0).max(0.0));
        self.cursor = x;
        if let (Ok(p), Some((src, _, _))) = (&self.player, self.playing()) {
            match src {
                Source::Original => p.seek(x),
                Source::Result => p.seek(self.in_to_out(x).max(0.0)),
            }
        }
    }

    /// Stops playback, leaving the cursor where playback was (so Space resumes there).
    fn stop_playback(&mut self) {
        if self.playing().is_some() {
            self.cursor = self.position_in();
        }
        if let Ok(p) = &self.player {
            p.stop();
        }
    }

    /// Piecewise-linear input→output frame map from the report's segment table.
    fn in_to_out(&self, x: f64) -> f64 {
        let Some(d) = &self.done else { return x };
        let x = x - d.skipped as f64;
        let segs = &d.result.report.segments;
        for s in segs {
            let (a, l) = (s.in_start as f64, s.in_len as f64);
            if x < a + l || std::ptr::eq(s, segs.last().unwrap()) {
                let v = ((x - a) / l).clamp(-1e9, 1e9);
                return s.out_start as f64 + v * s.out_len as f64;
            }
        }
        x
    }

    fn out_to_in(&self, y: f64) -> f64 {
        let Some(d) = &self.done else { return y };
        let segs = &d.result.report.segments;
        for s in segs {
            let (o, d_) = (s.out_start as f64, s.out_len as f64);
            if y < o + d_ || std::ptr::eq(s, segs.last().unwrap()) {
                let v = (y - o) / d_.max(1.0);
                return d.skipped as f64 + s.in_start as f64 + v * s.in_len as f64;
            }
        }
        y + d.skipped as f64
    }

    fn input_segments(&self) -> Option<Vec<SegRow>> {
        if let Some(d) = &self.done {
            return Some(
                d.result
                    .report
                    .segments
                    .iter()
                    .map(|s| {
                        let off = d.skipped as u64;
                        (
                            off + s.in_start,
                            off + s.in_start + s.in_len,
                            s.kind,
                            s.confidence,
                            s.reasons.clone(),
                            s.id,
                        )
                    })
                    .collect(),
            );
        }
        self.input.as_ref()?.segments.as_ref().map(|segs| {
            segs.iter()
                .map(|s| {
                    (
                        s.start,
                        s.end,
                        s.kind,
                        s.confidence,
                        s.reasons.clone(),
                        s.id,
                    )
                })
                .collect()
        })
    }

    fn zoom_to_segment(&mut self, id: usize) {
        if let Some(segs) = self.input_segments()
            && let Some(s) = segs.iter().find(|s| s.5 == id)
        {
            let n = self.input.as_ref().map_or(1, |i| i.audio.frames()) as f64;
            let len = (s.1 - s.0) as f64;
            let pad = (len * 2.0).max(0.15 * self.input.as_ref().unwrap().audio.sample_rate as f64);
            self.view = ((s.0 as f64 - pad).max(0.0), (s.1 as f64 + pad).min(n));
            self.cursor = s.0 as f64;
            self.selected = Some(id);
        }
    }

    // ---------------------------------------------------------------- panels

    fn top_bar(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        ui.horizontal(|ui| {
            if ui.add(emoji_button("📂", "Open audio…")).clicked()
                && let Some(p) = rfd::FileDialog::new()
                    .add_filter("Audio", supported_extensions())
                    .add_filter("WAV", &["wav", "wave"])
                    .pick_file()
            {
                self.open(&p, &ctx);
            }
            ui.separator();
            if let Some(inp) = &self.input {
                let a = &inp.audio;
                ui.label(
                    RichText::new(inp.path.file_name().unwrap_or_default().to_string_lossy())
                        .strong(),
                );
                ui.label(format!(
                    "{} Hz · {} · {} frames · {:.3} s",
                    a.sample_rate,
                    if a.channels.len() == 1 {
                        "mono"
                    } else {
                        "stereo"
                    },
                    a.frames(),
                    a.frames() as f64 / a.sample_rate as f64
                ));
                if may_have_encoder_padding(&inp.path) {
                    emoji_label(ui, "⚠️");
                    ui.label(RichText::new("lossy MP3/AAC: may include encoder padding").weak())
                        .on_hover_text(
                            "The decoder cannot always remove MP3/AAC encoder delay and padding,                              so the decoded length can exceed the original recording. The exact                              frame count applies to the decoded signal. Use WAV/FLAC when exact                              duration matters.",
                        );
                }
                if self.analyze_rx.is_some() {
                    ui.add(egui::Spinner::new());
                    ui.label("analyzing…");
                }
            } else {
                ui.label(RichText::new("no file loaded").weak());
            }
        });
    }

    fn side_panel(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        egui::ScrollArea::vertical().show(ui, |ui| {
            ui.horizontal(|ui| {
                emoji_label(ui, "⏱️");
                ui.heading("Stretch");
            });
            ui.label(
                RichText::new("factor = duration out / duration in")
                    .weak()
                    .small(),
            );
            let resp = ui.add(
                egui::TextEdit::singleline(&mut self.settings.ratio_text)
                    .hint_text("3/2 or 1.25")
                    .desired_width(f32::INFINITY),
            );
            ui.horizontal_wrapped(|ui| {
                for p in ["0.5", "0.65", "0.75", "0.9", "1.25", "1.5", "1.8", "2"] {
                    if ui
                        .selectable_label(self.settings.ratio_text == p, p)
                        .clicked()
                    {
                        self.settings.ratio_text = p.into();
                    }
                }
            });
            match self.ratio() {
                Ok(r) => {
                    let n_in = self.input.as_ref().map_or(0, |i| i.audio.frames() as u64);
                    let target = r
                        .target_frames(n_in)
                        .map_or("overflow".into(), |t| t.to_string());
                    ui.label(format!("= {r}  ({:.4}×), {target} frames", r.as_f64()));
                    if r.is_extreme() {
                        ui.colored_label(
                            Color32::from_rgb(230, 160, 40),
                            "outside the 0.5–2 quality range",
                        );
                    }
                }
                Err(e) => {
                    ui.colored_label(Color32::from_rgb(230, 80, 80), e);
                }
            }
            let enter = resp.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                emoji_label(ui, "🎛️");
                ui.heading("Options");
            });
            ui.checkbox(&mut self.settings.generic, "Generic WSOLA (baseline)")
                .on_hover_text("Uniform time map, no event handling: the comparison baseline.");
            ui.add_enabled_ui(!self.settings.generic, |ui| {
                ui.checkbox(&mut self.settings.plosive, "Protect plosive bursts");
                ui.checkbox(&mut self.settings.sibilant, "Resynthesize sibilants");
            });
            ui.checkbox(&mut self.settings.skip_silence, "Skip leading silence")
                .on_hover_text(
                    "Drop silence before the first sound (keeping 20 ms) before stretching.                      The exact output length then applies to the trimmed input.",
                );
            ui.add_enabled_ui(self.settings.skip_silence, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Threshold");
                    ui.add(
                        egui::DragValue::new(&mut self.settings.silence_db)
                            .range(-90.0..=-10.0)
                            .speed(0.5)
                            .suffix(" dBFS"),
                    );
                });
            });
            ui.horizontal(|ui| {
                ui.label("Quality");
                egui::ComboBox::from_id_salt("quality")
                    .selected_text(format!("{:?}", self.settings.quality))
                    .show_ui(ui, |ui| {
                        for q in [
                            QualityPreset::Fast,
                            QualityPreset::Standard,
                            QualityPreset::High,
                        ] {
                            ui.selectable_value(&mut self.settings.quality, q, format!("{q:?}"));
                        }
                    });
            });
            ui.horizontal(|ui| {
                ui.label("Seed");
                ui.add(egui::DragValue::new(&mut self.settings.seed));
            });

            ui.add_space(8.0);
            let busy = self.process_rx.is_some();
            let can = self.input.is_some() && self.ratio().is_ok() && !busy;
            ui.horizontal(|ui| {
                let btn = ui.add_enabled(
                    can,
                    emoji_button("⚙️", RichText::new("Process").strong())
                        .min_size(egui::vec2(120.0, 28.0)),
                );
                if btn.clicked() || (enter && can) {
                    self.process(&ctx);
                }
                ui.checkbox(&mut self.auto_process, "Auto")
                    .on_hover_text("Reprocess automatically when a setting changes.");
                if let Some(job) = &self.process_rx {
                    ui.add(egui::Spinner::new());
                    ui.label(format!("{:.1} s", job.started.elapsed().as_secs_f64()));
                }
            });
            if let Some(d) = &self.done
                && d.settings != self.settings
            {
                ui.colored_label(
                    Color32::from_rgb(230, 160, 40),
                    if self.auto_process && self.failed_settings.as_ref() != Some(&self.settings) {
                        "Settings changed: reprocessing…"
                    } else {
                        "Settings changed since this result."
                    },
                );
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                emoji_label(ui, "🎧");
                ui.heading("Listen");
            });
            match &self.player {
                Ok(p) => {
                    let playing = p.status();
                    let pos = self.position_in();
                    let mut action = None;
                    ui.horizontal(|ui| {
                        if self.input.is_some()
                            && ui
                                .add(emoji_button("▶️", "Original"))
                                .on_hover_text("Play the original from the current position")
                                .clicked()
                        {
                            action = Some(Source::Original);
                        }
                        if self.done.is_some()
                            && ui
                                .add(emoji_button("▶️", "Result"))
                                .on_hover_text("Play the result from the same source position")
                                .clicked()
                        {
                            action = Some(Source::Result);
                        }
                        if ui
                            .add_enabled(playing.is_some(), emoji_button("⏹️", "Stop"))
                            .clicked()
                        {
                            action = None;
                            p.stop();
                            self.cursor = pos;
                        }
                    });
                    match action {
                        Some(Source::Original) => self.play_original(pos),
                        Some(Source::Result) => self.play_result(pos),
                        None => {}
                    }
                    ui.label(
                        RichText::new(
                            "Original/Result switch keeps the position. Click a waveform to seek. \
                             Space play/stop, Left/Right 1 s (Shift 5 s), Home start.",
                        )
                        .weak()
                        .small(),
                    );
                    ui.label(
                        RichText::new(format!(
                            "{} · {} Hz · {} ch",
                            p.device_name, p.device_rate, p.device_channels
                        ))
                        .weak()
                        .small(),
                    );
                }
                Err(e) => {
                    ui.colored_label(
                        Color32::from_rgb(230, 80, 80),
                        format!("Playback unavailable: {e}"),
                    );
                }
            }

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                emoji_label(ui, "💾");
                ui.heading("Save");
            });
            ui.horizontal(|ui| {
                ui.label("Format");
                egui::ComboBox::from_id_salt("format")
                    .selected_text(self.format.label())
                    .show_ui(ui, |ui| {
                        for f in Format::ALL {
                            ui.selectable_value(&mut self.format, f, f.label());
                        }
                    });
            });
            if self.format != Format::Float32 {
                ui.label(
                    RichText::new("Integer output clips to [-1, 1) without dither.")
                        .weak()
                        .small(),
                );
            }
            ui.add_enabled_ui(self.done.is_some(), |ui| {
                if ui.add(emoji_button("💾", "Save result WAV…")).clicked() {
                    self.save_result();
                }
                if ui.add(emoji_button("🧾", "Save diagnostics JSON…")).clicked() {
                    self.save_json();
                }
                if ui.add(emoji_button("📁", "Save layers to folder…")).clicked() {
                    self.save_layers();
                }
            });

            ui.add_space(8.0);
            ui.horizontal(|ui| {
                emoji_label(ui, "🎨");
                ui.heading("Legend");
            });
            for k in wave::KINDS {
                if k == EventKind::Silence {
                    continue;
                }
                ui.horizontal(|ui| {
                    let (r, _) =
                        ui.allocate_exact_size(egui::vec2(14.0, 10.0), egui::Sense::hover());
                    ui.painter().rect_filled(r, 2.0, kind_color(k));
                    ui.label(format!("{k:?}"));
                });
            }
            ui.horizontal(|ui| {
                ui.checkbox(&mut self.show_masks, "Show renderer masks");
            });
            ui.label(
                RichText::new("marks above the result: red = dry-copied burst, orange = synthetic sibilant noise")
                    .weak()
                    .small(),
            );
        });
    }

    fn default_name(&self, suffix: &str, ext: &str) -> String {
        let stem = self
            .input
            .as_ref()
            .and_then(|i| i.path.file_stem())
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "output".into());
        let r = self
            .done
            .as_ref()
            .map(|d| d.result.report.ratio.to_string().replace('/', "-"))
            .unwrap_or_default();
        format!("{stem}_x{r}{suffix}.{ext}")
    }

    fn save_result(&mut self) {
        let Some(d) = &self.done else { return };
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("WAV audio", &["wav"])
            .set_file_name(self.default_name("", "wav"))
            .save_file()
        {
            match write_wav(&p, &d.result.audio, self.format) {
                Ok(()) => self.status = format!("Saved {}", p.display()),
                Err(e) => self.error = Some(e),
            }
        }
    }

    fn save_json(&mut self) {
        let Some(d) = &self.done else { return };
        if let Some(p) = rfd::FileDialog::new()
            .add_filter("JSON", &["json"])
            .set_file_name(self.default_name("_report", "json"))
            .save_file()
        {
            let r = serde_json::to_string_pretty(&d.result.report)
                .map_err(|e| e.to_string())
                .and_then(|s| std::fs::write(&p, s).map_err(|e| e.to_string()));
            match r {
                Ok(()) => self.status = format!("Saved {}", p.display()),
                Err(e) => self.error = Some(e),
            }
        }
    }

    fn save_layers(&mut self) {
        let Some(d) = &self.done else { return };
        let Some(layers) = &d.result.layers else {
            return;
        };
        if let Some(dir) = rfd::FileDialog::new().pick_folder() {
            let mut err = None;
            for (name, buf) in layers.named() {
                if let Err(e) = write_wav(&dir.join(format!("{name}.wav")), buf, Format::Float32) {
                    err = Some(e);
                }
            }
            match err {
                None => self.status = format!("Layers saved to {}", dir.display()),
                Some(e) => self.error = Some(e),
            }
        }
    }

    fn waveforms(&mut self, ui: &mut egui::Ui) {
        let Some(inp) = &self.input else {
            ui.centered_and_justified(|ui| {
                ui.label(
                    RichText::new("Drop an audio file here or use “Open audio…”")
                        .size(18.0)
                        .weak(),
                );
            });
            return;
        };
        let audio = inp.audio.clone();
        let n_in = audio.frames().max(1) as f64;
        let segs = self.input_segments().unwrap_or_default();
        let in_bands: Vec<Band> = segs
            .iter()
            .map(|s| Band {
                start: s.0 as f64,
                end: s.1 as f64,
                kind: s.2,
                selected: Some(s.5) == self.selected,
            })
            .collect();
        self.transport(ui);
        ui.add_space(4.0);
        let playing = self.playing();
        let has_result = self.done.is_some();
        let avail = ui.available_height();
        let lane_h = if has_result {
            (avail - 12.0) / 2.0
        } else {
            avail - 4.0
        };

        let in_resp = Lane {
            title: "Original",
            channels: &audio.channels,
            fs: audio.sample_rate,
            view: self.view,
            bands: &in_bands,
            masks: None,
            playhead: playing.filter(|p| p.0 == Source::Original).map(|p| p.1),
            cursor: Some(self.cursor),
            shade: self
                .done
                .as_ref()
                .filter(|d| d.skipped > 0)
                .map(|d| (0.0, d.skipped as f64)),
            dim: false,
        }
        .show(ui, lane_h);
        let mut lanes = vec![(in_resp, false)];

        if let Some(d) = &self.done {
            ui.add_space(6.0);
            let res = d.result.clone();
            let out_bands: Vec<Band> = res
                .report
                .segments
                .iter()
                .map(|s| Band {
                    start: s.out_start as f64,
                    end: (s.out_start + s.out_len) as f64,
                    kind: s.kind,
                    selected: Some(s.id) == self.selected,
                })
                .collect();
            let masks = res.layers.as_ref().filter(|_| self.show_masks).map(|l| {
                (
                    l.plosive_mask.channels[0].as_slice(),
                    l.sibilant_mask.channels[0].as_slice(),
                )
            });
            let title = format!(
                "Result ({})",
                if d.settings.generic {
                    "generic WSOLA"
                } else {
                    "SABLE"
                }
            );
            let out_resp = Lane {
                title: &title,
                channels: &res.audio.channels,
                fs: res.audio.sample_rate,
                view: (self.in_to_out(self.view.0), self.in_to_out(self.view.1)),
                bands: &out_bands,
                masks,
                playhead: playing.filter(|p| p.0 == Source::Result).map(|p| p.1),
                cursor: Some(self.in_to_out(self.cursor)),
                shade: None,
                dim: false,
            }
            .show(ui, lane_h);
            lanes.push((out_resp, true));
        }

        // Interaction: wheel zoom, drag pan, click cursor, double-click reset, hover info.
        for (lr, is_out) in lanes {
            let hover_in = lr
                .hover_frame
                .map(|f| if is_out { self.out_to_in(f) } else { f });
            let (v0, v1) = self.view;
            let span = v1 - v0;
            if let (Some(hf), Some(hin)) = (lr.hover_frame, hover_in) {
                let scroll = ui.input(|i| i.smooth_scroll_delta.y);
                if scroll != 0.0 {
                    let z = (-scroll as f64 / 300.0).exp();
                    let min_span = (audio.sample_rate as f64 * 0.005).min(n_in);
                    let new_span = (span * z).clamp(min_span, n_in);
                    let frac = (hin - v0) / span;
                    let mut a = hin - frac * new_span;
                    a = a.clamp(0.0, n_in - new_span);
                    self.view = (a, a + new_span);
                }
                let fs = audio.sample_rate as f64;
                let seg = segs
                    .iter()
                    .find(|s| (s.0 as f64) <= hin && hin < s.1 as f64);
                let mut text = format!(
                    "{:.3} s",
                    hf / if is_out {
                        self.done
                            .as_ref()
                            .map_or(fs, |d| d.result.audio.sample_rate as f64)
                    } else {
                        fs
                    }
                );
                if let Some(s) = seg {
                    text += &format!(
                        "\n{:?} #{}  conf {:.2}\n{:.1} ms in source",
                        s.2,
                        s.5,
                        s.3,
                        (s.1 - s.0) as f64 * 1000.0 / fs
                    );
                    if let Some(r) = s.4.first() {
                        text += &format!("\n{r}");
                    }
                }
                let _ = lr.response.clone().on_hover_text_at_pointer(text);
            }
            if lr.response.dragged() {
                let dx = lr.response.drag_delta().x as f64;
                let df = -dx / lr.rect.width() as f64 * span;
                let a = (v0 + df).clamp(0.0, n_in - span);
                self.view = (a, a + span);
            }
            if lr.response.clicked()
                && let Some(hin) = hover_in
            {
                let hin = hin.clamp(0.0, n_in);
                self.seek_in(hin);
                self.selected = segs
                    .iter()
                    .find(|s| (s.0 as f64) <= hin && hin < s.1 as f64)
                    .map(|s| s.5);
            }
            if lr.response.double_clicked() {
                self.view = (0.0, n_in);
            }
        }
    }

    /// Position readout and seek slider for the playing buffer (or the cursor).
    fn transport(&mut self, ui: &mut egui::Ui) {
        let fs_in = self.input.as_ref().map_or(48_000, |i| i.audio.sample_rate) as f64;
        let playing = self.playing();
        // The slider works in the coordinates of what is playing (result or original).
        let (label, mut pos, len, fs) = match playing {
            Some((Source::Result, p, l)) => (
                "Result",
                p,
                l,
                self.done
                    .as_ref()
                    .map_or(fs_in, |d| d.audio.sample_rate as f64),
            ),
            Some((Source::Original, p, l)) => ("Original", p, l, fs_in),
            None => ("Cursor", self.cursor, self.input_frames(), fs_in),
        };
        let len = len.max(1.0);
        ui.horizontal(|ui| {
            if ui
                .add(emoji_button("⏮️", ""))
                .on_hover_text("Back to start (Home)")
                .clicked()
            {
                self.seek_in(0.0);
            }
            let txt = format!("{label:<8} {:>7.3} / {:.3} s", pos / fs, len / fs);
            ui.add_sized(
                [200.0, 18.0],
                egui::Label::new(RichText::new(txt).monospace()),
            );
            ui.spacing_mut().slider_width = (ui.available_width() - 12.0).max(80.0);
            let resp = ui.add(
                egui::Slider::new(&mut pos, 0.0..=(len - 1.0).max(0.0))
                    .show_value(false)
                    .trailing_fill(true),
            );
            if resp.changed() {
                let x_in = match playing {
                    Some((Source::Result, _, _)) => self.out_to_in(pos),
                    _ => pos,
                };
                self.seek_in(x_in);
            }
        });
    }

    fn bottom_tabs(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.tab, Tab::Summary, "Summary");
            ui.selectable_value(&mut self.tab, Tab::Segments, "Segments");
            ui.selectable_value(&mut self.tab, Tab::Overlays, "Renderers");
            ui.selectable_value(&mut self.tab, Tab::Bursts, "Burst candidates");
            ui.separator();
            ui.label(
                RichText::new("wheel: zoom · drag: pan · click: seek · double-click: full view")
                    .weak()
                    .small(),
            );
        });
        ui.separator();
        match self.tab {
            Tab::Summary => self.summary(ui),
            Tab::Segments => self.segment_table(ui),
            Tab::Overlays => self.overlays(ui),
            Tab::Bursts => self.burst_table(ui),
        }
    }

    fn summary(&self, ui: &mut egui::Ui) {
        let Some(d) = &self.done else {
            ui.label(RichText::new("Process the file to see the report.").weak());
            return;
        };
        let r = &d.result.report;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("summary")
                    .num_columns(4)
                    .spacing([24.0, 4.0])
                    .show(ui, |ui| {
                        ui.label("ratio");
                        ui.label(
                            RichText::new(format!("{} ({:.6})", r.ratio, r.ratio_f64)).strong(),
                        );
                        ui.label("source frames");
                        ui.label(r.n_in.to_string());
                        ui.end_row();
                        ui.label("target frames");
                        ui.label(r.n_out.to_string());
                        ui.label("actual frames");
                        let ok = r.actual_frames == r.n_out;
                        ui.horizontal(|ui| {
                            emoji_label(ui, if ok { "✅" } else { "❌" });
                            ui.colored_label(
                                if ok {
                                    Color32::from_rgb(60, 180, 90)
                                } else {
                                    Color32::RED
                                },
                                format!(
                                    "{} {}",
                                    r.actual_frames,
                                    if ok { "exact" } else { "MISMATCH" }
                                ),
                            );
                        });
                        ui.end_row();
                        ui.label("achieved factor");
                        ui.label(format!("{:.6}", r.achieved_factor));
                        ui.label("path / allocator");
                        ui.label(format!("{:?} / {:?}", r.path, r.allocator_stage));
                        ui.end_row();
                        ui.label("bursts copied");
                        ui.label(format!(
                            "{} of {}",
                            r.plosive_overlays.iter().filter(|o| o.applied).count(),
                            r.plosive_overlays.len()
                        ));
                        ui.label("sibilants resynthesized");
                        ui.label(format!(
                            "{} of {}",
                            r.sibilant_overlays.iter().filter(|o| o.applied).count(),
                            r.sibilant_overlays.len()
                        ));
                        ui.end_row();
                        for l in &r.levels {
                            ui.label(format!("ch {} peak in/out", l.channel));
                            ui.label(format!("{:.3} / {:.3}", l.peak_in, l.peak_out));
                            ui.label(format!("ch {} RMS in/out", l.channel));
                            ui.label(format!(
                                "{:.1} / {:.1} dBFS",
                                20.0 * (l.rms_in.max(1e-9) as f64).log10(),
                                20.0 * (l.rms_out.max(1e-9) as f64).log10()
                            ));
                            ui.end_row();
                        }
                        if let Some(w) = &r.wsola {
                            ui.label("WSOLA grains");
                            ui.label(format!("{} ({} transient)", w.grains, w.transient_grains));
                            ui.label("uncovered filled");
                            ui.label(w.uncovered_filled.to_string());
                            ui.end_row();
                        }
                        ui.label("skipped silence");
                        ui.label(if d.skipped > 0 {
                            format!(
                                "{} frames ({:.1} ms)",
                                d.skipped,
                                d.skipped as f64 * 1000.0 / r.sample_rate as f64
                            )
                        } else {
                            "none".into()
                        });
                        ui.end_row();
                        ui.label("time");
                        ui.label(format!("{:.0} ms", d.elapsed.as_secs_f64() * 1000.0));
                        ui.end_row();
                    });
                ui.add_space(6.0);
                if r.warnings.is_empty() {
                    ui.label(RichText::new("No warnings.").weak());
                } else {
                    ui.label(RichText::new("Warnings").strong());
                    for w in &r.warnings {
                        ui.horizontal(|ui| {
                            emoji_label(ui, "⚠️");
                            ui.colored_label(Color32::from_rgb(230, 160, 40), format!("{w:?}"));
                        });
                    }
                }
            });
    }

    fn segment_table(&mut self, ui: &mut egui::Ui) {
        let Some(segs) = self.input_segments() else {
            ui.label(RichText::new("No analysis yet.").weak());
            return;
        };
        let fs = self.input.as_ref().map_or(48_000, |i| i.audio.sample_rate) as f64;
        let rep = self.done.as_ref().map(|d| d.result.clone());
        let row_h = 18.0;
        let mut clicked = None;
        let header = |ui: &mut egui::Ui| {
            ui.horizontal(|ui| {
                for (t, w) in [
                    ("#", 34.0),
                    ("kind", 120.0),
                    ("start", 70.0),
                    ("in ms", 60.0),
                    ("out ms", 60.0),
                    ("local ×", 60.0),
                    ("warp k", 52.0),
                    ("conf", 44.0),
                    ("reasons", 300.0),
                ] {
                    ui.add_sized([w, row_h], egui::Label::new(RichText::new(t).strong()));
                }
            });
        };
        header(ui);
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show_rows(ui, row_h, segs.len(), |ui, range| {
                for i in range {
                    let s = &segs[i];
                    let r = rep
                        .as_ref()
                        .and_then(|r| r.report.segments.iter().find(|x| x.id == s.5));
                    ui.horizontal(|ui| {
                        let sel = Some(s.5) == self.selected;
                        if ui
                            .add_sized(
                                [34.0, row_h],
                                egui::Button::selectable(sel, s.5.to_string()),
                            )
                            .clicked()
                        {
                            clicked = Some(s.5);
                        }
                        ui.add_sized(
                            [120.0, row_h],
                            egui::Label::new(
                                RichText::new(format!("{:?}", s.2)).color(kind_color(s.2)),
                            ),
                        );
                        ui.add_sized(
                            [70.0, row_h],
                            egui::Label::new(format!("{:.3} s", s.0 as f64 / fs)),
                        );
                        ui.add_sized(
                            [60.0, row_h],
                            egui::Label::new(format!("{:.1}", (s.1 - s.0) as f64 * 1000.0 / fs)),
                        );
                        let (out, local, k) = r.map_or(
                            ("–".to_string(), "–".to_string(), "–".to_string()),
                            |r| {
                                (
                                    format!("{:.1}", r.out_len as f64 * 1000.0 / fs),
                                    format!("{:.3}", r.local_ratio),
                                    if r.warp_k == 0.0 {
                                        "–".into()
                                    } else {
                                        format!("{:+.2}", r.warp_k)
                                    },
                                )
                            },
                        );
                        ui.add_sized([60.0, row_h], egui::Label::new(out));
                        ui.add_sized([60.0, row_h], egui::Label::new(local));
                        ui.add_sized([52.0, row_h], egui::Label::new(k));
                        ui.add_sized([44.0, row_h], egui::Label::new(format!("{:.2}", s.3)));
                        ui.add(
                            egui::Label::new(RichText::new(s.4.join("; ")).small().weak())
                                .truncate(),
                        );
                    });
                }
            });
        if let Some(id) = clicked {
            self.zoom_to_segment(id);
        }
    }

    fn overlays(&mut self, ui: &mut egui::Ui) {
        let Some(d) = &self.done else {
            ui.label(RichText::new("Process the file to see renderer decisions.").weak());
            return;
        };
        let res = d.result.clone();
        let mut clicked = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                for (title, list) in [
                    ("Plosive bursts", &res.report.plosive_overlays),
                    ("Sibilants", &res.report.sibilant_overlays),
                ] {
                    ui.label(RichText::new(title).strong());
                    if list.is_empty() {
                        ui.label(RichText::new("none detected").weak());
                    }
                    for o in list {
                        ui.horizontal(|ui| {
                            let (txt, col) = if o.applied {
                                ("applied", Color32::from_rgb(60, 180, 90))
                            } else {
                                ("kept WSOLA", Color32::from_rgb(230, 160, 40))
                            };
                            if ui.small_button(format!("#{}", o.segment_id)).clicked() {
                                clicked = Some(o.segment_id);
                            }
                            ui.colored_label(col, txt);
                            ui.label(RichText::new(&o.detail).small());
                        });
                    }
                    ui.add_space(6.0);
                }
            });
        if let Some(id) = clicked {
            self.zoom_to_segment(id);
        }
    }

    fn burst_table(&mut self, ui: &mut egui::Ui) {
        let Some(inp) = &self.input else { return };
        let fs = inp.audio.sample_rate as f64;
        let mut jump = None;
        egui::ScrollArea::vertical()
            .auto_shrink([false, false])
            .show(ui, |ui| {
                egui::Grid::new("bursts")
                    .striped(true)
                    .spacing([16.0, 2.0])
                    .show(ui, |ui| {
                        for h in [
                            "onset",
                            "jump dB",
                            "rise ms",
                            "HF frac",
                            "valley dB",
                            "confidence",
                            "",
                        ] {
                            ui.label(RichText::new(h).strong());
                        }
                        ui.end_row();
                        for b in &inp.bursts {
                            if ui.link(format!("{:.3} s", b.onset as f64 / fs)).clicked() {
                                jump = Some(b.onset as f64);
                            }
                            ui.label(format!("{:.1}", b.jump_db));
                            ui.label(format!("{:.1}", b.rise_ms));
                            ui.label(format!("{:.2}", b.hf_frac));
                            ui.label(b.valley_depth_db.map_or("–".into(), |v| format!("{v:.1}")));
                            ui.label(format!("{:.2}", b.confidence));
                            if b.accepted {
                                ui.colored_label(Color32::from_rgb(60, 180, 90), "protected");
                            } else {
                                ui.label(RichText::new("rejected").weak());
                            }
                            ui.end_row();
                        }
                    });
            });
        if let Some(on) = jump {
            let n = inp.audio.frames() as f64;
            let w = 0.12 * fs;
            self.view = ((on - w).max(0.0), (on + w).min(n));
            self.cursor = on;
        }
    }
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll_jobs();
        // Drag-and-drop.
        let dropped: Option<PathBuf> =
            ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone()));
        if let Some(p) = dropped {
            self.open(&p, ctx);
        }
        // Transport keys (ignored while typing in a text field).
        if !ctx.wants_keyboard_input() {
            // Consume the keys so a focused slider or button does not also react to them.
            let (space, left, right, home, shift) = ctx.input_mut(|i| {
                let shift = i.modifiers.shift;
                let m = if shift {
                    egui::Modifiers::SHIFT
                } else {
                    egui::Modifiers::NONE
                };
                (
                    i.consume_key(m, egui::Key::Space),
                    i.consume_key(m, egui::Key::ArrowLeft),
                    i.consume_key(m, egui::Key::ArrowRight),
                    i.consume_key(m, egui::Key::Home),
                    shift,
                )
            });
            if space {
                if self.playing().is_some() {
                    self.stop_playback();
                } else if self.done.is_some() {
                    self.play_result(self.cursor);
                } else {
                    self.play_original(self.cursor);
                }
            }
            let fs = self.input.as_ref().map_or(48_000, |i| i.audio.sample_rate) as f64;
            let step = if shift { 5.0 } else { 1.0 } * fs;
            if left {
                self.seek_in(self.position_in() - step);
            }
            if right {
                self.seek_in(self.position_in() + step);
            }
            if home {
                self.seek_in(0.0);
            }
        }
        // Auto-process: debounce settings edits; never retry a failed configuration.
        if self.settings != self.last_settings {
            self.last_settings = self.settings.clone();
            self.changed_at = Some(Instant::now());
        }
        if self.auto_process
            && self.process_rx.is_none()
            && self.analyze_rx.is_none()
            && self.input.is_some()
            && self.ratio().is_ok()
            && self
                .done
                .as_ref()
                .is_none_or(|d| d.settings != self.settings)
            && self.failed_settings.as_ref() != Some(&self.settings)
        {
            let wait = self.changed_at.map_or(Duration::ZERO, |t| {
                AUTO_DEBOUNCE.saturating_sub(t.elapsed())
            });
            if wait.is_zero() {
                self.changed_at = None;
                self.process(ctx);
            } else {
                ctx.request_repaint_after(wait);
            }
        }
        let busy =
            self.process_rx.is_some() || self.analyze_rx.is_some() || self.playing().is_some();
        if busy {
            ctx.request_repaint_after(Duration::from_millis(33));
        }

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.add_space(4.0);
            self.top_bar(ui);
            ui.add_space(2.0);
        });
        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                if let Some(e) = &self.error {
                    emoji_label(ui, "⚠️");
                    ui.colored_label(Color32::from_rgb(230, 80, 80), e);
                } else {
                    ui.label(&self.status);
                }
            });
        });
        egui::SidePanel::left("settings")
            .resizable(true)
            .default_width(250.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                self.side_panel(ui);
            });
        egui::TopBottomPanel::bottom("tabs")
            .resizable(true)
            .default_height(230.0)
            .min_height(120.0)
            .show(ctx, |ui| {
                ui.add_space(4.0);
                self.bottom_tabs(ui);
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.waveforms(ui);
        });
    }
}

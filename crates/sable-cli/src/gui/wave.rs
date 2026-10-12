//! Waveform lane: min/max envelope per pixel column, event-class bands, renderer masks,
//! time ticks and a playhead.

use eframe::egui::{self, Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, Vec2};
use sable_core::segments::EventKind;

/// Colour used for each event class (bands are drawn translucent).
pub fn kind_color(k: EventKind) -> Color32 {
    match k {
        EventKind::Voiced => Color32::from_rgb(70, 130, 220),
        EventKind::SibilantUnvoiced => Color32::from_rgb(240, 150, 40),
        EventKind::SibilantVoiced => Color32::from_rgb(210, 190, 40),
        EventKind::PlosiveClosure => Color32::from_rgb(130, 130, 150),
        EventKind::PlosiveBurst => Color32::from_rgb(230, 50, 50),
        EventKind::Aspiration => Color32::from_rgb(220, 110, 170),
        EventKind::Transition => Color32::from_rgb(150, 150, 150),
        EventKind::Silence => Color32::TRANSPARENT,
        EventKind::Other => Color32::from_rgb(110, 110, 110),
    }
}

pub const KINDS: [EventKind; 9] = [
    EventKind::Voiced,
    EventKind::SibilantUnvoiced,
    EventKind::SibilantVoiced,
    EventKind::PlosiveClosure,
    EventKind::PlosiveBurst,
    EventKind::Aspiration,
    EventKind::Transition,
    EventKind::Other,
    EventKind::Silence,
];

/// A band `[start, end)` in this lane's frames.
pub struct Band {
    pub start: f64,
    pub end: f64,
    pub kind: EventKind,
    pub selected: bool,
}

pub struct Lane<'a> {
    pub title: &'a str,
    pub channels: &'a [Vec<f32>],
    pub fs: u32,
    /// Visible frame range.
    pub view: (f64, f64),
    pub bands: &'a [Band],
    /// Optional `(plosive mask, sibilant mask)` drawn along the top of the lane.
    pub masks: Option<(&'a [f32], &'a [f32])>,
    pub playhead: Option<f64>,
    pub cursor: Option<f64>,
    pub dim: bool,
}

pub struct LaneResponse {
    pub response: egui::Response,
    /// Frame under the pointer, if hovering.
    pub hover_frame: Option<f64>,
    pub rect: Rect,
}

impl Lane<'_> {
    pub fn show(&self, ui: &mut egui::Ui, height: f32) -> LaneResponse {
        let (rect, response) = ui.allocate_exact_size(
            Vec2::new(ui.available_width(), height),
            Sense::click_and_drag(),
        );
        let painter = ui.painter_at(rect);
        let visuals = ui.visuals();
        painter.rect_filled(rect, 4.0, visuals.extreme_bg_color);
        let (v0, v1) = self.view;
        let span = (v1 - v0).max(1.0);
        let x_of = |f: f64| rect.left() + ((f - v0) / span) as f32 * rect.width();
        let f_of = |x: f32| v0 + ((x - rect.left()) / rect.width()) as f64 * span;

        // Event bands.
        for b in self.bands {
            if b.end < v0 || b.start > v1 || b.kind == EventKind::Silence {
                continue;
            }
            let (x0, x1) = (
                x_of(b.start).max(rect.left()),
                x_of(b.end).min(rect.right()),
            );
            let alpha = if b.selected { 110 } else { 45 };
            let c = kind_color(b.kind);
            let fill = Color32::from_rgba_unmultiplied(c.r(), c.g(), c.b(), alpha);
            painter.rect_filled(
                Rect::from_x_y_ranges(x0..=x1.max(x0 + 1.0), rect.y_range()),
                0.0,
                fill,
            );
            if b.selected {
                painter.rect_stroke(
                    Rect::from_x_y_ranges(x0..=x1.max(x0 + 1.0), rect.y_range()),
                    0.0,
                    Stroke::new(1.5_f32, c),
                    egui::StrokeKind::Inside,
                );
            }
        }

        // Waveform per channel lane.
        let nch = self.channels.len().max(1);
        let lane_h = rect.height() / nch as f32;
        let wave_color = if self.dim {
            visuals.weak_text_color()
        } else {
            visuals.strong_text_color()
        };
        let cols = rect.width().max(1.0) as usize;
        for (c, ch) in self.channels.iter().enumerate() {
            let mid = rect.top() + lane_h * (c as f32 + 0.5);
            let amp = lane_h * 0.45;
            if nch > 1 && c > 0 {
                painter.hline(
                    rect.x_range(),
                    rect.top() + lane_h * c as f32,
                    Stroke::new(1.0_f32, visuals.faint_bg_color),
                );
            }
            painter.hline(
                rect.x_range(),
                mid,
                Stroke::new(0.5_f32, visuals.weak_text_color().gamma_multiply(0.4)),
            );
            for col in 0..cols {
                let a = (v0 + span * col as f64 / cols as f64).floor().max(0.0) as usize;
                let b =
                    ((v0 + span * (col + 1) as f64 / cols as f64).ceil() as usize).min(ch.len());
                if a >= b {
                    continue;
                }
                let (mut lo, mut hi) = (f32::MAX, f32::MIN);
                for &v in &ch[a..b] {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                let x = rect.left() + col as f32 + 0.5;
                let y0 = mid - hi.clamp(-1.0, 1.0) * amp;
                let y1 = mid - lo.clamp(-1.0, 1.0) * amp;
                painter.line_segment(
                    [Pos2::new(x, y0), Pos2::new(x, y1.max(y0 + 1.0))],
                    Stroke::new(1.0_f32, wave_color),
                );
            }
        }

        // Renderer masks along the top edge: bars only where a renderer is active.
        if let Some((pm, sm)) = self.masks {
            for (row, (mask, color)) in [
                (pm, Color32::from_rgb(255, 60, 60)),
                (sm, Color32::from_rgb(255, 170, 40)),
            ]
            .into_iter()
            .enumerate()
            {
                let y_base = rect.top() + 4.0 + row as f32 * 8.0 + 6.0;
                for col in 0..cols {
                    let a = (v0 + span * col as f64 / cols as f64).max(0.0) as usize;
                    let b = ((v0 + span * (col + 1) as f64 / cols as f64) as usize).min(mask.len());
                    let m = if a < b {
                        mask[a..b].iter().fold(0f32, |m, &v| m.max(v))
                    } else {
                        0.0
                    };
                    if m > 0.0 {
                        let x = rect.left() + col as f32 + 0.5;
                        painter.line_segment(
                            [
                                Pos2::new(x, y_base),
                                Pos2::new(x, y_base - 6.0 * m.max(0.15)),
                            ],
                            Stroke::new(1.0_f32, color),
                        );
                    }
                }
            }
        }

        // Time ticks.
        let secs = span / self.fs as f64;
        let step = nice_step(secs / 8.0);
        let t0 = v0 / self.fs as f64;
        let mut t = (t0 / step).ceil() * step;
        let tick = visuals.weak_text_color();
        while t * (self.fs as f64) <= v1 {
            let x = x_of(t * self.fs as f64);
            painter.vline(
                x,
                (rect.bottom() - 5.0)..=rect.bottom(),
                Stroke::new(1.0_f32, tick),
            );
            painter.text(
                Pos2::new(x + 2.0, rect.bottom() - 2.0),
                Align2::LEFT_BOTTOM,
                format_time(t, step),
                FontId::proportional(10.0),
                tick,
            );
            t += step;
        }

        // Cursor and playhead.
        if let Some(c) = self.cursor
            && c >= v0
            && c <= v1
        {
            painter.vline(
                x_of(c),
                rect.y_range(),
                Stroke::new(1.0_f32, visuals.selection.bg_fill),
            );
        }
        if let Some(p) = self.playhead
            && p >= v0
            && p <= v1
        {
            painter.vline(
                x_of(p),
                rect.y_range(),
                Stroke::new(2.0_f32, Color32::from_rgb(40, 200, 120)),
            );
        }
        painter.text(
            rect.left_top() + Vec2::new(6.0, 22.0),
            Align2::LEFT_TOP,
            self.title,
            FontId::proportional(12.0),
            visuals.text_color(),
        );

        let hover_frame = response.hover_pos().map(|p| f_of(p.x));
        LaneResponse {
            response,
            hover_frame,
            rect,
        }
    }
}

fn nice_step(raw: f64) -> f64 {
    let raw = raw.max(1e-4);
    let p = 10f64.powf(raw.log10().floor());
    for m in [1.0, 2.0, 5.0, 10.0] {
        if raw <= m * p {
            return m * p;
        }
    }
    10.0 * p
}

pub fn format_time(t: f64, step: f64) -> String {
    if step >= 1.0 {
        format!("{t:.0} s")
    } else if step >= 0.1 {
        format!("{t:.1} s")
    } else if step >= 0.01 {
        format!("{t:.2} s")
    } else {
        format!("{:.1} ms", t * 1000.0)
    }
}

//! Colour emoji via Twemoji (egui's default fonts lack most emoji glyphs).
//!
//! Text labels use `egui_twemoji::EmojiLabel`; buttons get the same Twemoji SVG as an
//! image next to their text. Requires `egui_extras::install_image_loaders`.

use eframe::egui::{self, ImageSource, WidgetText};
use egui_twemoji::EmojiLabel;
use twemoji_assets::svg::SvgTwemojiAsset;

/// Every emoji the GUI shows (checked by a unit test so none silently falls back).
#[cfg(test)]
const USED: [&str; 14] = [
    "📂", "💾", "🧾", "📁", "🎧", "🎨", "✅", "❌", "⚠️", "⏱️", "🎛️", "▶️", "⏹️", "⚙️",
];

/// Looks an emoji up with and without the U+FE0F variation selector.
fn asset(emoji: &str) -> Option<(&'static SvgTwemojiAsset, String)> {
    let bare: String = emoji.chars().filter(|&c| c != '\u{FE0F}').collect();
    let with_vs = format!("{bare}\u{FE0F}");
    [emoji, bare.as_str(), with_vs.as_str()]
        .into_iter()
        .find_map(|e| SvgTwemojiAsset::from_emoji(e).map(|a| (a, e.to_string())))
}

/// Twemoji image for a single emoji.
pub fn emoji_image(emoji: &str, size: f32) -> Option<egui::Image<'static>> {
    let (svg, key) = asset(emoji)?;
    Some(
        egui::Image::new(ImageSource::Bytes {
            uri: format!("bytes://twemoji/{key}.svg").into(),
            bytes: egui::load::Bytes::Static(svg.as_bytes()),
        })
        .fit_to_exact_size(egui::vec2(size, size)),
    )
}

/// Button with a Twemoji icon followed by `text` (falls back to plain text).
pub fn emoji_button(emoji: &str, text: impl Into<WidgetText>) -> egui::Button<'static> {
    match emoji_image(emoji, 16.0) {
        Some(img) => egui::Button::image_and_text(img, text),
        None => egui::Button::new(text),
    }
}

/// A single Twemoji rendered inline, followed by a small gap. The emoji is normalized to
/// the form Twemoji knows so `EmojiLabel` recognizes it.
pub fn emoji_label(ui: &mut egui::Ui, emoji: &str) -> egui::Response {
    let key = asset(emoji).map_or_else(|| emoji.to_string(), |(_, k)| k);
    let spacing = ui.spacing().item_spacing.x;
    let r = EmojiLabel::new(key).show(ui);
    // EmojiLabel zeroes the item spacing of the surrounding layout; restore it.
    ui.spacing_mut().item_spacing.x = spacing;
    ui.add_space(4.0);
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_used_emoji_has_a_twemoji_asset() {
        for e in USED {
            assert!(asset(e).is_some(), "no Twemoji asset for {e:?}");
        }
    }
}

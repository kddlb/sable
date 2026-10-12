# Acoustic compromises and known limitations (v0.1)

SABLE's hard contract is the output length: every successful call returns exactly
`N_out = floor(N_in · p / q + 1/2)` frames per channel. When a local acoustic preference
conflicts with that contract, the preference loses. Every such case is **reported** in the
`AnalysisReport` (`warnings`, `allocator_stage`, overlay `detail` strings). None is
silently concealed. This document lists those trade-offs.

## 1. Burst protection vs. exact duration (allocator stages)

| Stage | When | What is given up | Reported as |
|---|---|---|---|
| A | normal | nothing; class bounds from spec §5.1 | `allocator_stage: A` |
| B | class bounds cannot reach `N_out` | non-burst segments may take any length in `[1, N_out]`; bursts stay 1× | `RelaxedNonBurstBudget` |
| C | even B is infeasible (burst-dominated clip) | burst protection waived: bursts become ordinary flexible segments (Stage C compliance 0.05) | `BurstProtectionWaived { segment_ids }` |
| coalesce | more segments than output frames | shortest non-burst segments merged into neighbours | `SegmentsCoalesced { from, to }` |
| D | coalescing would remove a burst anchor, or allocation fails | the whole clip uses one uniform time map and generic WSOLA | `UniformFallback { reason }` |

* A burst whose allocated duration differs from its input length (`D ≠ L`, stage C)
  **does not get the dry 1× overlay**. Copying it would misalign the release against
  the output time map, so it is rendered by WSOLA instead and listed in `plosive_overlays`
  with `applied: false` and the reason.
* On the synthetic fixtures (all rates, factors 0.5–2.0 and 1/4), stage C never occurred.
  Stage B occurred only at the out-of-range factor 1/4, whose 0.25× target is below the
  voiced class minimum of 0.40, and for the very short `edge_bursts` clip at 0.5.
  Stage C is exercised by the allocator property tests and `all_burst_clip_*` tests.
* Factors outside `[1/2, 2]` raise `ExtremeStretchFactor`. They are supported for
  length exactness only, with no quality target.

## 2. Plosive-phase warps

* The closure before a burst and the aspiration after it use the endpoint-preserving
  curve `F(v; k) = v + k v (1 − v)`, with `k` chosen so the time-map slope is exactly 1
  at the release boundary. `|k|` is clamped to 0.9 to keep the map strictly monotone.
  When the ideal `k` is outside that range, the slope at the boundary is not exactly 1,
  and `WarpSlopeRelaxed { k_ideal, k_used }` is emitted. In stage A this cannot happen
  (closure factors 0.85–1.15 give |k| ≤ 0.18; aspiration 0.65–1.65 gives |k| ≤ 0.54).
  It can occur after stage-B widening.
* Warps are applied only next to bursts that were actually anchored. Elsewhere the map is
  piecewise linear, so slope discontinuities at class boundaries remain. WSOLA's
  overlap-add smooths them, but no explicit derivative smoothing is done (spec §13 risk).

## 3. Plosive overlay wings

* The dry copy's context wings (4 ms before, 6 ms after) are clipped at file edges.
  When two anchored bursts are close, the wings are shortened to meet in the middle of
  the gap, so the two overlays never double-add. Short wings make a less gradual
  crossfade. The crossfade never passes through the protected core.
* A burst is copied bit-exactly; its surroundings are not. A dry 1× burst between
  stretched vowels can sound rhythmically unnatural at strong factors (spec §13). That
  needs listening, not metrics.

## 4. Sibilant resynthesis scope

* Only confident (`≥ 0.5`) sibilants with at least 40 ms of interior after two 10 ms
  guards (so 60 ms or more of source) are resynthesized. Shorter ones keep WSOLA. On the
  maintainer's first recording, two of nine detected sibilants (~55 ms each) were
  bypassed for this reason.
* Level calibration is a single **event-level** gain bounded to ±3 dB, applied on top of
  the time-varying spectral envelope. A per-frame tracking gain was deliberately avoided
  to prevent pumping. Measured calibrations on real speech were within ±1.4 dB.
* The voiced split is fixed at about 3 kHz (2–4 kHz transition, scaled down at low sample
  rates). A /z/ with strong frication below 2 kHz keeps that part from WSOLA.
* Frequency smoothing clips spikes above 4× the local median before averaging over
  1/6-octave neighbourhoods. That suppresses isolated harmonics in the noise profile, but
  strong harmonics in the high band of a voiced sibilant can still raise the noise
  spectrum slightly.
* Stereo noise uses the per-bin covariance model `Z_R = ρ Z1 + sqrt(1 − |ρ|²) Z2`. It
  reproduces the energy ratio and magnitude coherence well on fixtures (see
  QUALITY.md), but it is an approximation of the spatial image. DC and Nyquist bins are
  treated as real, so the phase of `ρ` is ignored there. If the coherence estimate is
  unstable across time (`> 0.4`), the event falls back to WSOLA and this is reported.
* Resynthesis never runs on events below the adaptive silence threshold, so silence and
  near-silence are never filled with fabricated hiss.

## 5. Detector conservatism

* The heuristic detector prefers `Transition`/`Other` (generic WSOLA) to a wrong label. On
  two real recordings it accepted only 1 of 24 and 1 of 12 burst candidates. Most real
  stops are therefore stretched by generic WSOLA, and A/B differences on real speech come
  mostly from sibilants and class-dependent timing, not from burst anchoring.
* Thresholds were tuned on synthetic fixtures and spot-checked on two recordings by one
  speaker. They are not validated across speakers, languages, or recording conditions.

## 6. Generic WSOLA limits

* WSOLA keeps pitch (F0 error ≤ 0.15 % on fixtures) but can sound phasey or warbly in
  strongly stretched vowels, and it smears transients that the detector did not protect.
* "Forward progression" is a preference, not a rule: a grain may step back by at most one
  synthesis hop. Expansion needs this to re-use a pitch period. A strict forward rule
  was tried and caused a 12 % F0 drop at 80 Hz.

## 7. Measurement caveats

* The repetition metric (max normalized autocorrelation of 2–60 ms lags) separates SABLE
  from generic WSOLA only weakly on synthetic /s/. Generic WSOLA on stationary noise does
  not repeat grains as obviously as expected. Treat it as a regression guard, not as
  evidence of perceptual improvement.
* No metric here establishes perceptual quality. Listening decisions belong in
  QUALITY.md and must be made by a human.

## 8. Stereo: L/R mode vs mid/side mode

* **Inter-channel delay (L/R mode).** Sibilant noise reproduces the measured stereo
  cross-spectrum. A delay between channels (spaced microphones) turns its phase into a
  ramp, and 1/6-octave averaging would cancel that ramp. The template therefore estimates
  the dominant lag per event (GCC-PHAT, ±2 ms), removes it before smoothing, and
  restores it afterwards. On a recording with a 0.73 ms offset this raised sibilant
  coherence from 0.65 to 0.85 (input 0.94). Several lags at once (e.g. strong room
  reflections) are only partly handled.
* **Mid/side mode (opt-in).** The mid gets every renderer. The side stays WSOLA on the
  same time map and the same grain positions: overlap-add is linear, so without special
  events M/S output equals L/R output to rounding (tested). Inside the mid's special
  events the side is ducked (default −6 dB, following the renderer masks). Measured
  consequences (QUALITY.md, "Stereo sibilance"):
  * **Off-centre or delayed sources lose their position during resynthesized
    sibilants.** Panning lives in the correlation between M and S, and fresh mid noise
    is uncorrelated with the side, so a 9 dB pan collapses to about 0 dB, and a spaced
    pair loses its 2 dB balance.
  * **The duck narrows the image during each /s/** (side-to-mid energy drops by about the
    duck depth). On a reverberant vocal this ducks the reverb under each /s/. Set
    `side_duck_db = 0` to keep the width.
  * For centred, reverberant material the coherence metrics of both modes are
    equivalent. Whether M/S sounds better (less WSOLA on the voice's own stereo
    sibilance, side artifacts masked by the duck) is a listening question. The A/B set
    includes `*_sable_ms.wav` renders for stereo inputs.

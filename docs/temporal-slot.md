# LS1 in place of a temporal upscaler: measured, not worth it

The idea: when a game uses DLSS (through D3DMetal, that's a MetalFX temporal upscale), its 3D frame reaches the
upscaler at render resolution and the HUD is drawn afterwards at native resolution. Running LS1 at that point instead
of MetalFX would upscale only the 3D image and leave the HUD sharp, for less GPU time than MetalFX takes there.

Measured November 2026 on the synthetic scene from nss-metal's quality notes (panning photo, moving spoked wheel,
static chain-link fence, 8×8-supersampled truth), 1488×928 → 2560×1600 (TLOU2's DLSS Balanced), M4 Max, frames
24–119 scored. Whole-frame PSNR; "flicker" is the mean frame-to-frame change on the static scene (truth: 0).

| | GPU | normal motion | 4× motion | static | flicker |
|---|---|---|---|---|---|
| MetalFX (temporal) | 0.50 ms* | 24.2 dB | 23.0 dB | 28.0 dB | 0.44/255 |
| NSS mid (temporal) | 1.36 ms | 25.1 dB | 22.8 dB | 31.4 dB | 0.34/255 |
| LS1 on the frame as given | 0.55 ms | 19.6 dB | 18.9 dB | 19.4 dB | 11.8/255 |
| LS1 after un-jittering, sharpness 1 | 0.62 ms | 23.3 dB | 22.5 dB | 23.0 dB | 5.3/255 |
| LS1, sharpness 3 (default) | 0.63 ms | 22.9 dB | 22.1 dB | 22.6 dB | 6.4/255 |
| FSR 1, after un-jittering | 0.56 ms | 23.1 dB | 22.3 dB | 22.7 dB | 5.2/255 |
| bicubic, after un-jittering | 0.38 ms | 23.3 dB | 22.5 dB | 23.0 dB | 5.0/255 |

\* MetalFX created at the exact render size. Games through D3DMetal create it at output size, where it costs
~1.2 ms (TLOU2); the spatial upscalers' cost doesn't depend on that.

Why it loses:

- **The frames are jittered.** With DLSS on, the game shifts its camera by a different sub-pixel offset every frame
  for the temporal upscaler to accumulate. A spatial upscaler shows those shifts as the whole image wobbling: 11.8/255
  of change per frame on a still scene. Resampling each frame back by its jitter (one bilinear pass, "un-jittering")
  brings it to ~5/255, still ~12× MetalFX's, and blurs.
- **The frames aren't anti-aliased.** With DLSS on, the game's own TAA is off: DLSS is the anti-aliasing. A spatial
  upscaler enlarges every stair-step and the shimmer on thin lines. On a still image MetalFX scores 5 dB higher and
  NSS 8 dB higher.
- **LS1's network doesn't help here.** It scores the same as plain bicubic, which costs 0.38 ms. It was trained on
  finished, anti-aliased frames, which is what it gets in Lossless Scaling's (and Lightscale's "Scale to window")
  use.
- Converting the linear HDR input to sRGB for LS1 and back made no difference (±0.1 dB) and costs 0.08 ms.

So in the DLSS slot LS1 would save ~0.6 ms against the game's MetalFX, but the image would be aliased and shimmer
noticeably. NSS fast (~0.8 ms) is the cheaper-than-MetalFX option there. LS1 remains the right tool for scaling a
finished frame to the window ("Scale to window"), where the game's own anti-aliasing has already run.

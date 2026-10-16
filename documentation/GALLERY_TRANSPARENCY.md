# Gallery image transparency

`gallery/emnlp_density_transparent.png` is a transparent derivative of
`gallery/emnlp_density.png`, created on 2026-10-07. The original and the published
copy under `docs/public/gallery/` are unchanged. No page references were changed.

## Conversion

The conversion operates directly on the original pixels without redrawing,
resizing or cropping. For each opaque source pixel with byte-valued RGB channels:

- Set `alpha = max(R, G, B)`.
- For nonzero alpha, set each output channel to `round(255 * channel / alpha)`.
- Set pure-black pixels to transparent black `(0, 0, 0, 0)`.

This is a black-to-alpha decomposition in the stored RGB channel space. It
retains faint contours with partial transparency and removes black from their
blended edges. The original foreground opacity cannot be uniquely recovered
from a flattened screenshot; this chooses the lowest alpha that keeps every
output channel within its valid range. Colours and labels consequently appear
paler on light backgrounds. This asset is not a redesigned light-theme plot.

## Verification

Checks were defined before conversion and run against the saved PNG:

- RGBA output with the original 3024 × 1890 dimensions.
- All 4,515,758 originally black pixels are fully transparent.
- All 1,199,602 nonblack pixels retain nonzero alpha; partial alpha is present.
- Compositing over black reproduces every original RGB byte exactly: maximum
  channel error 0, mean channel error 0.
- The source SHA-256 remains
  `a99663aa01d00a651e7f675e8504adcefaf9db0eb501ce12d09392bb66351d1e`.
- A light-background preview was visually inspected for labels and contours.

The one-off conversion and verification used `uv run --no-sync python` with
Pillow and NumPy. No application code or runtime dependencies changed.

/**
 * Pure formatting helpers for the hover tooltip (FrostedTooltip).
 *
 * Keeps the presentational component thin: number formatting and hex-color
 * detection/normalization live here so they can be unit-tested in isolation.
 */

/**
 * Matches a 3-, 6-, or 8-digit hex color. A leading '#' is required so that
 * plain numeric/hex-digit metadata (ids like `123456`, `deadbeef`) is not
 * mistaken for a color and given a spurious swatch. Color fields in this app
 * (`colour_code` etc.) always carry the '#'.
 */
const HEX_COLOR_RE = /^#(?:[0-9a-fA-F]{3}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$/;

/** True when a value is a string that looks like a CSS hex color. */
export function isHexColor(value: unknown): value is string {
  return typeof value === 'string' && HEX_COLOR_RE.test(value.trim());
}

/** Normalize a hex color to its trimmed '#'-prefixed form for use as a CSS color. */
export function normalizeHex(value: string): string {
  return value.trim();
}

/**
 * Format a metadata value for display.
 *
 * Real numbers are rendered with locale-aware grouping for large magnitudes
 * (so counts read as `12,345`) while small integers such as years are left
 * ungrouped (`2019`, not `2,019`); floats are rounded to at most 3 fraction
 * digits (`0.42184…` → `0.422`). Everything else is stringified and capped in
 * length. Numeric *strings* are deliberately left untouched to avoid mangling
 * ids, zip codes, or codes that only happen to be digits.
 */
/**
 * Integers with a magnitude below this stay ungrouped so that years (`2019`)
 * and small codes read naturally; only larger counts get thousands separators.
 * Number-typed ids at or above this threshold are out of scope and will group.
 */
const GROUPING_THRESHOLD = 10000;

export function formatMetadataValue(value: unknown, maxLen = 200): string {
  if (typeof value === 'number' && Number.isFinite(value)) {
    if (Number.isInteger(value)) {
      return Math.abs(value) >= GROUPING_THRESHOLD ? value.toLocaleString() : String(value);
    }
    return new Intl.NumberFormat(undefined, { maximumFractionDigits: 3 }).format(value);
  }
  const str = String(value ?? '');
  return str.length > maxLen ? str.substring(0, maxLen) + '…' : str;
}

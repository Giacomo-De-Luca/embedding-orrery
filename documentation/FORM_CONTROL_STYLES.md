# Form control styles: dot controls and hairline sliders

Decided 2026-10-23 from a side-by-side proposal page for the Explore Controls panel. The user chose the
dot-in-circle look over tick marks, the hairline slider, and no change to the buttons (they stay flat;
an accent-coloured, glowing Apply button was proposed and rejected). The chrome stays monochrome: every
"on" state uses `primary`, there is no brand accent token.

## Dot control (`lib/ui-primitives/dot-control.tsx`)

`Checkbox` and `RadioGroupItem` draw with one class list, `dotControlClassName`, and one
`DotIndicator`: a 16px ring (`border-input`, `dark:bg-input/30`) whose 8px dot fills with the
control's text colour (`text-primary` by default) when it is on. Checked controls also take a
`primary` border; hover lifts the border to `foreground/40`; focus shows the usual `ring` halo.

- Semantics are unchanged: a checkbox still has `role="checkbox"` and a radio `role="radio"`.
  The visual is identical by design, which is the trade-off the user accepted (a dot usually
  means "pick one"; here it means "on" as well).
- Recolour the dot through the control's text colour: `<Checkbox className="text-emerald-500" />`.
- A new on/off control must reuse `dotControlClassName`; do not reintroduce a tick icon.
- The topic list's selection ring in `TopicSearchSection.tsx` (an 8px ring that fills emerald) is a
  separate, older idiom and was left alone.

## Hairline slider (`lib/ui-primitives/slider.tsx`)

| Part | Spec |
|---|---|
| Root | 24px tall pointer target (`h-6`; `w-6` when vertical) around a 4px track |
| Track | `h-1`, `bg-foreground/12` (visible in both themes; `bg-muted` all but vanished in light mode), `@container` |
| Range | `bg-linear-to-r from-primary/55 to-primary` with `[background-size:100cqw_100%]`, so the gradient is sized to the **track** and clipped by the fill: a low value shows the dim start, a high value the full brightness ("the tint follows how much it's on"). Vertical sliders fall back to a solid fill (`data-[orientation=vertical]:bg-none`). |
| Thumb | 14px, `border-2 border-primary`, `bg-background`, a 2px `ring-offset-background` gap to the fill, a `ring-primary/20` halo on hover (`/30` active, `/35` focus-visible), 10% scale-up on hover and active |

`cqw` needs container-query units (Chrome 105, Safari 16, Firefox 110). Older browsers ignore the
`background-size` and get a gradient that spans the fill instead, which still looks right.

`SliderRow.tsx` (label + readout + slider) is unchanged. Proposed but not adopted, still open if
wanted: a value bubble above the thumb while dragging, a default-value notch with double-click to
reset, step ticks for sliders with 12 stops or fewer, and an accent-tinted readout while the Topics
slider's value is uncommitted on large collections.

## Tests

`lib/ui-primitives/__tests__/dotControl.test.ts` renders the primitives with React's server renderer
(the vitest environment is `node`, no jsdom) and pins the contract: a checked checkbox renders the dot
and no `<svg>`, an unchecked one renders no dot, checkbox and radio item share the same class list, and
the slider keeps the 24px target, the `@container` track and the track-sized gradient.

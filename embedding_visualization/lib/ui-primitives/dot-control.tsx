import * as React from "react"

import { cn } from "@/lib/utils/utils"

/**
 * The one look for on/off and pick-one controls: a 16px ring whose dot fills
 * when the control is on. `Checkbox` and `RadioGroupItem` both draw with these
 * (no tick marks anywhere in the app), so the two cannot drift apart. The dot
 * takes the control's text colour, so `className="text-emerald-500"` recolours
 * it. Design notes: documentation/FORM_CONTROL_STYLES.md.
 */
export const dotControlClassName = cn(
  "grid aspect-square size-4 shrink-0 place-content-center rounded-full border border-input text-primary shadow-xs outline-none dark:bg-input/30",
  "transition-[color,border-color,box-shadow] motion-reduce:transition-none",
  "hover:border-foreground/40 data-[state=checked]:border-primary",
  "focus-visible:border-ring focus-visible:ring-[3px] focus-visible:ring-ring/50",
  "aria-invalid:border-destructive aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40",
  "disabled:cursor-not-allowed disabled:opacity-50"
)

/** The filled dot of a control that is on. */
export function DotIndicator({ className, ...props }: React.ComponentProps<"span">) {
  return (
    <span
      data-slot="dot-indicator"
      aria-hidden
      className={cn("block size-2 rounded-full bg-current", className)}
      {...props}
    />
  )
}

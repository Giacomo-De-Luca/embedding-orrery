import * as React from "react"

import { cn } from "@/lib/utils/utils"

/**
 * The one look for on/off and pick-one controls: a 14px ring (the line box of a
 * `text-sm leading-none` label, so a row sits on one line) whose 6px dot fills
 * when the control is on. `Checkbox` and `RadioGroupItem` both draw with these,
 * so the two cannot drift apart (no tick marks on on/off controls; menus and
 * selects keep their check marks). The dot takes the control's text colour, so
 * `className="text-emerald-500"` recolours it. Design notes:
 * documentation/FORM_CONTROL_STYLES.md.
 */
export const dotControlClassName = cn(
  "grid aspect-square size-3.5 shrink-0 place-content-center rounded-full border border-input bg-transparent text-primary shadow-xs outline-none",
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
      className={cn("block size-1.5 rounded-full bg-current", className)}
      {...props}
    />
  )
}

/** Indent for content nested under a dot control's row: the 14px ring plus the 8px row gap. */
export const dotControlIndentClassName = "pl-5.5"

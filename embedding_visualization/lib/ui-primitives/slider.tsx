"use client"

import * as React from "react"
import { Slider as SliderPrimitive } from "radix-ui"

import { cn } from "@/lib/utils/utils"

/**
 * Hairline slider: a 4px track, a fill that brightens along the track (the
 * gradient is sized to the track through container units, so a low value stays
 * dim), a 14px ring thumb with a gap to the fill, and a 24px pointer target.
 * Design notes: documentation/FORM_CONTROL_STYLES.md.
 */
function Slider({
  className,
  defaultValue,
  value,
  min = 0,
  max = 100,
  ...props
}: React.ComponentProps<typeof SliderPrimitive.Root>) {
  const _values = React.useMemo(
    () =>
      Array.isArray(value)
        ? value
        : Array.isArray(defaultValue)
          ? defaultValue
          : [min, max],
    [value, defaultValue, min, max]
  )

  return (
    <SliderPrimitive.Root
      data-slot="slider"
      defaultValue={defaultValue}
      value={value}
      min={min}
      max={max}
      className={cn(
        "relative flex h-6 w-full touch-none items-center select-none data-[disabled]:opacity-50 data-[orientation=vertical]:h-full data-[orientation=vertical]:min-h-44 data-[orientation=vertical]:w-6 data-[orientation=vertical]:flex-col",
        className
      )}
      {...props}
    >
      <SliderPrimitive.Track
        data-slot="slider-track"
        className="@container relative grow overflow-hidden rounded-full bg-foreground/12 data-[orientation=horizontal]:h-1 data-[orientation=horizontal]:w-full data-[orientation=vertical]:h-full data-[orientation=vertical]:w-1"
      >
        <SliderPrimitive.Range
          data-slot="slider-range"
          className="absolute bg-linear-to-r from-primary/55 to-primary bg-no-repeat [background-size:100cqw_100%] data-[orientation=horizontal]:h-full data-[orientation=vertical]:w-full data-[orientation=vertical]:bg-none data-[orientation=vertical]:bg-primary"
        />
      </SliderPrimitive.Track>
      {Array.from({ length: _values.length }, (_, index) => (
        <SliderPrimitive.Thumb
          data-slot="slider-thumb"
          key={index}
          className="block size-3.5 shrink-0 rounded-full border-2 border-primary bg-background ring-4 ring-primary/0 ring-offset-2 ring-offset-background outline-hidden transition-[scale,box-shadow] duration-150 motion-reduce:transition-none hover:scale-110 hover:ring-primary/20 focus-visible:ring-primary/35 active:scale-110 active:ring-primary/30 data-[disabled]:pointer-events-none"
        />
      ))}
    </SliderPrimitive.Root>
  )
}

export { Slider }

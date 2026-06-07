"use client"

import Color from "color"
import { PipetteIcon } from "lucide-react"
import { Slider } from "radix-ui"
import {
  type ComponentProps,
  createContext,
  type HTMLAttributes,
  memo,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react"
import { Button } from "@/lib/ui-primitives/button"
import { Input } from "@/lib/ui-primitives/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/lib/ui-primitives/select"
import { cn } from "@/lib/utils/utils"

type HslColor = { h: number; s: number; l: number; a: number }

interface ColorPickerContextValue {
  color: HslColor
  setColor: (patch: Partial<HslColor>) => void
  mode: string
  setMode: (mode: string) => void
}

const ColorPickerContext = createContext<ColorPickerContextValue | undefined>(undefined)

const useColorPicker = () => {
  const context = useContext(ColorPickerContext)

  if (!context) {
    throw new Error("useColorPicker must be used within a ColorPickerProvider")
  }

  return context
}

export type ColorPickerProps = HTMLAttributes<HTMLDivElement> & {
  value?: Parameters<typeof Color>[0]
  defaultValue?: Parameters<typeof Color>[0]
  onChange?: (value: Parameters<typeof Color.rgb>[0]) => void
}

export const ColorPicker = ({
  value,
  defaultValue = "#000000",
  onChange,
  className,
  ...props
}: ColorPickerProps) => {
  const selectedColor = Color(value)
  const defaultColor = Color(defaultValue)

  const [color, setColorState] = useState<HslColor>({
    h: selectedColor.hue() || defaultColor.hue() || 0,
    s: selectedColor.saturationl() || defaultColor.saturationl() || 100,
    l: selectedColor.lightness() || defaultColor.lightness() || 50,
    a: selectedColor.alpha() * 100 || defaultColor.alpha() * 100,
  })
  const [mode, setMode] = useState("hex")

  // Ref keeps current color accessible in setColor without stale closures
  const colorRef = useRef(color)
  colorRef.current = color

  // Sync state from controlled prop (no onChange — breaks the feedback loop)
  useEffect(() => {
    if (value) {
      const c = Color(value)
      const [h, s, l] = c.hsl().array()
      setColorState({ h, s, l, a: c.alpha() * 100 })
    }
  }, [value])

  // User-driven update: merge patch into current color, set state, and notify parent
  const setColor = useCallback((patch: Partial<HslColor>) => {
    const next = { ...colorRef.current, ...patch }
    setColorState(next)
    if (onChange) {
      const c = Color.hsl(next.h, next.s, next.l).alpha(next.a / 100)
      const rgba = c.rgb().array()
      onChange([rgba[0], rgba[1], rgba[2], next.a / 100])
    }
  }, [onChange])

  return (
    <ColorPickerContext.Provider value={{ color, setColor, mode, setMode }}>
      <div className={cn("flex size-full flex-col gap-4", className)} {...(props as any)} />
    </ColorPickerContext.Provider>
  )
}

export type ColorPickerSelectionProps = HTMLAttributes<HTMLDivElement>

export const ColorPickerSelection = memo(({ className, ...props }: ColorPickerSelectionProps) => {
  const containerRef = useRef<HTMLDivElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [positionX, setPositionX] = useState(0)
  const [positionY, setPositionY] = useState(0)
  const { color, setColor } = useColorPicker()

  const backgroundGradient = useMemo(() => {
    return `linear-gradient(0deg, rgba(0,0,0,1), rgba(0,0,0,0)),
            linear-gradient(90deg, rgba(255,255,255,1), rgba(255,255,255,0)),
            hsl(${color.h}, 100%, 50%)`
  }, [color.h])

  const handlePointerMove = useCallback(
    (event: PointerEvent) => {
      if (!(isDragging && containerRef.current)) {
        return
      }
      const rect = containerRef.current.getBoundingClientRect()
      const x = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width))
      const y = Math.max(0, Math.min(1, (event.clientY - rect.top) / rect.height))
      setPositionX(x)
      setPositionY(y)
      const s = x * 100
      const topLightness = x < 0.01 ? 100 : 50 + 50 * (1 - x)
      const l = topLightness * (1 - y)
      setColor({ s, l })
    },
    [isDragging, setColor],
  )

  useEffect(() => {
    const handlePointerUp = () => setIsDragging(false)

    if (isDragging) {
      window.addEventListener("pointermove", handlePointerMove)
      window.addEventListener("pointerup", handlePointerUp)
    }

    return () => {
      window.removeEventListener("pointermove", handlePointerMove)
      window.removeEventListener("pointerup", handlePointerUp)
    }
  }, [isDragging, handlePointerMove])

  return (
    <div
      className={cn("relative size-full cursor-crosshair rounded", className)}
      onPointerDown={e => {
        e.preventDefault()
        setIsDragging(true)
        handlePointerMove(e.nativeEvent)
      }}
      ref={containerRef}
      style={{
        background: backgroundGradient,
      }}
      {...(props as any)}
    >
      <div
        className="-translate-x-1/2 -translate-y-1/2 pointer-events-none absolute h-4 w-4 rounded-full border-2 border-white"
        style={{
          left: `${positionX * 100}%`,
          top: `${positionY * 100}%`,
          boxShadow: "0 0 0 1px rgba(0,0,0,0.5)",
        }}
      />
    </div>
  )
})

ColorPickerSelection.displayName = "ColorPickerSelection"

export type ColorPickerHueProps = ComponentProps<typeof Slider.Root>

export const ColorPickerHue = ({ className, ...props }: ColorPickerHueProps) => {
  const { color, setColor } = useColorPicker()

  return (
    <Slider.Root
      className={cn("relative flex h-4 w-full touch-none", className)}
      max={360}
      onValueChange={([h]) => setColor({ h })}
      step={1}
      value={[color.h]}
      {...(props as any)}
    >
      <Slider.Track className="relative my-0.5 h-3 w-full grow rounded-full bg-[linear-gradient(90deg,#FF0000,#FFFF00,#00FF00,#00FFFF,#0000FF,#FF00FF,#FF0000)]">
        <Slider.Range className="absolute h-full" />
      </Slider.Track>
      <Slider.Thumb className="block h-4 w-4 rounded-full border border-primary/50 bg-background shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50" />
    </Slider.Root>
  )
}

export type ColorPickerAlphaProps = ComponentProps<typeof Slider.Root>

export const ColorPickerAlpha = ({ className, ...props }: ColorPickerAlphaProps) => {
  const { color, setColor } = useColorPicker()

  return (
    <Slider.Root
      className={cn("relative flex h-4 w-full touch-none", className)}
      max={100}
      onValueChange={([a]) => setColor({ a })}
      step={1}
      value={[color.a]}
      {...(props as any)}
    >
      <Slider.Track
        className="relative my-0.5 h-3 w-full grow rounded-full"
        style={{
          background:
            'url("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAMUlEQVQ4T2NkYGAQYcAP3uCTZhw1gGGYhAGBZIA/nYDCgBDAm9BGDWAAJyRCgLaBCAAgXwixzAS0pgAAAABJRU5ErkJggg==") left center',
        }}
      >
        <div className="absolute inset-0 rounded-full bg-gradient-to-r from-transparent to-black/50" />
        <Slider.Range className="absolute h-full rounded-full bg-transparent" />
      </Slider.Track>
      <Slider.Thumb className="block h-4 w-4 rounded-full border border-primary/50 bg-background shadow transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring disabled:pointer-events-none disabled:opacity-50" />
    </Slider.Root>
  )
}

export type ColorPickerEyeDropperProps = ComponentProps<typeof Button>

export const ColorPickerEyeDropper = ({ className, ...props }: ColorPickerEyeDropperProps) => {
  const { setColor } = useColorPicker()

  const handleEyeDropper = async () => {
    try {
      // @ts-expect-error - EyeDropper API is experimental
      const eyeDropper = new EyeDropper()
      const result = await eyeDropper.open()
      const c = Color(result.sRGBHex)
      const [h, s, l] = c.hsl().array()
      setColor({ h, s, l, a: 100 })
    } catch (error) {
      console.error("EyeDropper failed:", error)
    }
  }

  return (
    <Button
      className={cn("shrink-0 text-muted-foreground", className)}
      onClick={handleEyeDropper}
      size="icon"
      type="button"
      variant="outline"
      {...(props as any)}
    >
      <PipetteIcon size={16} />
    </Button>
  )
}

export type ColorPickerOutputProps = ComponentProps<typeof SelectTrigger>

const formats = ["hex", "rgb", "css", "hsl"]

export const ColorPickerOutput = ({ className, ...props }: ColorPickerOutputProps) => {
  const { mode, setMode } = useColorPicker()

  return (
    <Select onValueChange={setMode} value={mode}>
      <SelectTrigger className="h-8 w-20 shrink-0 text-xs" {...(props as any)}>
        <SelectValue placeholder="Mode" />
      </SelectTrigger>
      <SelectContent>
        {formats.map(format => (
          <SelectItem className="text-xs" key={format} value={format}>
            {format.toUpperCase()}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}

/**
 * Editable numeric input that commits on blur or Enter.
 * Shows a percentage suffix when `showPercent` is true.
 */
const EditableNumberInput = ({ value, onChange, min = 0, max = 255, showPercent, className }: {
  value: number
  onChange: (v: number) => void
  min?: number
  max?: number
  showPercent?: boolean
  className?: string
}) => {
  const [draft, setDraft] = useState(String(Math.round(value)))
  const prevValue = useRef(value)

  // Sync draft when value changes externally (e.g. from dragging the picker)
  useEffect(() => {
    if (Math.round(value) !== Math.round(prevValue.current)) {
      setDraft(String(Math.round(value)))
      prevValue.current = value
    }
  }, [value])

  const commit = () => {
    const n = parseFloat(draft)
    if (!isNaN(n) && isFinite(n)) {
      const clamped = Math.max(min, Math.min(max, Math.round(n)))
      onChange(clamped)
      setDraft(String(clamped))
    } else {
      setDraft(String(Math.round(value)))
    }
  }

  return (
    <div className="relative">
      <Input
        type="text"
        value={draft}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
        className={cn(
          "h-8 bg-secondary px-2 text-xs shadow-none",
          showPercent && "pr-5",
          className,
        )}
      />
      {showPercent && (
        <span className="-translate-y-1/2 absolute top-1/2 right-2 text-muted-foreground text-xs">
          %
        </span>
      )}
    </div>
  )
}

export type ColorPickerFormatProps = HTMLAttributes<HTMLDivElement>

export const ColorPickerFormat = ({ className, ...props }: ColorPickerFormatProps) => {
  const { color, setColor, mode } = useColorPicker()
  const colorObj = Color.hsl(color.h, color.s, color.l, color.a / 100)

  if (mode === "hex") {
    const hex = colorObj.hex()

    const commitHex = (input: string) => {
      try {
        const cleaned = input.startsWith('#') ? input : `#${input}`
        const parsed = Color(cleaned)
        const [h, s, l] = parsed.hsl().array()
        setColor({ h, s, l })
      } catch { /* ignore invalid hex */ }
    }

    return (
      <div
        className={cn(
          "-space-x-px relative flex w-full items-center rounded-md shadow-sm",
          className,
        )}
        {...(props as any)}
      >
        <HexInput value={hex} onCommit={commitHex} className="rounded-r-none" />
        <EditableNumberInput
          value={color.a}
          onChange={(a) => setColor({ a })}
          min={0}
          max={100}
          showPercent
          className="w-[3.25rem] rounded-l-none"
        />
      </div>
    )
  }

  if (mode === "rgb") {
    const rgb = colorObj.rgb().array().map(v => Math.round(v))
    const setChannel = (index: number, v: number) => {
      const newRgb = [...rgb]
      newRgb[index] = v
      try {
        const c = Color.rgb(newRgb[0], newRgb[1], newRgb[2])
        const [h, s, l] = c.hsl().array()
        setColor({ h, s, l })
      } catch { /* ignore */ }
    }

    return (
      <div
        className={cn("-space-x-px flex items-center rounded-md shadow-sm", className)}
        {...(props as any)}
      >
        {rgb.map((value, index) => (
          <EditableNumberInput
            key={index}
            value={value}
            onChange={(v) => setChannel(index, v)}
            min={0}
            max={255}
            className={cn(
              "w-14 rounded-r-none",
              index > 0 && "rounded-l-none",
            )}
          />
        ))}
        <EditableNumberInput
          value={color.a}
          onChange={(a) => setColor({ a })}
          min={0}
          max={100}
          showPercent
          className="w-[3.25rem] rounded-l-none"
        />
      </div>
    )
  }

  if (mode === "css") {
    const rgb = colorObj.rgb().array().map(v => Math.round(v))

    return (
      <div className={cn("w-full rounded-md shadow-sm", className)} {...(props as any)}>
        <Input
          className="h-8 w-full bg-secondary px-2 text-xs shadow-none"
          readOnly
          type="text"
          value={`rgba(${rgb.join(", ")}, ${color.a}%)`}
        />
      </div>
    )
  }

  if (mode === "hsl") {
    const hslKeys: (keyof HslColor)[] = ['h', 's', 'l']
    const hslMax = [360, 100, 100]

    return (
      <div
        className={cn("-space-x-px flex items-center rounded-md shadow-sm", className)}
        {...(props as any)}
      >
        {hslKeys.map((key, index) => (
          <EditableNumberInput
            key={key}
            value={Math.round(color[key])}
            onChange={(v) => setColor({ [key]: v })}
            min={0}
            max={hslMax[index]}
            className={cn(
              "w-14 rounded-r-none",
              index > 0 && "rounded-l-none",
            )}
          />
        ))}
        <EditableNumberInput
          value={color.a}
          onChange={(a) => setColor({ a })}
          min={0}
          max={100}
          showPercent
          className="w-[3.25rem] rounded-l-none"
        />
      </div>
    )
  }

  return null
}

/**
 * Editable hex color input that commits on blur or Enter.
 */
const HexInput = ({ value, onCommit, className }: {
  value: string
  onCommit: (hex: string) => void
  className?: string
}) => {
  const [draft, setDraft] = useState(value)
  const prevValue = useRef(value)

  useEffect(() => {
    if (value !== prevValue.current) {
      setDraft(value)
      prevValue.current = value
    }
  }, [value])

  const commit = () => {
    onCommit(draft)
  }

  return (
    <Input
      type="text"
      value={draft}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => { if (e.key === 'Enter') e.currentTarget.blur() }}
      className={cn("h-8 bg-secondary px-2 text-xs shadow-none", className)}
    />
  )
}

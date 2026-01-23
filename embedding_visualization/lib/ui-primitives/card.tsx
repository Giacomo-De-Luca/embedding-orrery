import * as React from "react"

import { cn } from "@/lib/utils/utils"
import { cva, type VariantProps } from "class-variance-authority"
import { Slot } from "@radix-ui/react-slot"




const cardVariant = cva(
  "dark:bg-input/10 backdrop-blur-sm flex flex-col gap-6 rounded-xl py-6 shadow-sm",  {
    variants: {
      variant: {
        default: "",
        destructive:
          "bg-destructive text-white hover:bg-destructive/90 focus-visible:ring-destructive/20 dark:focus-visible:ring-destructive/40 dark:bg-destructive/60",
        outline:
          "border",
        secondary:
          "bg-secondary text-secondary-foreground hover:bg-secondary/80",
        ghost:
          "hover:bg-accent hover:text-accent-foreground dark:hover:bg-accent/50",
        circular:
          "bg-secondary/30 backdrop-blur-md border-glass text-secondary-foreground  rounded-full aspect-square  hover:bg-secondary/80",
        circularghost:
          "hover:bg-accent backdrop-blur-md border-glass hover:text-accent-foreground rounded-full aspect-square dark:hover:bg-accent/50",
        link: "text-primary underline-offset-4 hover:underline",

      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
)




function Card({ className, 
                variant,
                asChild = false,
                ...props 
              }: React.ComponentProps<"div">  &
                  VariantProps<typeof cardVariant> & {
                      asChild?: boolean
                    }) {
                    const Comp = asChild ? Slot : "div"
                  
                  return (


                  <Comp
                    data-slot="card"
                    className={cn(cardVariant({ variant, className }))}
                    {...props}
                  />
                )
              }

function CardHeader({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-header"
      className={cn(
        "@container/card-header grid auto-rows-min grid-rows-[auto_auto] items-start gap-2 px-6 has-data-[slot=card-action]:grid-cols-[1fr_auto] [.border-b]:pb-2",
        className
      )}
      {...props}
    />
  )
}

function CardTitle({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-title"
      className={cn("leading-none font-semibold", className)}
      {...props}
    />
  )
}

function CardDescription({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-description"
      className={cn("text-muted-foreground text-sm", className)}
      {...props}
    />
  )
}

function CardAction({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-action"
      className={cn(
        "col-start-2 row-span-2 row-start-1 self-start justify-self-end",
        className
      )}
      {...props}
    />
  )
}

function CardContent({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-content"
      className={cn("px-6", className)}
      {...props}
    />
  )
}

function CardFooter({ className, ...props }: React.ComponentProps<"div">) {
  return (
    <div
      data-slot="card-footer"
      className={cn("flex items-center px-6 [.border-t]:pt-6", className)}
      {...props}
    />
  )
}

export {
  Card,
  CardHeader,
  CardFooter,
  CardTitle,
  CardAction,
  CardDescription,
  CardContent,
}

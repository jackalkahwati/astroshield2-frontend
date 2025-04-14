"use client"

import { useState } from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { format } from "date-fns"
import { CalendarIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "@/components/ui/form"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Input } from "@/components/ui/input"
import { Calendar } from "@/components/ui/calendar"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { createManeuver } from "@/lib/api-client"
import { useToast } from "@/components/ui/use-toast"
import type { ManeuverData } from "@/lib/types"

const maneuverTypes = [
  { value: "hohmann", label: "Hohmann Transfer" },
  { value: "stationkeeping", label: "Station Keeping" },
  { value: "phasing", label: "Phasing Maneuver" },
  { value: "collision", label: "Collision Avoidance" }
] as const

const formSchema = z.object({
  type: z.enum(["hohmann", "stationkeeping", "phasing", "collision"] as const),
  scheduledTime: z.date(),
  delta_v: z.number().min(0),
  duration: z.number().min(0),
  fuel_required: z.number().min(0),
})

type FormValues = z.infer<typeof formSchema>

export function PlanManeuverForm() {
  const [open, setOpen] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const { toast } = useToast()
  
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      type: "hohmann",
      delta_v: 0,
      duration: 0,
      fuel_required: 0,
    },
  })

  async function onSubmit(data: FormValues) {
    try {
      setIsSubmitting(true);
      const maneuverData = {
        type: data.type,
        status: "scheduled",
        scheduledTime: data.scheduledTime.toISOString(),
        details: {
          delta_v: data.delta_v,
          duration: data.duration,
          fuel_required: data.fuel_required,
        }
      }
      
      const response = await fetch("/api/v1/maneuvers", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(maneuverData),
      });
      
      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || "Failed to create maneuver");
      }
      
      toast({
        title: "Maneuver Planned",
        description: "The maneuver has been scheduled successfully.",
      })
      
      setOpen(false)
      form.reset()
      
      // Refresh the page to show the new maneuver
      window.location.reload();
    } catch (error) {
      console.error("Error creating maneuver:", error);
      toast({
        title: "Error",
        description: error instanceof Error ? error.message : "Failed to plan maneuver. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false);
    }
  }

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button>Plan New Maneuver</Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Plan New Maneuver</DialogTitle>
          <DialogDescription>
            Schedule a new orbital maneuver. All fields are required.
          </DialogDescription>
        </DialogHeader>
        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <FormField
              control={form.control}
              name="type"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Maneuver Type</FormLabel>
                  <Select onValueChange={field.onChange} defaultValue={field.value}>
                    <FormControl>
                      <SelectTrigger>
                        <SelectValue placeholder="Select maneuver type" />
                      </SelectTrigger>
                    </FormControl>
                    <SelectContent>
                      {maneuverTypes.map((type) => (
                        <SelectItem key={type.value} value={type.value}>
                          {type.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="scheduledTime"
              render={({ field }) => (
                <FormItem className="flex flex-col">
                  <FormLabel>Scheduled Time</FormLabel>
                  <Popover>
                    <PopoverTrigger asChild>
                      <FormControl>
                        <Button
                          variant={"outline"}
                          className={cn(
                            "w-full pl-3 text-left font-normal",
                            !field.value && "text-muted-foreground"
                          )}
                        >
                          {field.value ? (
                            format(field.value, "PPP")
                          ) : (
                            <span>Pick a date</span>
                          )}
                          <CalendarIcon className="ml-auto h-4 w-4 opacity-50" />
                        </Button>
                      </FormControl>
                    </PopoverTrigger>
                    <PopoverContent className="w-auto p-0" align="start">
                      <Calendar
                        mode="single"
                        selected={field.value}
                        onSelect={field.onChange}
                        disabled={(date) =>
                          date < new Date() || date < new Date("1900-01-01")
                        }
                        initialFocus
                      />
                    </PopoverContent>
                  </Popover>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="delta_v"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Delta-V (m/s)</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      {...field}
                      onChange={e => field.onChange(parseFloat(e.target.value))}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="duration"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Duration (seconds)</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      {...field}
                      onChange={e => field.onChange(parseFloat(e.target.value))}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="fuel_required"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Fuel Required (kg)</FormLabel>
                  <FormControl>
                    <Input
                      type="number"
                      {...field}
                      onChange={e => field.onChange(parseFloat(e.target.value))}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />
            <DialogFooter>
              <Button type="submit" disabled={isSubmitting}>
                {isSubmitting ? "Planning..." : "Plan Maneuver"}
              </Button>
            </DialogFooter>
          </form>
        </Form>
      </DialogContent>
    </Dialog>
  )
} 
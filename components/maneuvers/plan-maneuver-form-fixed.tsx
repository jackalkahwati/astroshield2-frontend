"use client"

import { useState } from "react"
import { zodResolver } from "@hookform/resolvers/zod"
import { useForm } from "react-hook-form"
import * as z from "zod"
import { format } from "date-fns"
import { CalendarIcon, ClockIcon } from "lucide-react"
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
import { useToast } from "@/components/ui/use-toast"

const maneuverTypes = [
  { value: "collision_avoidance", label: "Collision Avoidance" },
  { value: "station_keeping", label: "Station Keeping" },
  { value: "hohmann_transfer", label: "Hohmann Transfer" },
  { value: "phasing", label: "Phasing Maneuver" }
] as const

const satelliteOptions = [
  { value: "sat-001", label: "ASTROSHIELD-1" },
  { value: "sat-002", label: "ASTROSHIELD-2" },
  { value: "sat-003", label: "SENTINEL-1" },
  { value: "sat-004", label: "GUARDIAN-1" }
] as const

const formSchema = z.object({
  satellite_id: z.string().min(1, "Please select a satellite"),
  type: z.enum(["collision_avoidance", "station_keeping", "hohmann_transfer", "phasing"] as const),
  scheduled_date: z.date({
    required_error: "Please select a date",
  }),
  scheduled_time: z.string().min(1, "Please select a time"),
  delta_v: z.number().min(0.001, "Delta-V must be greater than 0"),
  burn_duration: z.number().min(1, "Duration must be at least 1 second"),
  direction_x: z.number().min(-1).max(1),
  direction_y: z.number().min(-1).max(1),
  direction_z: z.number().min(-1).max(1),
  priority: z.number().min(1).max(5),
  notes: z.string().optional(),
})

type FormValues = z.infer<typeof formSchema>

export function PlanManeuverForm() {
  const [open, setOpen] = useState(false)
  const [isSubmitting, setIsSubmitting] = useState(false)
  const { toast } = useToast()
  
  const form = useForm<FormValues>({
    resolver: zodResolver(formSchema),
    defaultValues: {
      satellite_id: "",
      type: "collision_avoidance",
      scheduled_time: "12:00",
      delta_v: 0.01,
      burn_duration: 10,
      direction_x: 0.0,
      direction_y: 0.0,
      direction_z: 0.1,
      priority: 3,
      notes: "",
    },
  })

  async function onSubmit(data: FormValues) {
    try {
      setIsSubmitting(true);
      
      // Combine date and time
      const scheduledDateTime = new Date(data.scheduled_date);
      const [hours, minutes] = data.scheduled_time.split(':');
      scheduledDateTime.setHours(parseInt(hours), parseInt(minutes), 0, 0);
      
      // Format data for backend
      const maneuverData = {
        satellite_id: data.satellite_id,
        type: data.type,
        scheduled_time: scheduledDateTime.toISOString(),
        parameters: {
          delta_v: data.delta_v,
          burn_duration: data.burn_duration,
          direction: {
            x: data.direction_x,
            y: data.direction_y,
            z: data.direction_z
          },
          target_orbit: null
        },
        priority: data.priority,
        notes: data.notes || undefined
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
        title: "Maneuver Planned Successfully",
        description: `${maneuverTypes.find(t => t.value === data.type)?.label} scheduled for ${format(scheduledDateTime, "PPP 'at' p")}`,
      })
      
      setOpen(false)
      form.reset()
      
      // Refresh the page
      window.location.reload();
    } catch (error) {
      console.error("Error creating maneuver:", error);
      toast({
        title: "Error Planning Maneuver",
        description: error instanceof Error ? error.message : "Failed to plan maneuver. Please try again.",
        variant: "destructive",
      })
    } finally {
      setIsSubmitting(false);
    }
  }

  const presetManeuvers = [
    {
      name: "Emergency Collision Avoidance",
      type: "collision_avoidance" as const,
      delta_v: 0.05,
      burn_duration: 30,
      direction: { x: 0.1, y: 0.0, z: -0.1 },
      priority: 5
    },
    {
      name: "Standard Station Keeping",
      type: "station_keeping" as const,
      delta_v: 0.01,
      burn_duration: 10,
      direction: { x: 0.0, y: 0.0, z: 0.1 },
      priority: 2
    },
    {
      name: "Orbit Raise Maneuver",
      type: "hohmann_transfer" as const,
      delta_v: 0.15,
      burn_duration: 180,
      direction: { x: 0.0, y: 1.0, z: 0.0 },
      priority: 3
    }
  ];

  const loadPreset = (preset: typeof presetManeuvers[0]) => {
    form.setValue("type", preset.type);
    form.setValue("delta_v", preset.delta_v);
    form.setValue("burn_duration", preset.burn_duration);
    form.setValue("direction_x", preset.direction.x);
    form.setValue("direction_y", preset.direction.y);
    form.setValue("direction_z", preset.direction.z);
    form.setValue("priority", preset.priority);
  };

  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger asChild>
        <Button className="bg-blue-600 hover:bg-blue-700">Plan New Maneuver</Button>
      </DialogTrigger>
      <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle>Plan New Maneuver</DialogTitle>
          <DialogDescription>
            Schedule a new orbital maneuver for your satellite. All fields are required.
          </DialogDescription>
        </DialogHeader>
        
        {/* Preset Maneuvers */}
        <div className="space-y-2">
          <div className="text-sm font-medium">Quick Presets</div>
          <div className="flex flex-wrap gap-2">
            {presetManeuvers.map((preset) => (
              <Button
                key={preset.name}
                type="button"
                variant="outline"
                size="sm"
                onClick={() => loadPreset(preset)}
                className="text-xs"
              >
                {preset.name}
              </Button>
            ))}
          </div>
        </div>

        <Form {...form}>
          <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="satellite_id"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Satellite</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Select satellite" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        {satelliteOptions.map((satellite) => (
                          <SelectItem key={satellite.value} value={satellite.value}>
                            {satellite.label}
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
                name="type"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Maneuver Type</FormLabel>
                    <Select onValueChange={field.onChange} value={field.value}>
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
            </div>

            <div className="grid grid-cols-2 gap-4">
              <FormField
                control={form.control}
                name="scheduled_date"
                render={({ field }) => (
                  <FormItem className="flex flex-col">
                    <FormLabel>Scheduled Date</FormLabel>
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
                name="scheduled_time"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Scheduled Time</FormLabel>
                    <FormControl>
                      <div className="relative">
                        <Input
                          type="time"
                          {...field}
                          className="pl-10"
                        />
                        <ClockIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 opacity-50" />
                      </div>
                    </FormControl>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="grid grid-cols-3 gap-4">
              <FormField
                control={form.control}
                name="delta_v"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Delta-V (m/s)</FormLabel>
                    <FormControl>
                      <Input
                        type="number"
                        step="0.001"
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
                name="burn_duration"
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
                name="priority"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Priority</FormLabel>
                    <Select onValueChange={(value) => field.onChange(parseInt(value))} value={field.value?.toString()}>
                      <FormControl>
                        <SelectTrigger>
                          <SelectValue placeholder="Priority" />
                        </SelectTrigger>
                      </FormControl>
                      <SelectContent>
                        <SelectItem value="1">1 - Low</SelectItem>
                        <SelectItem value="2">2 - Normal</SelectItem>
                        <SelectItem value="3">3 - Medium</SelectItem>
                        <SelectItem value="4">4 - High</SelectItem>
                        <SelectItem value="5">5 - Critical</SelectItem>
                      </SelectContent>
                    </Select>
                    <FormMessage />
                  </FormItem>
                )}
              />
            </div>

            <div className="space-y-2">
              <div className="text-sm font-medium">Direction Vector</div>
              <div className="grid grid-cols-3 gap-4">
                <FormField
                  control={form.control}
                  name="direction_x"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>X</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          step="0.1"
                          min="-1"
                          max="1"
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
                  name="direction_y"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Y</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          step="0.1"
                          min="-1"
                          max="1"
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
                  name="direction_z"
                  render={({ field }) => (
                    <FormItem>
                      <FormLabel>Z</FormLabel>
                      <FormControl>
                        <Input
                          type="number"
                          step="0.1"
                          min="-1"
                          max="1"
                          {...field}
                          onChange={e => field.onChange(parseFloat(e.target.value))}
                        />
                      </FormControl>
                      <FormMessage />
                    </FormItem>
                  )}
                />
              </div>
            </div>

            <FormField
              control={form.control}
              name="notes"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>Notes (Optional)</FormLabel>
                  <FormControl>
                    <Input
                      placeholder="Additional notes or comments..."
                      {...field}
                    />
                  </FormControl>
                  <FormMessage />
                </FormItem>
              )}
            />

            <DialogFooter>
              <Button type="button" variant="outline" onClick={() => setOpen(false)}>
                Cancel
              </Button>
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
"use client"

import { Skeleton } from "@/components/ui/skeleton"

export function SkeletonChart() {
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <Skeleton className="h-6 w-1/3" /> {/* Title skeleton */}
        <Skeleton className="h-8 w-[180px] rounded-md" /> {/* Chart type selector skeleton */}
      </div>
      <Skeleton className="h-[250px] w-full rounded-md" /> {/* Chart skeleton */}
      <div className="grid grid-cols-3 gap-3 pt-2"> {/* Legend skeleton */}
        <Skeleton className="h-5 w-20" />
        <Skeleton className="h-5 w-24" />
        <Skeleton className="h-5 w-16" />
      </div>
    </div>
  )
}

export function SkeletonStatCard() {
  return (
    <div className="space-y-2 p-6">
      <div className="flex justify-between items-center">
        <div className="flex items-center gap-2">
          <Skeleton className="h-5 w-5 rounded-full" /> {/* Icon skeleton */}
          <Skeleton className="h-5 w-24" /> {/* Title skeleton */}
        </div>
        <Skeleton className="h-6 w-16 rounded-full" /> {/* Badge skeleton */}
      </div>
      <Skeleton className="h-10 w-1/2 mt-2" /> {/* Value skeleton */}
      <Skeleton className="h-4 w-3/4 mt-1" /> {/* Description skeleton */}
      <div className="pt-4 mt-4 border-t flex justify-between">
        <Skeleton className="h-4 w-20" /> {/* Footer text skeleton */}
        <Skeleton className="h-4 w-16" /> {/* Footer value skeleton */}
      </div>
    </div>
  )
}

export function SkeletonListItem() {
  return (
    <div className="flex items-center space-x-4 py-2">
      <Skeleton className="h-10 w-10 rounded-full" /> {/* Avatar skeleton */}
      <div className="space-y-2 flex-1">
        <Skeleton className="h-4 w-3/4" /> {/* Title skeleton */}
        <Skeleton className="h-3 w-1/2" /> {/* Subtitle skeleton */}
      </div>
      <Skeleton className="h-6 w-16 rounded-full" /> {/* Badge/status skeleton */}
    </div>
  )
}

export function SkeletonTableRow() {
  return (
    <div className="flex w-full py-3">
      <Skeleton className="h-5 w-1/6 mr-2" />
      <Skeleton className="h-5 w-2/6 mr-2" />
      <Skeleton className="h-5 w-1/6 mr-2" />
      <Skeleton className="h-5 w-1/6 mr-2" />
      <Skeleton className="h-5 w-1/6" />
    </div>
  )
}

export function SkeletonDashboard() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {[...Array(3)].map((_, i) => (
          <Skeleton key={i} className="h-[180px] rounded-lg" />
        ))}
      </div>
      <Skeleton className="h-[350px] rounded-lg" />
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Skeleton className="h-[300px] rounded-lg" />
        <Skeleton className="h-[300px] rounded-lg" />
      </div>
    </div>
  )
} 
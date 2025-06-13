import { cn } from "@/lib/utils"

function Skeleton({
  className,
  ...props
}: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted", className)}
      {...props}
    />
  )
}

export { Skeleton }

// Add specialized skeleton components for common use cases
export function SkeletonCard() {
  return (
    <div className="space-y-3">
      <Skeleton className="h-6 w-1/3" />
      <Skeleton className="h-24 w-full" />
    </div>
  )
}

export function SkeletonButton() {
  return <Skeleton className="h-10 w-32 rounded-md" />
}

export function SkeletonAvatar() {
  return <Skeleton className="h-12 w-12 rounded-full" />
}

export function SkeletonListItem() {
  return (
    <div className="flex items-center space-x-4 py-2">
      <Skeleton className="h-10 w-10 rounded-full" />
      <div className="space-y-2 flex-1">
        <Skeleton className="h-4 w-3/4" />
        <Skeleton className="h-3 w-1/2" />
      </div>
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

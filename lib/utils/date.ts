import { format, parseISO } from "date-fns"

/**
 * Safely parses a date string and returns a Date object or null if invalid
 * @param dateString - The date string to parse
 * @returns Date object or null if invalid
 */
export function safeDate(dateString: string | null | undefined): Date | null {
  if (!dateString || isNaN(Date.parse(dateString))) {
    return null;
  }
  return new Date(dateString);
}

/**
 * Formats a date string safely, returning a fallback if the date is invalid
 * @param dateOrString - The date string or Date object to format
 * @param fallback - The fallback string to return if date is invalid (default: "N/A")
 * @returns Formatted date string or fallback
 */
export function formatDate(dateOrString: string | Date | null | undefined, fallback: string = "N/A"): string {
  if (!dateOrString) {
    return fallback
  }
  
  let validDate: Date
  if (dateOrString instanceof Date) {
    validDate = dateOrString
  } else if (!isNaN(Date.parse(dateOrString))) {
    validDate = new Date(dateOrString)
  } else {
    return fallback
  }

  return format(validDate, "PPp") // e.g., "Apr 29, 2023, 1:30 PM"
}

/**
 * Returns a relative time string (e.g., "2 minutes ago") or fallback if invalid
 * @param dateString - The date string to format
 * @param fallback - The fallback string to return if date is invalid (default: "N/A")
 * @returns Relative time string or fallback
 */
export function getRelativeTime(dateString: string | null | undefined, fallback: string = "N/A"): string {
  const date = safeDate(dateString);
  if (!date) {
    return fallback;
  }

  const now = new Date();
  const diffInSeconds = Math.floor((now.getTime() - date.getTime()) / 1000);

  if (diffInSeconds < 60) {
    return "just now";
  } else if (diffInSeconds < 3600) {
    const minutes = Math.floor(diffInSeconds / 60);
    return `${minutes} minute${minutes === 1 ? "" : "s"} ago`;
  } else if (diffInSeconds < 86400) {
    const hours = Math.floor(diffInSeconds / 3600);
    return `${hours} hour${hours === 1 ? "" : "s"} ago`;
  } else {
    const days = Math.floor(diffInSeconds / 86400);
    return `${days} day${days === 1 ? "" : "s"} ago`;
  }
} 
import { useTheme } from "@/components/providers/theme-provider"
import { getThemeColors, getChartConfig, getChartColorSequence } from "@/lib/chart-colors"

/**
 * Hook that provides theme-aware colors for charts and components
 * Automatically detects current theme and returns appropriate colors
 */
export const useThemeAwareColors = () => {
  const { theme } = useTheme()
  
  // Determine if we're in dark mode
  const isDark = theme === "dark" || (theme === "system" && 
    typeof window !== "undefined" && 
    window.matchMedia("(prefers-color-scheme: dark)").matches)
  
  return {
    colors: getThemeColors(isDark),
    chartConfig: getChartConfig(isDark), 
    colorSequence: getChartColorSequence(isDark),
    isDark
  }
} 
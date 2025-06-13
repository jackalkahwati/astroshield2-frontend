# AstroShield Theme System Implementation

## Overview

AstroShield now supports full dark/light mode switching with persistent preferences and theme-aware components optimized for space operations.

## Features Implemented

### 🌙 **Theme Provider System**
- **Custom Theme Provider**: Built-in theme context with localStorage persistence
- **Three Theme Options**: Dark, Light, and System (follows OS preference)
- **Default Dark Mode**: Optimized for space operations centers
- **Smooth Transitions**: 0.3s ease transitions for all theme changes

### 🎛️ **Theme Toggle Controls**
- **Dropdown Toggle**: Full theme selector with Dark/Light/System options
- **Simple Toggle**: Quick dark/light switch button
- **Top Bar Integration**: Accessible theme control in the main navigation
- **Icon Animations**: Smooth sun/moon icon transitions

### 🎨 **Theme-Aware Colors**

#### **Chart Colors**
- **Dynamic Color Functions**: `getThemeColors(isDark)` for all chart elements
- **Optimized Contrast**: Different color palettes for light vs dark backgrounds
- **Status Colors**: Consistent green/red/amber across themes
- **Chart Infrastructure**: Grid lines, axes, and tooltips adapt to theme

#### **Component Colors**
- **CSS Variables**: Automatic theme switching via CSS custom properties
- **Surface Colors**: `--surface-1`, `--surface-2`, `--surface-3` for layered UI
- **Text Colors**: `--text-primary`, `--text-muted` for proper contrast
- **Border Colors**: `--border-subtle` for consistent edge definition

### 🖼️ **UI Component Updates**

#### **Mapbox Integration**
- **Theme-Aware Styling**: Map controls and popups adapt to current theme
- **Light Mode**: White popups with dark text for daylight operations
- **Dark Mode**: Dark popups with light text for night operations

#### **Scrollbars & Animations**
- **Custom Scrollbars**: Styled for both light and dark themes
- **Smooth Animations**: All elements transition gracefully between themes
- **Chart Grids**: Grid lines adjust opacity and color for optimal visibility

## Usage Examples

### **Using Theme-Aware Colors in Components**

```tsx
import { useThemeAwareColors } from "@/hooks/use-theme-aware-colors"

export function MyChart() {
  const { colors, chartConfig, isDark } = useThemeAwareColors()
  
  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid 
          strokeDasharray="3 3" 
          stroke={colors.grid} 
        />
        <XAxis stroke={colors.axis} />
        <YAxis stroke={colors.axis} />
        <Tooltip contentStyle={chartConfig.tooltip.contentStyle} />
        <Bar 
          dataKey="value" 
          fill={colors.status.good} 
        />
      </BarChart>
    </ResponsiveContainer>
  )
}
```

### **Using CSS Variables**

```tsx
// Components automatically adapt via CSS variables
<div className="bg-surface-2 border border-border-subtle text-text-primary">
  Content adapts to theme automatically
</div>
```

### **Manual Theme Detection**

```tsx
import { useTheme } from "@/components/providers/theme-provider"

export function MyComponent() {
  const { theme, setTheme } = useTheme()
  
  return (
    <Button onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>
      Current theme: {theme}
    </Button>
  )
}
```

## Theme Optimization for Space Operations

### **Dark Mode (Default)**
- **Eye Strain Reduction**: Muted colors for extended viewing
- **High Contrast Alerts**: Bright red/orange only for critical situations
- **Professional Appearance**: Matches real space operations centers
- **Battery Efficiency**: Lower power consumption on OLED displays

### **Light Mode**
- **Daylight Operations**: Better visibility in bright environments
- **Documentation Mode**: Easier reading for reports and analysis
- **Accessibility**: Higher contrast for visually impaired operators
- **Print Friendly**: Better for printing charts and reports

## Implementation Notes

### **Performance Optimizations**
- **CSS Transitions**: Hardware-accelerated theme switching
- **Local Storage**: Instant theme restoration on page load
- **System Detection**: Automatic theme based on OS preference
- **Minimal Re-renders**: Efficient theme context updates

### **Accessibility Features**
- **WCAG Compliance**: Proper contrast ratios in both themes
- **Screen Reader Support**: Theme state announced to assistive technology
- **Keyboard Navigation**: Full keyboard access to theme controls
- **Reduced Motion**: Respects user's motion preferences

## File Structure

```
components/
├── providers/
│   └── theme-provider.tsx     # Main theme context provider
└── ui/
    └── theme-toggle.tsx       # Theme toggle components

hooks/
└── use-theme-aware-colors.ts  # Theme-aware color hook

lib/
└── chart-colors.ts           # Theme-aware color system

app/
├── globals.css               # Theme CSS variables and styles
└── layout.tsx               # Theme provider integration
```

## Testing the Implementation

1. **Toggle Themes**: Use the theme button in the top right to switch modes
2. **Check Persistence**: Refresh the page - theme should be preserved
3. **System Theme**: Set to "System" and change your OS theme
4. **Component Adaptation**: All charts, maps, and UI elements should adapt
5. **Smooth Transitions**: Theme changes should be smooth, not jarring

The theme system provides a professional, accessible experience suitable for 24/7 space operations while maintaining the visual clarity needed for mission-critical data analysis. 
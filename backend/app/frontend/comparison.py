"""Frontend utilities for trajectory comparisons."""

from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime

def generate_comparison_chart(metrics: Dict[str, Any]) -> str:
    """Generate a base64 encoded chart of trajectory comparison metrics."""
    # Create a figure for the chart
    plt.figure(figsize=(10, 6))
    
    # Extract relevant metrics for plotting
    impact_locations = metrics.get("impact_locations", [])
    if not impact_locations:
        return ""
    
    # Extract location data for scatter plot
    lats = [loc["lat"] for loc in impact_locations]
    lons = [loc["lon"] for loc in impact_locations]
    labels = [loc["name"] for loc in impact_locations]
    
    # Create scatter plot
    plt.scatter(lons, lats, s=100, alpha=0.7)
    
    # Add labels to points
    for i, label in enumerate(labels):
        plt.annotate(label, (lons[i], lats[i]), xytext=(5, 5), 
                    textcoords='offset points')
    
    # Add centroid if available
    if "impact_centroid" in metrics:
        centroid = metrics["impact_centroid"]
        plt.scatter([centroid["lon"]], [centroid["lat"]], 
                   marker='*', s=200, color='red', label='Centroid')
    
    # Add plot details
    plt.title('Impact Location Comparison')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    
    # Convert plot to base64 encoded string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100)
    buffer.seek(0)
    image_png = buffer.getvalue()
    buffer.close()
    
    encoded = base64.b64encode(image_png).decode('utf-8')
    return f"data:image/png;base64,{encoded}"

def generate_comparison_report(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a detailed comparison report based on trajectory metrics."""
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "trajectory_count": metrics.get("trajectory_count", 0),
        "charts": {},
        "summary": {}
    }
    
    # Add chart
    chart_image = generate_comparison_chart(metrics)
    if chart_image:
        report["charts"]["impact_locations"] = chart_image
    
    # Generate summary statistics
    if "impact_locations" in metrics and len(metrics["impact_locations"]) > 0:
        report["summary"]["max_distance_km"] = metrics.get("max_impact_distance_km", 0)
        
        # Calculate average uncertainty
        if "uncertainty_radii" in metrics:
            uncertainties = [ur["radius_km"] for ur in metrics["uncertainty_radii"]]
            report["summary"]["avg_uncertainty_km"] = sum(uncertainties) / len(uncertainties)
            report["summary"]["max_uncertainty_km"] = max(uncertainties)
            
        # Calculate time spread
        if "impact_times" in metrics:
            times = [datetime.fromisoformat(it["time"]) for it in metrics["impact_times"]]
            time_diffs = [(t - times[0]).total_seconds() for t in times]
            report["summary"]["max_time_difference_s"] = max(time_diffs) if time_diffs else 0
    
    return report
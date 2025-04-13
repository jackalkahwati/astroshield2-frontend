"""UDL Data Visualization Module

This module provides visualization capabilities for UDL data, working closely
with the UDLDataProcessor to display orbital data, conjunctions, and space weather
information in an intuitive graphical format.
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import logging
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .udl_data_processor import UDLDataProcessor

logger = logging.getLogger(__name__)

class UDLVisualizer:
    """Visualizer for UDL data processed by the UDLDataProcessor."""
    
    def __init__(self, data_processor: Optional[UDLDataProcessor] = None):
        """Initialize the UDL data visualizer.
        
        Args:
            data_processor: Optional UDLDataProcessor instance to use for data processing
        """
        self.data_processor = data_processor
        
        # Define color schemes for different risk levels
        self.risk_colors = {
            "CRITICAL": "#FF0000",  # Red
            "HIGH": "#FF7700",      # Orange
            "MODERATE": "#FFCC00",  # Yellow
            "LOW": "#00CC00"        # Green
        }
        
        # Initialize figure sizes and styles
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.style.use('ggplot')
    
    def plot_orbital_altitude(self, orbital_data: Dict[str, Any], save_path: Optional[str] = None) -> plt.Figure:
        """Plot orbital altitude over time for a space object.
        
        Args:
            orbital_data: Processed orbital data from UDLDataProcessor
            save_path: Optional path to save the plot image
            
        Returns:
            Matplotlib Figure object
        """
        # For demonstration, we'll create sample data based on the altitude profile
        # In a real implementation, this would extract time series data from state vectors
        
        # Create sample altitude data over time
        altitude_profile = orbital_data.get('altitude_profile', {})
        min_alt = altitude_profile.get('min_altitude', 500)
        max_alt = altitude_profile.get('max_altitude', 550)
        avg_alt = altitude_profile.get('average_altitude', 525)
        
        time_range = orbital_data.get('time_range', {})
        start_time = datetime.fromisoformat(time_range.get('start', datetime.utcnow().isoformat()))
        end_time = datetime.fromisoformat(time_range.get('end', datetime.utcnow().isoformat()))
        
        # Generate sample altitude data
        days = (end_time - start_time).days
        dates = [start_time + timedelta(hours=i*12) for i in range(days*2 + 1)]
        
        # Create synthetic altitude variations around the average
        np.random.seed(42)  # For reproducibility
        altitudes = [avg_alt + np.random.normal(0, (max_alt - min_alt) / 6) for _ in range(len(dates))]
        
        # Plot altitude over time
        fig, ax = plt.subplots()
        ax.plot(dates, altitudes, 'b-', linewidth=2)
        
        # Add maneuvers as vertical lines if any
        for maneuver in orbital_data.get('recent_maneuvers', []):
            maneuver_time = datetime.fromisoformat(maneuver.get('time', '').replace('Z', '+00:00'))
            if start_time <= maneuver_time <= end_time:
                ax.axvline(x=maneuver_time, color='r', linestyle='--', alpha=0.7, 
                           label=f"{maneuver.get('type', 'MANEUVER')} (ΔV: {maneuver.get('delta_v', 0)} m/s)")
        
        # Format the plot
        ax.set_title(f"Orbital Altitude for Object {orbital_data.get('object_id', 'Unknown')}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Altitude (km)")
        ax.grid(True, alpha=0.3)
        
        # Format date on x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, days//5)))
        plt.xticks(rotation=45)
        
        # Add legends if there were maneuvers
        if orbital_data.get('recent_maneuvers', []):
            plt.legend()
            
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def plot_conjunction_risk_timeline(self, conjunctions: List[Dict[str, Any]], 
                                      save_path: Optional[str] = None) -> plt.Figure:
        """Plot conjunction risks over time.
        
        Args:
            conjunctions: List of conjunction data from UDLDataProcessor
            save_path: Optional path to save the plot image
            
        Returns:
            Matplotlib Figure object
        """
        # Prepare data
        dates = []
        miss_distances = []
        risk_levels = []
        colors = []
        
        for conj in conjunctions:
            # Skip items without necessary data
            if not all(key in conj for key in ['time_of_closest_approach', 'miss_distance', 'risk_level']):
                continue
                
            # Convert time string to datetime
            time_str = conj['time_of_closest_approach'].replace('Z', '+00:00')
            try:
                tca = datetime.fromisoformat(time_str)
                dates.append(tca)
                miss_distances.append(conj['miss_distance'])
                risk_levels.append(conj['risk_level'])
                colors.append(self.risk_colors.get(conj['risk_level'], '#AAAAAA'))
            except (ValueError, TypeError):
                logger.warning(f"Invalid date format: {conj['time_of_closest_approach']}")
                
        # Create plot
        fig, ax = plt.subplots()
        
        # Sort by date
        if dates:
            # Convert to numpy arrays for easier manipulation
            dates_arr = np.array(dates)
            miss_distances_arr = np.array(miss_distances)
            colors_arr = np.array(colors)
            risk_levels_arr = np.array(risk_levels)
            
            # Get the indices that would sort dates
            sort_idx = np.argsort(dates_arr)
            
            # Plot scatter with size inversely proportional to miss distance
            sizes = 1000 / np.array(miss_distances_arr)[sort_idx]
            sizes = np.clip(sizes, 20, 200)  # Limit the size range
            
            scatter = ax.scatter(dates_arr[sort_idx], miss_distances_arr[sort_idx], 
                        c=colors_arr[sort_idx], s=sizes, alpha=0.7, edgecolors='black')
            
            # Draw a horizontal line at 25km (typical threshold for moderate risk)
            ax.axhline(y=25, color='orange', linestyle='--', alpha=0.5, label='Moderate Risk Threshold')
            
            # Draw a horizontal line at 5km (typical threshold for high risk)
            ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='High Risk Threshold')
            
            # Format the plot
            ax.set_title("Conjunction Risk Timeline")
            ax.set_xlabel("Date")
            ax.set_ylabel("Miss Distance (km)")
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Format date on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.xticks(rotation=45)
            
            # Add legend for risk levels
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=level,
                                     markerfacecolor=self.risk_colors[level], markersize=10)
                               for level in ['CRITICAL', 'HIGH', 'MODERATE', 'LOW'] if level in risk_levels_arr]
            
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            ax.text(0.5, 0.5, "No valid conjunction data available", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            
        return fig
    
    def plot_space_weather_conditions(self, weather_data: Dict[str, Any], 
                                    save_path: Optional[str] = None) -> plt.Figure:
        """Plot space weather conditions.
        
        Args:
            weather_data: Space weather data from UDLDataProcessor
            save_path: Optional path to save the plot image
            
        Returns:
            Matplotlib Figure object
        """
        # Create a figure with subplots
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f"Space Weather Conditions: {weather_data.get('timestamp', 'Current')}", fontsize=16)
        
        # 1. Plot solar activity (top left)
        solar_data = weather_data.get('solar_activity', {})
        solar_values = [
            solar_data.get('sunspot_number', 45), 
            solar_data.get('solar_flux', 110.5),
            10 if solar_data.get('flare_activity', 'LOW') == 'LOW' else 
               50 if solar_data.get('flare_activity', 'LOW') == 'MODERATE' else 
               100  # HIGH
        ]
        solar_labels = ['Sunspot Number', 'Solar Flux (10.7 cm)', 'Flare Activity']
        
        axs[0, 0].bar(solar_labels, solar_values, color=['skyblue', 'lightgreen', 'salmon'])
        axs[0, 0].set_title('Solar Activity')
        axs[0, 0].set_ylabel('Value')
        axs[0, 0].tick_params(axis='x', rotation=45)
        
        # 2. Plot geomagnetic activity (top right)
        geo_data = weather_data.get('geomagnetic_conditions', {})
        geo_values = [
            geo_data.get('kp_index', 2),
            abs(geo_data.get('dst_index', -15)),
            10 if geo_data.get('aurora_activity', 'LOW') == 'LOW' else 
               50 if geo_data.get('aurora_activity', 'LOW') == 'MODERATE' else 
               100  # HIGH
        ]
        geo_labels = ['Kp Index', '|Dst| Index', 'Aurora Activity']
        
        axs[0, 1].bar(geo_labels, geo_values, color=['lightgreen', 'skyblue', 'plum'])
        axs[0, 1].set_title('Geomagnetic Activity')
        axs[0, 1].set_ylabel('Value')
        axs[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Plot radiation levels (bottom left)
        rad_data = weather_data.get('radiation_levels', {})
        rad_values = [
            50 if rad_data.get('inner_belt_flux', 'NORMAL') == 'NORMAL' else 
               75 if rad_data.get('inner_belt_flux', 'NORMAL') == 'ELEVATED' else 
               100,  # HIGH
            50 if rad_data.get('outer_belt_flux', 'NORMAL') == 'NORMAL' else 
               75 if rad_data.get('outer_belt_flux', 'NORMAL') == 'ELEVATED' else 
               100,  # HIGH
            50 if rad_data.get('saa_intensity', 'NORMAL') == 'NORMAL' else 
               75 if rad_data.get('saa_intensity', 'NORMAL') == 'ELEVATED' else 
               100  # HIGH
        ]
        rad_labels = ['Inner Belt Flux', 'Outer Belt Flux', 'SAA Intensity']
        
        axs[1, 0].bar(rad_labels, rad_values, color=['salmon', 'salmon', 'salmon'])
        axs[1, 0].set_title('Radiation Levels')
        axs[1, 0].set_ylabel('Intensity Level')
        axs[1, 0].set_ylim(0, 110)
        axs[1, 0].tick_params(axis='x', rotation=45)
        
        # 4. Plot operational impacts (bottom right)
        op_data = weather_data.get('operational_impact', {})
        impact_values = []
        impact_labels = []
        
        for impact, level in op_data.items():
            impact_labels.append(impact.replace('_', ' ').title())
            impact_values.append(
                25 if level == 'MINIMAL' or level == 'NONE' else
                50 if level == 'LOW' else
                75 if level == 'NORMAL' or level == 'MODERATE' else
                100  # HIGH
            )
        
        # If empty, provide defaults
        if not impact_values:
            impact_labels = ['Satellite Charging', 'Single Event Upsets', 'Drag Effects', 'Communications']
            impact_values = [25, 50, 25, 25]  # Default values
            
        colors = ['green' if v < 40 else 'yellow' if v < 70 else 'orange' if v < 90 else 'red' for v in impact_values]
        
        axs[1, 1].bar(impact_labels, impact_values, color=colors)
        axs[1, 1].set_title('Operational Impacts')
        axs[1, 1].set_ylabel('Impact Level')
        axs[1, 1].set_ylim(0, 110)
        axs[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        fig.subplots_adjust(top=0.9)
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_3d_orbit_visualization(self, object_id: str, state_vectors: List[Dict[str, Any]], 
                                    conjunctions: Optional[List[Dict[str, Any]]] = None) -> go.Figure:
        """Create an interactive 3D visualization of an orbit with optional conjunction points.
        
        Args:
            object_id: ID of the primary object
            state_vectors: List of state vectors for the object
            conjunctions: Optional list of conjunction data to highlight on the orbit
            
        Returns:
            Plotly Figure object
        """
        # Extract position data from state vectors
        positions = []
        timestamps = []
        
        # In a real implementation, this would process actual state vector data
        # For demonstration, we'll generate a simple circular orbit
        
        # Generate a simple orbit
        theta = np.linspace(0, 2*np.pi, 100)
        orbit_radius = 7000  # km (typical LEO orbit)
        
        # Slightly inclined circular orbit
        x = orbit_radius * np.cos(theta)
        y = orbit_radius * np.sin(theta)
        z = orbit_radius * 0.1 * np.sin(theta)  # Small inclination
        
        # Create figure
        fig = go.Figure()
        
        # Add orbit trace
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color='blue', width=4),
            name=f"Object {object_id} Orbit"
        ))
        
        # Add Earth
        earth_radius = 6371  # km
        
        # Create sphere
        phi, theta = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x_earth = earth_radius * np.sin(theta) * np.cos(phi)
        y_earth = earth_radius * np.sin(theta) * np.sin(phi)
        z_earth = earth_radius * np.cos(theta)
        
        fig.add_trace(go.Surface(
            x=x_earth, y=y_earth, z=z_earth,
            colorscale='Blues',
            opacity=0.6,
            name="Earth"
        ))
        
        # Add conjunction points if provided
        if conjunctions:
            # In a real implementation, these would be accurate conjunction points
            # For demonstration, we'll place them at random points on the orbit
            
            conj_x = []
            conj_y = []
            conj_z = []
            conj_colors = []
            conj_sizes = []
            conj_texts = []
            
            # Use a subset of orbit points for conjunctions
            np.random.seed(42)  # For reproducibility
            indices = np.random.choice(len(x), min(len(conjunctions), 5), replace=False)
            
            for i, conj in enumerate(conjunctions[:5]):  # Limit to 5 for demo
                idx = indices[i % len(indices)]
                
                conj_x.append(x[idx])
                conj_y.append(y[idx])
                conj_z.append(z[idx])
                
                risk_level = conj.get('risk_level', 'LOW')
                conj_colors.append(self.risk_colors.get(risk_level, '#AAAAAA'))
                
                # Size inversely proportional to miss distance
                miss_distance = conj.get('miss_distance', 10)
                conj_sizes.append(max(8, min(20, 100/miss_distance)))
                
                # Create hover text
                secondary = conj.get('secondary_object', {})
                secondary_name = secondary.get('name', 'Unknown object')
                tca = conj.get('time_of_closest_approach', 'Unknown time')
                
                conj_texts.append(
                    f"Object: {secondary_name}<br>"
                    f"Time: {tca}<br>"
                    f"Miss Distance: {miss_distance:.2f} km<br>"
                    f"Risk: {risk_level}"
                )
            
            fig.add_trace(go.Scatter3d(
                x=conj_x, y=conj_y, z=conj_z,
                mode='markers',
                marker=dict(
                    size=conj_sizes,
                    color=conj_colors,
                    line=dict(width=1, color='white')
                ),
                text=conj_texts,
                hoverinfo='text',
                name="Conjunctions"
            ))
        
        # Add current position (last point in orbit)
        fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[z[-1]],
            mode='markers',
            marker=dict(
                size=12,
                color='red',
                symbol='circle',
                line=dict(width=1, color='white')
            ),
            name=f"Current Position"
        ))
        
        # Layout
        fig.update_layout(
            title=f"3D Orbit Visualization for Object {object_id}",
            scene=dict(
                xaxis_title="X (km)",
                yaxis_title="Y (km)",
                zaxis_title="Z (km)",
                aspectmode='data'
            ),
            legend=dict(x=0, y=0),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        
        return fig
    
    def plot_anomaly_timeline(self, anomaly_data: Dict[str, Any], 
                           save_path: Optional[str] = None) -> plt.Figure:
        """Plot detected anomalies on a timeline.
        
        Args:
            anomaly_data: Anomaly detection results from UDLDataProcessor
            save_path: Optional path to save the plot image
            
        Returns:
            Matplotlib Figure object
        """
        anomalies = anomaly_data.get('anomalies', [])
        
        if not anomalies:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "No anomalies detected", 
                   horizontalalignment='center', verticalalignment='center',
                   transform=ax.transAxes)
            return fig
            
        # Parse anomaly data
        dates = []
        types = []
        severities = []
        details = []
        
        severity_colors = {
            "CRITICAL": "red",
            "HIGH": "orange",
            "MEDIUM": "yellow",
            "LOW": "green"
        }
        
        for anomaly in anomalies:
            # Convert time string to datetime
            time_str = anomaly.get('detection_time', '').replace('Z', '+00:00')
            try:
                detection_time = datetime.fromisoformat(time_str)
                dates.append(detection_time)
                types.append(anomaly.get('type', 'Unknown'))
                severities.append(anomaly.get('severity', 'LOW'))
                details.append(anomaly.get('details', 'No details'))
            except (ValueError, TypeError):
                logger.warning(f"Invalid date format: {anomaly.get('detection_time', '')}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Create a horizontal timeline
        y_positions = range(len(dates))
        colors = [severity_colors.get(severity, 'blue') for severity in severities]
        
        # Sort by date
        if dates:
            date_indices = np.argsort(dates)
            sorted_dates = [dates[i] for i in date_indices]
            sorted_types = [types[i] for i in date_indices]
            sorted_colors = [colors[i] for i in date_indices]
            sorted_details = [details[i] for i in date_indices]
            
            # Plot timeline
            markerline, stemline, baseline = ax.stem(
                sorted_dates, 
                [1] * len(sorted_dates),
                linefmt='grey',
                basefmt=' '
            )
            
            plt.setp(markerline, markersize=15, markerfacecolor=sorted_colors, markeredgecolor='black')
            
            # Add labels
            for i, (date, anomaly_type, detail) in enumerate(zip(sorted_dates, sorted_types, sorted_details)):
                ax.annotate(
                    anomaly_type.replace('_', ' ').title(),
                    xy=(date, 1),
                    xytext=(0, 10),  # 10 points above
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    fontweight='bold'
                )
                
                ax.annotate(
                    detail,
                    xy=(date, 1),
                    xytext=(0, -25),  # 25 points below
                    textcoords='offset points',
                    ha='center',
                    va='top',
                    fontsize=8,
                    wrap=True
                )
            
            # Format the plot
            time_range = anomaly_data.get('time_range', {})
            start_time = datetime.fromisoformat(time_range.get('start', (datetime.utcnow() - timedelta(days=30)).isoformat()))
            end_time = datetime.fromisoformat(time_range.get('end', datetime.utcnow().isoformat()))
            
            # Set x-axis limits
            buffer = timedelta(days=(end_time - start_time).days * 0.05)  # 5% buffer on each side
            ax.set_xlim(start_time - buffer, end_time + buffer)
            
            # Hide y-axis
            ax.get_yaxis().set_visible(False)
            
            # Format date on x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, (end_time - start_time).days // 10)))
            plt.xticks(rotation=45)
            
            # Add title and grid
            ax.set_title(f"Anomaly Timeline for Object {anomaly_data.get('object_id', 'Unknown')}")
            ax.grid(True, axis='x', alpha=0.3)
            
            # Add legend for severity levels
            from matplotlib.lines import Line2D
            unique_severities = set(severities)
            legend_elements = [Line2D([0], [0], marker='o', color='w', label=severity,
                                     markerfacecolor=severity_colors[severity], markersize=10)
                               for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW"] 
                               if severity in unique_severities]
            
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            # Save if path provided
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                
        return fig
    
    def generate_data_dashboard(self, object_id: str, days: int = 7,
                              save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """Generate a complete data dashboard for a space object.
        
        Args:
            object_id: ID of the space object
            days: Number of days of historical data to analyze
            save_dir: Optional directory to save plot images
            
        Returns:
            Dictionary of generated figures
        """
        if self.data_processor is None:
            raise ValueError("Data processor is required for dashboard generation")
            
        figures = {}
        
        try:
            # Process orbital data
            orbital_data = self.data_processor.process_orbital_data(object_id, days)
            if orbital_data:
                save_path = f"{save_dir}/orbital_altitude.png" if save_dir else None
                figures['orbital_altitude'] = self.plot_orbital_altitude(orbital_data, save_path)
            
            # Analyze conjunctions
            conjunctions = self.data_processor.analyze_conjunction_risk(object_id, days_ahead=days)
            if conjunctions:
                save_path = f"{save_dir}/conjunction_timeline.png" if save_dir else None
                figures['conjunction_timeline'] = self.plot_conjunction_risk_timeline(conjunctions, save_path)
                
                # Create 3D visualization
                figures['orbit_3d'] = self.create_3d_orbit_visualization(object_id, [], conjunctions)
            
            # Analyze space weather
            weather_data = self.data_processor.analyze_space_weather_impact([object_id])
            if weather_data:
                save_path = f"{save_dir}/space_weather.png" if save_dir else None
                figures['space_weather'] = self.plot_space_weather_conditions(weather_data, save_path)
            
            # Detect anomalies
            anomalies = self.data_processor.detect_anomalies(object_id, days=days)
            if anomalies:
                save_path = f"{save_dir}/anomaly_timeline.png" if save_dir else None
                figures['anomaly_timeline'] = self.plot_anomaly_timeline(anomalies, save_path)
                
        except Exception as e:
            logger.error(f"Error generating dashboard: {str(e)}")
            
        return figures 
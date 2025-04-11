#!/usr/bin/env python
"""UDL Visualization Example

This script demonstrates how to use the UDLVisualizer to create
visualizations of UDL data for space objects.

Usage:
    python udl_visualization_example.py --object-id <object_id> [--days <days>] [--output-dir <dir>]
"""

import argparse
import logging
import os
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

from asttroshield.api_client.udl_client import UDLClient
from asttroshield.udl_integration import USSFDULIntegrator
from asttroshield.udl_data_processor import UDLDataProcessor
from asttroshield.udl_visualization import UDLVisualizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_visualization_pipeline(api_key=None, api_url=None):
    """Set up the UDL visualization pipeline."""
    # Initialize the UDL client
    udl_client = UDLClient(api_key=api_key, api_url=api_url)
    
    # Initialize the UDL integrator
    udl_integrator = USSFDULIntegrator(udl_client=udl_client)
    
    # Initialize the UDL data processor
    processor = UDLDataProcessor(
        udl_client=udl_client,
        udl_integrator=udl_integrator
    )
    
    # Initialize the UDL visualizer
    visualizer = UDLVisualizer(data_processor=processor)
    
    return visualizer, processor, udl_client

def create_orbital_visualization(visualizer, processor, object_id, days=7, output_dir=None):
    """Create orbital visualizations for a space object."""
    logger.info(f"Creating orbital visualizations for object {object_id}")
    
    try:
        # Process orbital data
        orbital_data = processor.process_orbital_data(object_id, days)
        
        if not orbital_data:
            logger.error("Failed to retrieve orbital data")
            return
            
        # Create save path if output directory is provided
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"orbital_altitude_{object_id}.png")
            
        # Plot orbital altitude
        fig = visualizer.plot_orbital_altitude(orbital_data, save_path)
        
        if not output_dir:
            plt.show()
            
        logger.info("Orbital altitude visualization completed")
        
    except Exception as e:
        logger.error(f"Error creating orbital visualization: {str(e)}")

def create_conjunction_visualization(visualizer, processor, object_id, days=7, output_dir=None):
    """Create conjunction risk visualizations for a space object."""
    logger.info(f"Creating conjunction visualizations for object {object_id}")
    
    try:
        # Analyze conjunction risks
        conjunctions = processor.analyze_conjunction_risk(object_id, days)
        
        if not conjunctions:
            logger.error("Failed to retrieve conjunction data")
            return
            
        # Create save paths if output directory is provided
        timeline_save_path = None
        orbit_save_path = None
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            timeline_save_path = os.path.join(output_dir, f"conjunction_timeline_{object_id}.png")
            orbit_save_path = os.path.join(output_dir, f"orbit_3d_{object_id}.html")
            
        # Plot conjunction risk timeline
        fig_timeline = visualizer.plot_conjunction_risk_timeline(conjunctions, timeline_save_path)
        
        # Create 3D orbit visualization with conjunctions
        fig_3d = visualizer.create_3d_orbit_visualization(object_id, [], conjunctions)
        
        # Show or save the plots
        if not output_dir:
            plt.show()
            fig_3d.show()
        else:
            # Save 3D plot as HTML
            fig_3d.write_html(orbit_save_path)
            logger.info(f"3D visualization saved to {orbit_save_path}")
            
        logger.info("Conjunction visualizations completed")
        
    except Exception as e:
        logger.error(f"Error creating conjunction visualization: {str(e)}")

def create_space_weather_visualization(visualizer, processor, object_id, output_dir=None):
    """Create space weather visualizations."""
    logger.info("Creating space weather visualization")
    
    try:
        # Analyze space weather impacts
        weather_data = processor.analyze_space_weather_impact([object_id])
        
        if not weather_data:
            logger.error("Failed to retrieve space weather data")
            return
            
        # Create save path if output directory is provided
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"space_weather_{object_id}.png")
            
        # Plot space weather conditions
        fig = visualizer.plot_space_weather_conditions(weather_data, save_path)
        
        if not output_dir:
            plt.show()
            
        logger.info("Space weather visualization completed")
        
    except Exception as e:
        logger.error(f"Error creating space weather visualization: {str(e)}")

def create_anomaly_visualization(visualizer, processor, object_id, days=30, output_dir=None):
    """Create anomaly timeline visualization for a space object."""
    logger.info(f"Creating anomaly visualization for object {object_id}")
    
    try:
        # Detect anomalies
        anomalies = processor.detect_anomalies(object_id, days)
        
        if not anomalies:
            logger.error("Failed to retrieve anomaly data")
            return
            
        # Create save path if output directory is provided
        save_path = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f"anomaly_timeline_{object_id}.png")
            
        # Plot anomaly timeline
        fig = visualizer.plot_anomaly_timeline(anomalies, save_path)
        
        if not output_dir:
            plt.show()
            
        logger.info("Anomaly visualization completed")
        
    except Exception as e:
        logger.error(f"Error creating anomaly visualization: {str(e)}")

def create_complete_dashboard(visualizer, object_id, days=7, output_dir=None):
    """Create a complete dashboard of visualizations for a space object."""
    logger.info(f"Creating complete dashboard for object {object_id}")
    
    try:
        # Generate the dashboard
        dashboard_dir = None
        if output_dir:
            dashboard_dir = os.path.join(output_dir, f"dashboard_{object_id}")
            os.makedirs(dashboard_dir, exist_ok=True)
            
        figures = visualizer.generate_data_dashboard(object_id, days, dashboard_dir)
        
        if not output_dir:
            # Display figures interactively
            for name, fig in figures.items():
                if 'orbit_3d' in name:
                    fig.show()
                else:
                    plt.figure(fig.number)
                    plt.show()
                    
        logger.info(f"Complete dashboard created with {len(figures)} visualizations")
        
        # Return the number of created visualizations
        return len(figures)
        
    except Exception as e:
        logger.error(f"Error creating dashboard: {str(e)}")
        return 0

def main():
    """Main function to run the example."""
    parser = argparse.ArgumentParser(description="UDL Visualization Example")
    parser.add_argument("--object-id", required=True, help="ID of the space object to visualize")
    parser.add_argument("--days", type=int, default=7, help="Number of days of data to visualize")
    parser.add_argument("--output-dir", help="Output directory for visualizations")
    parser.add_argument("--api-key", help="UDL API key (optional)")
    parser.add_argument("--api-url", help="UDL API URL (optional)")
    parser.add_argument("--viz-type", choices=["orbital", "conjunction", "weather", "anomaly", "dashboard", "all"], 
                       default="dashboard", help="Type of visualization to create")
    
    args = parser.parse_args()
    
    # Setup visualization pipeline
    visualizer, processor, _ = setup_visualization_pipeline(api_key=args.api_key, api_url=args.api_url)
    
    # Create requested visualizations
    if args.viz_type == "orbital" or args.viz_type == "all":
        create_orbital_visualization(visualizer, processor, args.object_id, args.days, args.output_dir)
        
    if args.viz_type == "conjunction" or args.viz_type == "all":
        create_conjunction_visualization(visualizer, processor, args.object_id, args.days, args.output_dir)
        
    if args.viz_type == "weather" or args.viz_type == "all":
        create_space_weather_visualization(visualizer, processor, args.object_id, args.output_dir)
        
    if args.viz_type == "anomaly" or args.viz_type == "all":
        create_anomaly_visualization(visualizer, processor, args.object_id, args.days, args.output_dir)
        
    if args.viz_type == "dashboard" or args.viz_type == "all":
        num_viz = create_complete_dashboard(visualizer, args.object_id, args.days, args.output_dir)
        print(f"\nCreated dashboard with {num_viz} visualizations")
    
    # Provide a summary
    if args.output_dir:
        print(f"\nAll visualizations saved to: {args.output_dir}")
        print("Use a web browser to view the HTML files for interactive 3D visualizations")

if __name__ == "__main__":
    main() 
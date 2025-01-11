import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from datetime import datetime, timedelta

from ..data_generation.physical_data_gen import PhysicalPropertiesGenerator
from ..data_generation.proximity_data_gen import ProximityDataGenerator
from ..data_generation.remote_sensing_data_gen import RemoteSensingDataGenerator
from ..data_generation.eclipse_data_gen import EclipseDataGenerator
from ..data_generation.track_data_gen import TrackDataGenerator

def plot_physical_properties():
    """Visualize physical properties data"""
    generator = PhysicalPropertiesGenerator()
    
    # Generate sequence data
    records, anomalies = generator.generate_property_sequence(
        'active_satellite',
        duration_hours=24
    )
    
    # Extract time series data
    times = range(len(records))
    temperatures = [r.get('temperature', r.get('thermal', {}).get('temperature', 293.15)) for r in records]
    amrs = [r.get('amr', 0.0) for r in records]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot temperature
    ax1.plot(times, temperatures, label='Temperature (K)')
    ax1.set_title('Satellite Temperature Over Time')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (K)')
    ax1.grid(True)
    
    # Plot AMR
    ax2.plot(times, amrs, label='Area-to-Mass Ratio', color='orange')
    ax2.set_title('Area-to-Mass Ratio Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('AMR (m²/kg)')
    ax2.grid(True)
    
    # Mark anomalies
    for anomaly in anomalies:
        timestamp = anomaly.get('timestamp', 0)
        ax1.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
        ax2.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('physical_properties.png')
    plt.close()

def plot_proximity_operations():
    """Visualize proximity operations data"""
    generator = ProximityDataGenerator()
    
    # Generate conjunction data
    records, anomalies = generator.generate_conjunction_sequence(
        duration_hours=6
    )
    
    # Extract time series data
    times = range(len(records))
    ranges = [r.get('range_km', 1000.0) for r in records]
    probabilities = [r.get('collision_probability', 0.0) for r in records]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot range
    ax1.plot(times, ranges, label='Range')
    ax1.set_title('Range Between Objects')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Range (km)')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Plot collision probability
    ax2.plot(times, probabilities, label='Collision Probability', color='orange')
    ax2.set_title('Collision Probability Over Time')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('Probability')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    # Mark anomalies
    for anomaly in anomalies:
        timestamp = anomaly.get('timestamp', 0)
        ax1.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
        ax2.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('proximity_operations.png')
    plt.close()

def plot_remote_sensing():
    """Visualize remote sensing data"""
    generator = RemoteSensingDataGenerator()
    
    # Generate observation data
    records, anomalies = generator.generate_observation_sequence(
        duration_hours=12
    )
    
    # Extract time series data
    times = range(len(records))
    optical_snr = [r.get('measurements', {}).get('optical_snr', 0.0) for r in records]
    radar_snr = [r.get('measurements', {}).get('radar_snr', 0.0) for r in records]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot optical SNR
    ax1.plot(times, optical_snr, label='Optical SNR')
    ax1.set_title('Optical Signal-to-Noise Ratio')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('SNR')
    ax1.set_yscale('log')
    ax1.grid(True)
    
    # Plot radar SNR
    ax2.plot(times, radar_snr, label='Radar SNR', color='orange')
    ax2.set_title('Radar Signal-to-Noise Ratio')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('SNR')
    ax2.set_yscale('log')
    ax2.grid(True)
    
    # Mark anomalies
    for anomaly in anomalies:
        timestamp = anomaly.get('timestamp', 0)
        ax1.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
        ax2.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('remote_sensing.png')
    plt.close()

def plot_eclipse_data():
    """Visualize eclipse data"""
    generator = EclipseDataGenerator()
    
    # Generate eclipse data
    records, anomalies = generator.generate_eclipse_sequence(
        duration_hours=24
    )
    
    # Extract time series data
    times = range(len(records))
    temperatures = [r.get('thermal', {}).get('temperature', 293.15) for r in records]
    battery_states = [r.get('power', {}).get('battery_state', 1.0) for r in records]
    eclipse_types = [r.get('eclipse_type', 'sunlight') for r in records]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot temperature
    ax1.plot(times, temperatures, label='Temperature')
    ax1.set_title('Satellite Temperature During Eclipse')
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Temperature (K)')
    ax1.grid(True)
    
    # Plot battery state
    ax2.plot(times, battery_states, label='Battery State', color='orange')
    ax2.set_title('Battery State During Eclipse')
    ax2.set_xlabel('Time Step')
    ax2.set_ylabel('State of Charge')
    ax2.grid(True)
    
    # Shade eclipse periods
    for i, eclipse_type in enumerate(eclipse_types):
        if eclipse_type == 'umbra':
            ax1.axvspan(i, i+1, color='gray', alpha=0.2)
            ax2.axvspan(i, i+1, color='gray', alpha=0.2)
        elif eclipse_type == 'penumbra':
            ax1.axvspan(i, i+1, color='gray', alpha=0.1)
            ax2.axvspan(i, i+1, color='gray', alpha=0.1)
    
    # Mark anomalies
    for anomaly in anomalies:
        timestamp = anomaly.get('timestamp', 0)
        ax1.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
        ax2.axvline(x=timestamp, color='r', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('eclipse_data.png')
    plt.close()

def plot_track_data():
    """Visualize track data"""
    generator = TrackDataGenerator()
    
    # Generate track data
    records, anomalies = generator.generate_track_sequence(
        duration_hours=6
    )
    
    # Create 3D figure
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each track
    colors = plt.cm.rainbow(np.linspace(0, 1, len(records)))
    for record, color in zip(records, colors):
        states = record.get('states', [])
        if states:
            positions = np.array([state.get('position', [0, 0, 0]) for state in states])
            
            ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                    color=color, label=record.get('track_type', 'unknown'))
    
    # Plot Earth (simplified)
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    earth_radius = 6378.137
    x = earth_radius * np.outer(np.cos(u), np.sin(v))
    y = earth_radius * np.outer(np.sin(u), np.sin(v))
    z = earth_radius * np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_surface(x, y, z, color='b', alpha=0.1)
    
    ax.set_title('Space Object Tracks')
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    ax.set_zlabel('Z (km)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('track_data.png')
    plt.close()

def plot_training_data_distributions():
    """Visualize training data distributions"""
    # Generate training data from each model
    physical_gen = PhysicalPropertiesGenerator()
    proximity_gen = ProximityDataGenerator()
    remote_sensing_gen = RemoteSensingDataGenerator()
    eclipse_gen = EclipseDataGenerator()
    track_gen = TrackDataGenerator()
    
    # Generate samples
    X_physical, y_physical = physical_gen.generate_training_data(num_samples=100)
    X_proximity, y_proximity = proximity_gen.generate_training_data(num_samples=100)
    X_remote, y_remote = remote_sensing_gen.generate_training_data(num_samples=100)
    X_eclipse, y_eclipse = eclipse_gen.generate_training_data(num_samples=100)
    X_track, y_track = track_gen.generate_training_data(num_samples=100)
    
    # Create figure
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    
    # Plot label distributions for each model
    sns.boxplot(data=y_physical, ax=axes[0])
    axes[0].set_title('Physical Properties Labels')
    axes[0].set_xticklabels(['AMR', 'Attitude', 'Thermal', 'Structural', 'Confidence'])
    
    sns.boxplot(data=y_proximity, ax=axes[1])
    axes[1].set_title('Proximity Operations Labels')
    axes[1].set_xticklabels(['Emergency', 'Critical', 'Warning', 'Confidence'])
    
    sns.boxplot(data=y_remote, ax=axes[2])
    axes[2].set_title('Remote Sensing Labels')
    axes[2].set_xticklabels(['Optical', 'Radar', 'Atmospheric', 'Target', 'Confidence'])
    
    sns.boxplot(data=y_eclipse, ax=axes[3])
    axes[3].set_title('Eclipse Labels')
    axes[3].set_xticklabels(['Thermal', 'Power', 'Eclipse', 'Sensor', 'Confidence'])
    
    sns.boxplot(data=y_track, ax=axes[4])
    axes[4].set_title('Track Labels')
    axes[4].set_xticklabels(['Maneuver', 'Breakup', 'Measurement', 'Misid', 'Confidence'])
    
    plt.tight_layout()
    plt.savefig('training_distributions.png')
    plt.close()

if __name__ == '__main__':
    # Generate all visualizations
    plot_physical_properties()
    plot_proximity_operations()
    plot_remote_sensing()
    plot_eclipse_data()
    plot_track_data()
    plot_training_data_distributions() 
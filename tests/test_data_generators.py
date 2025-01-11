import os
import sys
import numpy as np
import datetime

# Add parent directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from ml.data_generation.stability_data_gen import StabilityDataGenerator
from ml.data_generation.environmental_data_gen import EnvironmentalDataGenerator

def test_stability_data():
    """Test if stability data generator produces physically valid orbits"""
    generator = StabilityDataGenerator()
    
    # Generate a small sample
    print("Generating stability data...")
    X, y = generator.generate_training_data(num_samples=2, sequence_length=30)  # Reduced samples and sequence length
    
    # Check orbit properties
    for i in range(len(X)):
        orbit = X[i]
        
        # Calculate orbital elements
        positions = orbit[:, :3]
        velocities = orbit[:, 3:]
        
        # Check energy conservation
        r = np.sqrt(np.sum(positions**2, axis=1))
        v = np.sqrt(np.sum(velocities**2, axis=1))
        energy = 0.5 * v**2 - generator.G * generator.M / r
        energy_variation = np.std(energy) / np.abs(np.mean(energy))
        
        # Calculate eccentricity
        r_min = np.min(r)
        r_max = np.max(r)
        ecc = (r_max - r_min) / (r_max + r_min)
        
        print(f"\nOrbit {i}:")
        print(f"Energy variation: {energy_variation:.2e}")
        print(f"Min radius: {(np.min(r) - generator.R)/1000:.0f} km")
        print(f"Max radius: {(np.max(r) - generator.R)/1000:.0f} km")
        print(f"Eccentricity: {ecc:.3f}")
        print(f"Stability score: {y[i][0]:.3f}")
        
        # Verify physics
        if energy_variation > 1e-4:
            print("WARNING: Energy not well conserved")
        if ecc > 0.1:
            print("WARNING: Orbit more eccentric than expected")
        if np.min(r) < generator.R:
            print("WARNING: Orbit intersects Earth")

def test_environmental_data():
    """Test if environmental data generator produces correct radiation belt profiles"""
    generator = EnvironmentalDataGenerator()
    earth_radius = 6371.0  # Earth radius in km
    
    # Test specific L-shell values
    L_values = [1.3, 2.5, 5.0]  # Inner belt, slot region, outer belt
    print("\nTesting specific L-shell values...")
    
    for L in L_values:
        # Calculate position at this L-shell
        altitude = (L - 1) * earth_radius  # Convert L-shell to altitude in km
        
        # Calculate Van Allen belt parameters directly
        va_params = generator._van_allen_belts(L, generator.B0 * (earth_radius * 1000 / (L * earth_radius * 1000))**3, 0.5)
        
        # Extract radiation intensities
        inner_flux = va_params['inner_flux'] / generator.inner_belt['max_flux_p']
        outer_flux = va_params['outer_flux'] / generator.outer_belt['max_flux_e']
        total_dose = va_params['total_dose'] / (generator.inner_belt['max_flux_p'] + generator.outer_belt['max_flux_e'])
        belt_region = va_params['belt_region']
        
        print(f"\nL-shell = {L:.1f} (altitude = {altitude:.0f} km):")
        print(f"Inner belt intensity: {inner_flux:.3f}")
        print(f"Outer belt intensity: {outer_flux:.3f}")
        print(f"Total dose: {total_dose:.3f}")
        print(f"Belt region: {belt_region:.1f}")
        
        # Verify physics
        if L == 1.3:  # Inner belt
            if inner_flux < 0.5:
                print("WARNING: Inner belt intensity too low")
            if outer_flux > 0.1:
                print("WARNING: Unexpected outer belt contribution")
        elif L == 2.5:  # Slot region
            if inner_flux > 0.2 or outer_flux > 0.2:
                print("WARNING: Radiation too high in slot region")
        elif L == 5.0:  # Outer belt
            if outer_flux < 0.3:
                print("WARNING: Outer belt intensity too low")
            if inner_flux > 0.1:
                print("WARNING: Unexpected inner belt contribution")

def main():
    """Run data generator tests"""
    try:
        print("\nTesting Stability Data Generator...")
        print("=" * 50)
        test_stability_data()
    except Exception as e:
        print(f"Error in stability test: {str(e)}")
    
    try:
        print("\nTesting Environmental Data Generator...")
        print("=" * 50)
        test_environmental_data()
    except Exception as e:
        print(f"Error in environmental test: {str(e)}")

if __name__ == '__main__':
    main()

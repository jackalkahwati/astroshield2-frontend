import unittest
import sys
import os
from .test_data_generators import TestDataGenerators
from . import visualize_data

def run_tests():
    """Run all unit tests"""
    print("Running unit tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(TestDataGenerators)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    return result.wasSuccessful()

def generate_visualizations():
    """Generate all visualizations"""
    print("\nGenerating visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs('validation_results', exist_ok=True)
    
    # Change to output directory
    original_dir = os.getcwd()
    os.chdir('validation_results')
    
    try:
        # Generate visualizations
        visualize_data.plot_physical_properties()
        print("- Generated physical properties visualization")
        
        visualize_data.plot_proximity_operations()
        print("- Generated proximity operations visualization")
        
        visualize_data.plot_remote_sensing()
        print("- Generated remote sensing visualization")
        
        visualize_data.plot_eclipse_data()
        print("- Generated eclipse data visualization")
        
        visualize_data.plot_track_data()
        print("- Generated track data visualization")
        
        visualize_data.plot_training_data_distributions()
        print("- Generated training data distributions visualization")
        
    finally:
        # Restore original directory
        os.chdir(original_dir)

def main():
    """Run full validation suite"""
    print("Starting model validation...\n")
    
    # Run tests
    tests_passed = run_tests()
    
    if not tests_passed:
        print("\nTests failed! Stopping validation.")
        return 1
    
    # Generate visualizations
    generate_visualizations()
    
    print("\nValidation complete! Check validation_results/ directory for visualization plots.")
    return 0

if __name__ == '__main__':
    sys.exit(main()) 
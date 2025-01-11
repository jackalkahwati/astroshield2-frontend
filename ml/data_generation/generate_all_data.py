import numpy as np
from ml.data_generation.stability_data_gen import StabilityDataGenerator
from ml.data_generation.maneuver_data_gen import ManeuverDataGenerator
from ml.data_generation.environmental_data_gen import EnvironmentalDataGenerator
from ml.data_generation.physical_data_gen import PhysicalDataGenerator
from ml.data_generation.launch_data_gen import LaunchDataGenerator
from ml.data_generation.compliance_data_gen import ComplianceDataGenerator
import os
import datetime
import traceback

class DataGenerationManager:
    """Manage synthetic data generation for all evaluators"""
    
    def __init__(self, output_dir: str = 'synthetic_data'):
        self.output_dir = output_dir
        self.generators = {
            'stability': StabilityDataGenerator(),
            'maneuver': ManeuverDataGenerator(),
            'environmental': EnvironmentalDataGenerator(),
            'physical': PhysicalDataGenerator(),
            'launch': LaunchDataGenerator(),
            'compliance': ComplianceDataGenerator()
        }
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

    def generate_all_data(
        self,
        num_samples: int = 50,
        validate: bool = True
    ) -> dict:
        """Generate synthetic data for all evaluators"""
        results = {}
        
        for name, generator in self.generators.items():
            print(f"\nGenerating {name} data...")
            try:
                # Generate data
                if name == 'maneuver':
                    print(f"Generating {num_samples} maneuver sequences...")
                    X, A, R = generator.generate_training_data(num_samples=num_samples)
                    data = {'X': X, 'A': A, 'R': R}
                else:
                    print(f"Generating {num_samples} {name} sequences...")
                    X, y = generator.generate_training_data(num_samples=num_samples)
                    data = {'X': X, 'y': y}
                
                print(f"Generated data shapes: X={X.shape}")
                
                # Validate data if requested
                if validate:
                    print("Validating sequences...")
                    valid_count = 0
                    for i in range(len(X)):
                        try:
                            if name == 'maneuver':
                                is_valid = generator.validate_physics(X[i], A[i])
                            elif name == 'compliance':
                                is_valid = generator.validate_data(X[i, 0])
                            else:
                                is_valid = generator.validate_physics(X[i])
                            
                            if is_valid:
                                valid_count += 1
                        except Exception as e:
                            print(f"Validation error in {name} sample {i}:")
                            print(traceback.format_exc())
                            continue
                    
                    validation_rate = valid_count / len(X)
                    print(f"Validation rate: {validation_rate:.2%}")
                    
                    if validation_rate < 0.95:
                        print(f"Warning: Low validation rate for {name} data")
                
                # Save data
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(self.output_dir, f"{name}_data_{timestamp}.npz")
                np.savez(save_path, **data)
                print(f"Data saved to {save_path}")
                
                # Store results
                results[name] = {
                    'data': data,
                    'path': save_path,
                    'validation_rate': validation_rate if validate else None
                }
                
            except Exception as e:
                print(f"Error generating {name} data:")
                print(traceback.format_exc())
                continue
        
        return results

    def analyze_data_statistics(self, results: dict) -> dict:
        """Analyze statistics of generated data"""
        stats = {}
        
        for name, result in results.items():
            try:
                data = result['data']
                X = data['X']
                
                # Handle NaN and Inf values
                X_clean = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10)
                
                stats[name] = {
                    'shape': X.shape,
                    'mean': float(np.mean(X_clean)),
                    'std': float(np.std(X_clean)),
                    'min': float(np.min(X_clean)),
                    'max': float(np.max(X_clean)),
                    'nan_count': int(np.isnan(X).sum()),
                    'inf_count': int(np.isinf(X).sum())
                }
                
                if 'y' in data:
                    y = data['y']
                    y_clean = np.nan_to_num(y, nan=0.0, posinf=1e10, neginf=-1e10)
                    stats[name].update({
                        'y_shape': y.shape,
                        'y_mean': float(np.mean(y_clean)),
                        'y_std': float(np.std(y_clean)),
                        'y_min': float(np.min(y_clean)),
                        'y_max': float(np.max(y_clean))
                    })
                elif 'R' in data:  # Maneuver data
                    R = data['R']
                    R_clean = np.nan_to_num(R, nan=0.0, posinf=1e10, neginf=-1e10)
                    stats[name].update({
                        'R_shape': R.shape,
                        'R_mean': float(np.mean(R_clean)),
                        'R_std': float(np.std(R_clean)),
                        'R_min': float(np.min(R_clean)),
                        'R_max': float(np.max(R_clean))
                    })
            except Exception as e:
                print(f"Error analyzing {name} data:")
                print(traceback.format_exc())
                continue
        
        return stats

    def print_data_summary(self, results: dict, stats: dict):
        """Print summary of generated data"""
        print("\nData Generation Summary")
        print("=" * 50)
        
        for name in results.keys():
            try:
                print(f"\n{name.upper()} Data:")
                print("-" * 30)
                
                # Data shapes
                print(f"Input shape: {stats[name]['shape']}")
                if 'y_shape' in stats[name]:
                    print(f"Output shape: {stats[name]['y_shape']}")
                elif 'R_shape' in stats[name]:
                    print(f"Reward shape: {stats[name]['R_shape']}")
                
                # Basic statistics
                print(f"Mean: {stats[name]['mean']:.3f}")
                print(f"Std: {stats[name]['std']:.3f}")
                print(f"Range: [{stats[name]['min']:.3f}, {stats[name]['max']:.3f}]")
                
                # Data quality
                print(f"NaN count: {stats[name]['nan_count']}")
                print(f"Inf count: {stats[name]['inf_count']}")
                
                # Validation rate
                if results[name]['validation_rate'] is not None:
                    print(f"Validation rate: {results[name]['validation_rate']:.2%}")
                
                # File location
                print(f"Saved to: {results[name]['path']}")
                
            except Exception as e:
                print(f"Error printing summary for {name}:")
                print(traceback.format_exc())
                continue

if __name__ == '__main__':
    # Generate synthetic data
    manager = DataGenerationManager()
    
    print("Starting synthetic data generation...")
    results = manager.generate_all_data(num_samples=50, validate=True)
    
    # Analyze and print summary
    stats = manager.analyze_data_statistics(results)
    manager.print_data_summary(results, stats)

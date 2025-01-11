import torch
from threat_detector import ThreatDetector
from rl_maneuver import RLManeuverPlanner
from game_theory import GameTheoryDeception
from compliance_evaluator import ComplianceEvaluator
from analyst_evaluator import AnalystEvaluator
from stimulation_evaluator import StimulationEvaluator
from launch_evaluator import LaunchEvaluator
from environmental_evaluator import EnvironmentalEvaluator
import os

def export_analysis_models(export_dir: str = 'models'):
    """Export all analysis models to ONNX format"""
    os.makedirs(export_dir, exist_ok=True)
    
    print("Starting export of analysis models...")
    
    # Export ThreatDetector
    print("\nExporting ThreatDetector model...")
    threat_detector = ThreatDetector()
    threat_detector_path = os.path.join(export_dir, 'threat_detector.onnx')
    threat_detector.export_to_onnx(threat_detector_path)
    
    # Export RLManeuverPlanner
    print("\nExporting RLManeuverPlanner model...")
    rl_maneuver = RLManeuverPlanner()
    rl_maneuver_path = os.path.join(export_dir, 'rl_maneuver.onnx')
    rl_maneuver.export_to_onnx(rl_maneuver_path)
    
    # Export GameTheoryDeception
    print("\nExporting GameTheoryDeception model...")
    game_theory = GameTheoryDeception()
    game_theory_path = os.path.join(export_dir, 'game_theory.onnx')
    game_theory.export_to_onnx(game_theory_path)
    
    # Export ComplianceEvaluator
    print("\nExporting ComplianceEvaluator model...")
    compliance = ComplianceEvaluator()
    compliance_path = os.path.join(export_dir, 'compliance_evaluator.onnx')
    compliance.export_to_onnx(compliance_path)
    
    # Export AnalystEvaluator
    print("\nExporting AnalystEvaluator model...")
    analyst = AnalystEvaluator()
    analyst_path = os.path.join(export_dir, 'analyst_evaluator.onnx')
    analyst.export_to_onnx(analyst_path)
    
    # Export StimulationEvaluator
    print("\nExporting StimulationEvaluator model...")
    stimulation = StimulationEvaluator()
    stimulation_path = os.path.join(export_dir, 'stimulation_evaluator.onnx')
    stimulation.export_to_onnx(stimulation_path)
    
    # Export LaunchEvaluator
    print("\nExporting LaunchEvaluator model...")
    launch = LaunchEvaluator()
    launch_path = os.path.join(export_dir, 'launch_evaluator.onnx')
    launch.export_to_onnx(launch_path)
    
    # Export EnvironmentalEvaluator
    print("\nExporting EnvironmentalEvaluator model...")
    environmental = EnvironmentalEvaluator()
    environmental_path = os.path.join(export_dir, 'environmental_evaluator.onnx')
    environmental.export_to_onnx(environmental_path)
    
    print("\nAll analysis models exported successfully!")
    print("\nExported models:")
    print(f"1. ThreatDetector: {threat_detector_path}")
    print(f"2. RLManeuverPlanner: {rl_maneuver_path}")
    print(f"3. GameTheoryDeception: {game_theory_path}")
    print(f"4. ComplianceEvaluator: {compliance_path}")
    print(f"5. AnalystEvaluator: {analyst_path}")
    print(f"6. StimulationEvaluator: {stimulation_path}")
    print(f"7. LaunchEvaluator: {launch_path}")
    print(f"8. EnvironmentalEvaluator: {environmental_path}")

if __name__ == '__main__':
    export_analysis_models()

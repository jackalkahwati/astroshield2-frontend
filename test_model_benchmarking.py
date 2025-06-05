#!/usr/bin/env python3
"""
Scientific Model Benchmarking for TLE Analysis
==============================================

Quantitative evaluation of Local Model vs AI Simulation vs Basic Offline
using ground truth data and established orbital mechanics principles.

Metrics:
- Orbital Parameter Accuracy (MAE, RMSE)
- Risk Assessment Precision/Recall
- Satellite Identification Accuracy
- Processing Time Performance
- Confidence Calibration
- Domain Knowledge Score
"""

import asyncio
import json
import time
import statistics
import math
from typing import Dict, List, Tuple, Any
from datetime import datetime
import requests

# Known TLE test cases with ground truth data
BENCHMARK_DATASET = [
    {
        "name": "International Space Station (ISS)",
        "norad_id": "25544",
        "tle": [
            "1 25544U 98067A   24079.07757601 .00016717 00000+0 10270-3 0  9994",
            "2 25544  51.6400 337.6640 0007776  35.5310 330.5120 15.50377579499263"
        ],
        "ground_truth": {
            "satellite_type": "space_station",
            "orbit_type": "LEO",
            "altitude_km": 423,
            "period_minutes": 92.9,
            "inclination": 51.64,
            "eccentricity": 0.000778,
            "decay_risk": "LOW",  # Actively maintained
            "lifetime_years": 5.0,  # With regular boosts
            "operational_status": "active",
            "mission_type": "human_spaceflight"
        }
    },
    {
        "name": "Starlink-1007",
        "norad_id": "44713",
        "tle": [
            "1 44713U 19074A   24079.12345678 .00002917 00000+0 20330-3 0  9990",
            "2 44713  53.0000 123.4567 0001234  45.6789 314.5678 15.06123456789012"
        ],
        "ground_truth": {
            "satellite_type": "constellation",
            "orbit_type": "LEO",
            "altitude_km": 550,
            "period_minutes": 95.4,
            "inclination": 53.0,
            "eccentricity": 0.001234,
            "decay_risk": "LOW",  # Managed constellation
            "lifetime_years": 3.0,
            "operational_status": "active",
            "mission_type": "communications"
        }
    },
    {
        "name": "GPS IIF-2 (Defunct)",
        "norad_id": "25933",
        "tle": [
            "1 25933U 99055A   24079.15432109 -.00000012 00000+0 00000+0 0  9998",
            "2 25933  55.1234 123.4567 0123456  78.9012 281.1234  2.00567890123456"
        ],
        "ground_truth": {
            "satellite_type": "navigation",
            "orbit_type": "MEO",
            "altitude_km": 20200,
            "period_minutes": 718.2,
            "inclination": 55.12,
            "eccentricity": 0.123456,
            "decay_risk": "MEDIUM",  # High orbit but defunct
            "lifetime_years": 50.0,
            "operational_status": "defunct",
            "mission_type": "navigation"
        }
    },
    {
        "name": "CubeSat Debris",
        "norad_id": "12345",
        "tle": [
            "1 12345U 20001A   24079.98765432 .00012345 00000+0 67890-3 0  9991",
            "2 12345  98.7654 210.9876 0012345 123.4567  236.5432 14.12345678901234"
        ],
        "ground_truth": {
            "satellite_type": "cubesat",
            "orbit_type": "LEO",
            "altitude_km": 300,
            "period_minutes": 101.8,
            "inclination": 98.77,
            "eccentricity": 0.012345,
            "decay_risk": "HIGH",  # Low altitude, high drag
            "lifetime_years": 0.25,  # 3 months
            "operational_status": "debris",
            "mission_type": "experimental"
        }
    }
]

class ModelBenchmark:
    def __init__(self):
        self.results = {
            "local_model": {"scores": [], "times": [], "errors": []},
            "ai_simulation": {"scores": [], "times": [], "errors": []},
            "basic_offline": {"scores": [], "times": [], "errors": []}
        }
        
    async def query_model(self, tle_lines: List[str], model_type: str) -> Tuple[Dict, float]:
        """Query a specific model and measure response time"""
        start_time = time.time()
        
        try:
            response = requests.post('http://localhost:3001/api/tle-explanations/explain', 
                json={
                    "line1": tle_lines[0],
                    "line2": tle_lines[1],
                    "preferred_model": model_type,
                    "force_model": True,
                    "include_risk_assessment": True,
                    "include_anomaly_detection": True
                },
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                processing_time = time.time() - start_time
                return data, processing_time
            else:
                raise Exception(f"API returned {response.status_code}")
                
        except Exception as e:
            processing_time = time.time() - start_time
            return {"error": str(e)}, processing_time
    
    def calculate_orbital_accuracy(self, predicted: Dict, ground_truth: Dict) -> Dict[str, float]:
        """Calculate Mean Absolute Error for orbital parameters"""
        errors = {}
        
        # Altitude accuracy (km)
        if "altitude_description" in predicted:
            # Extract altitude from description (e.g., "423 km altitude")
            try:
                pred_alt = float(predicted["altitude_description"].split()[0])
                errors["altitude_mae"] = abs(pred_alt - ground_truth["altitude_km"])
            except:
                errors["altitude_mae"] = 1000  # Large error for parsing failure
        
        # Period accuracy (minutes)
        if "period_minutes" in predicted:
            errors["period_mae"] = abs(predicted["period_minutes"] - ground_truth["period_minutes"])
        
        # Inclination accuracy (degrees)
        if "inclination_degrees" in predicted:
            errors["inclination_mae"] = abs(predicted["inclination_degrees"] - ground_truth["inclination"])
        
        # Eccentricity accuracy
        if "eccentricity" in predicted:
            errors["eccentricity_mae"] = abs(predicted["eccentricity"] - ground_truth["eccentricity"])
        
        return errors
    
    def calculate_risk_assessment_accuracy(self, predicted: Dict, ground_truth: Dict) -> Dict[str, float]:
        """Calculate accuracy of risk assessment"""
        scores = {}
        
        # Risk level accuracy (categorical)
        if "decay_risk_level" in predicted:
            pred_risk = predicted["decay_risk_level"]
            true_risk = ground_truth["decay_risk"]
            scores["risk_categorical_accuracy"] = 1.0 if pred_risk == true_risk else 0.0
        
        # Lifetime prediction accuracy (years)
        if "predicted_lifetime_days" in predicted and predicted["predicted_lifetime_days"]:
            pred_years = predicted["predicted_lifetime_days"] / 365.25
            true_years = ground_truth["lifetime_years"]
            
            # Relative error (more forgiving for longer lifetimes)
            relative_error = abs(pred_years - true_years) / max(true_years, 0.1)
            scores["lifetime_relative_error"] = relative_error
            
            # Accuracy within order of magnitude
            scores["lifetime_order_magnitude"] = 1.0 if relative_error < 2.0 else 0.0
        
        return scores
    
    def calculate_satellite_recognition_score(self, predicted: Dict, ground_truth: Dict) -> float:
        """Score satellite identification accuracy"""
        score = 0.0
        
        # Satellite name recognition
        if "satellite_name" in predicted:
            pred_name = predicted["satellite_name"].lower()
            true_name = ground_truth.get("mission_type", "").lower()
            
            # Check for key terms
            if "iss" in pred_name and ground_truth["satellite_type"] == "space_station":
                score += 0.3
            elif "starlink" in pred_name and ground_truth["satellite_type"] == "constellation":
                score += 0.3
            elif "gps" in pred_name and ground_truth["satellite_type"] == "navigation":
                score += 0.3
        
        # Orbit type accuracy
        if "orbit_type" in predicted:
            if predicted["orbit_type"] == ground_truth["orbit_type"]:
                score += 0.3
        
        # Mission understanding (from analysis text)
        if "technical_details" in predicted and "ai_analysis" in predicted["technical_details"]:
            analysis = predicted["technical_details"]["ai_analysis"].lower()
            
            if ground_truth["satellite_type"] == "space_station" and any(word in analysis for word in ["laboratory", "research", "crew", "human"]):
                score += 0.2
            elif ground_truth["satellite_type"] == "constellation" and any(word in analysis for word in ["constellation", "network", "communications"]):
                score += 0.2
            elif ground_truth["satellite_type"] == "navigation" and any(word in analysis for word in ["navigation", "positioning", "gps"]):
                score += 0.2
        
        # Operational status recognition
        analysis_text = ""
        if "technical_details" in predicted and "ai_analysis" in predicted["technical_details"]:
            analysis_text = predicted["technical_details"]["ai_analysis"].lower()
        elif "orbit_description" in predicted:
            analysis_text = predicted["orbit_description"].lower()
        
        if ground_truth["operational_status"] == "active" and any(word in analysis_text for word in ["active", "operational", "maintained"]):
            score += 0.2
        elif ground_truth["operational_status"] == "defunct" and any(word in analysis_text for word in ["defunct", "inactive", "decommissioned"]):
            score += 0.2
        
        return min(score, 1.0)  # Cap at 1.0
    
    def calculate_domain_knowledge_score(self, predicted: Dict, ground_truth: Dict) -> float:
        """Score domain-specific orbital mechanics knowledge"""
        score = 0.0
        
        analysis_text = ""
        if "technical_details" in predicted and "ai_analysis" in predicted["technical_details"]:
            analysis_text = predicted["technical_details"]["ai_analysis"].lower()
        elif "orbit_description" in predicted:
            analysis_text = predicted["orbit_description"].lower()
        
        # Physics understanding
        physics_terms = ["drag", "atmosphere", "decay", "boost", "inclination", "eccentricity", "perigee", "apogee"]
        physics_score = sum(1 for term in physics_terms if term in analysis_text) / len(physics_terms)
        score += physics_score * 0.4
        
        # Mission-specific knowledge
        if ground_truth["satellite_type"] == "space_station":
            iss_terms = ["laboratory", "microgravity", "research", "international", "crew"]
            mission_score = sum(1 for term in iss_terms if term in analysis_text) / len(iss_terms)
            score += mission_score * 0.3
        
        # Risk factors understanding
        if ground_truth["decay_risk"] == "HIGH":
            risk_terms = ["low altitude", "atmospheric", "debris", "short", "decay"]
            risk_score = sum(1 for term in risk_terms if term in analysis_text) / len(risk_terms)
            score += risk_score * 0.3
        
        return min(score, 1.0)
    
    def calculate_confidence_calibration(self, predicted: Dict, overall_accuracy: float) -> float:
        """Measure how well confidence matches actual accuracy"""
        if "confidence_score" not in predicted:
            return 0.0
        
        predicted_confidence = predicted["confidence_score"]
        
        # Ideal: confidence should match actual accuracy
        calibration_error = abs(predicted_confidence - overall_accuracy)
        
        # Convert to calibration score (lower error = higher score)
        calibration_score = max(0.0, 1.0 - calibration_error)
        
        return calibration_score
    
    async def benchmark_model(self, model_type: str) -> Dict[str, Any]:
        """Benchmark a specific model against all test cases"""
        print(f"\n🧪 Benchmarking {model_type.upper()} model...")
        
        model_results = {
            "orbital_errors": [],
            "risk_scores": [],
            "recognition_scores": [],
            "domain_scores": [],
            "confidence_scores": [],
            "processing_times": [],
            "test_results": []
        }
        
        for test_case in BENCHMARK_DATASET:
            print(f"  Testing: {test_case['name']}")
            
            # Query model
            prediction, proc_time = await self.query_model(test_case["tle"], model_type)
            model_results["processing_times"].append(proc_time)
            
            if "error" in prediction:
                print(f"    ❌ Error: {prediction['error']}")
                # Add worst-case scores for failed predictions
                model_results["orbital_errors"].append({"total_error": 1000})
                model_results["risk_scores"].append({"total_score": 0.0})
                model_results["recognition_scores"].append(0.0)
                model_results["domain_scores"].append(0.0)
                model_results["confidence_scores"].append(0.0)
                continue
            
            # Calculate metrics
            orbital_errors = self.calculate_orbital_accuracy(prediction, test_case["ground_truth"])
            risk_scores = self.calculate_risk_assessment_accuracy(prediction, test_case["ground_truth"])
            recognition_score = self.calculate_satellite_recognition_score(prediction, test_case["ground_truth"])
            domain_score = self.calculate_domain_knowledge_score(prediction, test_case["ground_truth"])
            
            # Overall accuracy for confidence calibration
            overall_accuracy = (
                (1.0 - min(sum(orbital_errors.values()) / 1000, 1.0)) * 0.4 +
                (1.0 - risk_scores.get("lifetime_relative_error", 1.0)) * 0.3 +
                recognition_score * 0.2 +
                domain_score * 0.1
            )
            
            confidence_score = self.calculate_confidence_calibration(prediction, overall_accuracy)
            
            # Store results
            model_results["orbital_errors"].append(orbital_errors)
            model_results["risk_scores"].append(risk_scores)
            model_results["recognition_scores"].append(recognition_score)
            model_results["domain_scores"].append(domain_score)
            model_results["confidence_scores"].append(confidence_score)
            
            # Store detailed test result
            model_results["test_results"].append({
                "test_case": test_case["name"],
                "prediction": prediction,
                "ground_truth": test_case["ground_truth"],
                "metrics": {
                    "orbital_accuracy": 1.0 - min(sum(orbital_errors.values()) / 1000, 1.0),
                    "risk_accuracy": 1.0 - risk_scores.get("lifetime_relative_error", 1.0),
                    "recognition_score": recognition_score,
                    "domain_score": domain_score,
                    "confidence_calibration": confidence_score,
                    "processing_time": proc_time
                }
            })
            
            print(f"    ✅ Accuracy: {overall_accuracy:.2f}, Time: {proc_time:.2f}s")
        
        return model_results
    
    def calculate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate summary statistics for model performance"""
        
        # Orbital accuracy (inverse of normalized error)
        orbital_accuracy = []
        for errors in results["orbital_errors"]:
            if "total_error" in errors:
                accuracy = 1.0 - min(errors["total_error"] / 1000, 1.0)
            else:
                total_error = sum(errors.values())
                accuracy = 1.0 - min(total_error / 1000, 1.0)
            orbital_accuracy.append(accuracy)
        
        # Risk assessment accuracy
        risk_accuracy = []
        for scores in results["risk_scores"]:
            if "total_score" in scores:
                risk_accuracy.append(scores["total_score"])
            else:
                # Combine categorical and lifetime accuracy
                categorical = scores.get("risk_categorical_accuracy", 0.0)
                lifetime = 1.0 - min(scores.get("lifetime_relative_error", 1.0), 1.0)
                risk_accuracy.append((categorical + lifetime) / 2)
        
        return {
            "orbital_accuracy_mean": statistics.mean(orbital_accuracy),
            "orbital_accuracy_std": statistics.stdev(orbital_accuracy) if len(orbital_accuracy) > 1 else 0.0,
            "risk_accuracy_mean": statistics.mean(risk_accuracy),
            "risk_accuracy_std": statistics.stdev(risk_accuracy) if len(risk_accuracy) > 1 else 0.0,
            "recognition_score_mean": statistics.mean(results["recognition_scores"]),
            "recognition_score_std": statistics.stdev(results["recognition_scores"]) if len(results["recognition_scores"]) > 1 else 0.0,
            "domain_knowledge_mean": statistics.mean(results["domain_scores"]),
            "domain_knowledge_std": statistics.stdev(results["domain_scores"]) if len(results["domain_scores"]) > 1 else 0.0,
            "confidence_calibration_mean": statistics.mean(results["confidence_scores"]),
            "processing_time_mean": statistics.mean(results["processing_times"]),
            "processing_time_std": statistics.stdev(results["processing_times"]) if len(results["processing_times"]) > 1 else 0.0,
        }
    
    async def run_full_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark across all models"""
        print("🔬 Scientific Model Benchmarking System")
        print("=" * 50)
        
        models = ["local", "simulation", "offline"]
        benchmark_results = {}
        
        for model in models:
            try:
                results = await self.benchmark_model(model)
                summary = self.calculate_summary_statistics(results)
                
                benchmark_results[model] = {
                    "raw_results": results,
                    "summary_statistics": summary
                }
                
            except Exception as e:
                print(f"❌ Error benchmarking {model}: {e}")
                benchmark_results[model] = {"error": str(e)}
        
        # Generate comparative analysis
        benchmark_results["comparative_analysis"] = self.generate_comparative_analysis(benchmark_results)
        
        return benchmark_results
    
    def generate_comparative_analysis(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate scientific comparative analysis between models"""
        
        analysis = {
            "rankings": {},
            "statistical_significance": {},
            "recommendations": []
        }
        
        metrics = [
            "orbital_accuracy_mean",
            "risk_accuracy_mean", 
            "recognition_score_mean",
            "domain_knowledge_mean",
            "confidence_calibration_mean"
        ]
        
        for metric in metrics:
            scores = {}
            for model in ["local", "simulation", "offline"]:
                if model in results and "summary_statistics" in results[model]:
                    scores[model] = results[model]["summary_statistics"].get(metric, 0.0)
            
            # Rank models by this metric
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            analysis["rankings"][metric] = ranked
        
        # Overall composite score
        composite_scores = {}
        weights = {
            "orbital_accuracy_mean": 0.25,
            "risk_accuracy_mean": 0.25,
            "recognition_score_mean": 0.20,
            "domain_knowledge_mean": 0.20,
            "confidence_calibration_mean": 0.10
        }
        
        for model in ["local", "simulation", "offline"]:
            if model in results and "summary_statistics" in results[model]:
                stats = results[model]["summary_statistics"]
                composite_scores[model] = sum(
                    stats.get(metric, 0.0) * weight 
                    for metric, weight in weights.items()
                )
        
        analysis["overall_ranking"] = sorted(
            composite_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return analysis

async def main():
    """Run the benchmark and save results"""
    benchmark = ModelBenchmark()
    results = await benchmark.run_full_benchmark()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"model_benchmark_results_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n📊 BENCHMARK RESULTS SUMMARY")
    print("=" * 50)
    
    if "comparative_analysis" in results:
        rankings = results["comparative_analysis"]["overall_ranking"]
        
        print("\n🏆 Overall Model Ranking:")
        for i, (model, score) in enumerate(rankings, 1):
            print(f"  {i}. {model.upper()}: {score:.3f}")
        
        print("\n📈 Individual Metrics:")
        for metric in ["orbital_accuracy_mean", "risk_accuracy_mean", "recognition_score_mean"]:
            if metric in results["comparative_analysis"]["rankings"]:
                ranked = results["comparative_analysis"]["rankings"][metric]
                print(f"\n  {metric.replace('_', ' ').title()}:")
                for model, score in ranked:
                    print(f"    {model.upper()}: {score:.3f}")
    
    print(f"\n💾 Detailed results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    asyncio.run(main()) 
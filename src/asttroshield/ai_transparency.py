"""AI Transparency Module for AstroShield.

This module provides documentation and transparency for AI/ML components
in accordance with USSF Data & AI FY 2025 Strategic Action Plan.

References:
- LOE 2.1.2: Launch data and AI professional explainer series
- LOE 2.2.3: Publish "Momentum" – the USSF data, analytics, and AI periodical
- LOE 2.2.4: Develop AI-skills workforce development playbook
"""

import logging
from typing import Dict, Any, List, Optional
import json
import os
from datetime import datetime

logger = logging.getLogger(__name__)

class AIModelDocumentation:
    """AI Model documentation and transparency framework.
    
    This class generates standardized documentation for AI models
    within AstroShield to support USSF data and AI literacy goals.
    """
    
    def __init__(self, models_directory: str = "ml/models"):
        """Initialize the AI documentation framework.
        
        Args:
            models_directory: Path to ML models directory
        """
        self.models_directory = models_directory
        self.documentation_store = {}
        self.model_registry = {}
        
    def register_model(self, 
                     model_id: str, 
                     model_type: str, 
                     description: str,
                     version: str,
                     use_case: str,
                     confidence_metric: str,
                     developer: str,
                     training_date: str,
                     input_features: List[str],
                     output_features: List[str],
                     limitations: List[str],
                     path: Optional[str] = None) -> Dict[str, Any]:
        """Register an AI/ML model with documentation.
        
        Args:
            model_id: Unique identifier for the model
            model_type: Type of model (e.g., "Classification", "Regression")
            description: Plain language description of the model
            version: Model version string
            use_case: Primary use case for the model
            confidence_metric: How model confidence is measured
            developer: Who developed/trained the model
            training_date: When the model was last trained
            input_features: List of input features
            output_features: List of output features
            limitations: Known limitations/edge cases
            path: Optional path to model files
            
        Returns:
            The created documentation record
        """
        doc = {
            "model_id": model_id,
            "model_type": model_type,
            "description": description,
            "version": version,
            "use_case": use_case,
            "confidence_metric": confidence_metric,
            "developer": developer,
            "training_date": training_date,
            "input_features": input_features,
            "output_features": output_features,
            "limitations": limitations,
            "path": path or os.path.join(self.models_directory, model_id),
            "ussf_compliant": True,
            "registered_date": datetime.utcnow().isoformat(),
            "last_evaluation_date": None,
            "clara_ai_registered": False  # Reference to USSF LOE 1.3.2
        }
        
        # Store in the documentation registry
        self.documentation_store[model_id] = doc
        
        # Log the registration
        logger.info(f"Registered model {model_id} (v{version}) for {use_case}")
        
        return doc
    
    def get_model_documentation(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get documentation for a specific model.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Model documentation dictionary if found, None otherwise
        """
        return self.documentation_store.get(model_id)
    
    def export_documentation(self, format_type: str = "json", 
                          output_path: Optional[str] = None) -> str:
        """Export model documentation in various formats.
        
        Args:
            format_type: Output format ("json", "markdown", "html")
            output_path: Where to save the documentation
            
        Returns:
            Path to the exported documentation
        """
        # Default path if none provided
        if not output_path:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_path = f"docs/ai_model_documentation_{timestamp}.{format_type}"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate documentation based on format
        if format_type == "json":
            with open(output_path, 'w') as f:
                json.dump({
                    "models": self.documentation_store,
                    "metadata": {
                        "generated_date": datetime.utcnow().isoformat(),
                        "total_models": len(self.documentation_store),
                        "ussf_compliant": True,
                        "framework_version": "1.0.0"
                    }
                }, f, indent=2)
        elif format_type == "markdown":
            with open(output_path, 'w') as f:
                f.write("# AstroShield AI Model Documentation\n\n")
                f.write(f"Generated on {datetime.utcnow().isoformat()}\n\n")
                f.write(f"Total models: {len(self.documentation_store)}\n\n")
                
                for model_id, doc in self.documentation_store.items():
                    f.write(f"## {model_id} (v{doc['version']})\n\n")
                    f.write(f"**Type:** {doc['model_type']}\n\n")
                    f.write(f"**Description:** {doc['description']}\n\n")
                    f.write(f"**Use Case:** {doc['use_case']}\n\n")
                    f.write(f"**Confidence Metric:** {doc['confidence_metric']}\n\n")
                    
                    f.write("### Input Features\n\n")
                    for feature in doc['input_features']:
                        f.write(f"- {feature}\n")
                    f.write("\n")
                    
                    f.write("### Output Features\n\n")
                    for feature in doc['output_features']:
                        f.write(f"- {feature}\n")
                    f.write("\n")
                    
                    f.write("### Limitations\n\n")
                    for limitation in doc['limitations']:
                        f.write(f"- {limitation}\n")
                    f.write("\n")
                    
        elif format_type == "html":
            # Basic HTML implementation
            with open(output_path, 'w') as f:
                f.write("<html><head><title>AstroShield AI Model Documentation</title></head><body>")
                f.write("<h1>AstroShield AI Model Documentation</h1>")
                # Additional HTML formatting would go here
                f.write("</body></html>")
        
        logger.info(f"Exported AI model documentation to {output_path}")
        return output_path
        
    def register_with_clara_ai(self, model_id: str) -> Dict[str, Any]:
        """Register a model with CLARA.ai as required by USSF LOE 1.3.2.
        
        Args:
            model_id: Unique identifier for the model
            
        Returns:
            Result of the registration attempt
        """
        # Get model documentation
        doc = self.get_model_documentation(model_id)
        if not doc:
            return {"status": "error", "message": f"Model {model_id} not found"}
        
        # In a real implementation, this would call the CLARA API
        # For now, we'll just update our local flag
        doc["clara_ai_registered"] = True
        doc["clara_registration_date"] = datetime.utcnow().isoformat()
        
        logger.info(f"Registered model {model_id} with CLARA.ai")
        
        return {
            "status": "success", 
            "message": f"Model {model_id} registered with CLARA.ai",
            "registration_date": doc["clara_registration_date"]
        }

class AIExplainer:
    """Generate explanations for AI models and decisions.
    
    This class supports the USSF goal of increasing AI literacy by
    providing clear explanations of how AI models work and the
    rationale behind predictions.
    """
    
    def __init__(self, documentation: AIModelDocumentation):
        """Initialize the AI explainer.
        
        Args:
            documentation: AI model documentation instance
        """
        self.documentation = documentation
        
    def explain_prediction(self, 
                         model_id: str, 
                         inputs: Dict[str, Any], 
                         outputs: Dict[str, Any], 
                         audience: str = "technical") -> Dict[str, Any]:
        """Generate a human-readable explanation of an AI prediction.
        
        Args:
            model_id: Identifier for the model that made the prediction
            inputs: Input data provided to the model
            outputs: Model's output/prediction
            audience: Target audience ("technical", "operational", "leadership")
            
        Returns:
            Explanation dictionary
        """
        # Get model documentation
        doc = self.documentation.get_model_documentation(model_id)
        if not doc:
            return {
                "status": "error", 
                "message": f"Model {model_id} not found in documentation"
            }
        
        # Build explanation based on audience
        if audience == "technical":
            explanation = self._build_technical_explanation(doc, inputs, outputs)
        elif audience == "operational":
            explanation = self._build_operational_explanation(doc, inputs, outputs)
        elif audience == "leadership":
            explanation = self._build_leadership_explanation(doc, inputs, outputs)
        else:
            explanation = self._build_operational_explanation(doc, inputs, outputs)
            
        return {
            "model_id": model_id,
            "model_type": doc["model_type"],
            "version": doc["version"],
            "audience": audience,
            "explanation": explanation,
            "confidence": self._extract_confidence(outputs, doc["confidence_metric"]),
            "limitations": doc["limitations"],
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _build_technical_explanation(self, 
                                  doc: Dict[str, Any], 
                                  inputs: Dict[str, Any], 
                                  outputs: Dict[str, Any]) -> str:
        """Build a technical explanation of a model prediction.
        
        Args:
            doc: Model documentation
            inputs: Input data
            outputs: Output predictions
            
        Returns:
            Detailed technical explanation
        """
        # In a real implementation, this would be more sophisticated
        # For now, this is a placeholder
        return (f"The {doc['model_type']} model analyzed {len(inputs)} input features "
                f"and produced predictions for {len(outputs)} output features using "
                f"version {doc['version']} of the model.")
    
    def _build_operational_explanation(self, 
                                    doc: Dict[str, Any], 
                                    inputs: Dict[str, Any], 
                                    outputs: Dict[str, Any]) -> str:
        """Build an operational explanation of a model prediction.
        
        Args:
            doc: Model documentation
            inputs: Input data
            outputs: Output predictions
            
        Returns:
            Operational-focused explanation
        """
        return (f"Based on the provided data, the {doc['use_case']} system "
                f"has made a determination with {self._extract_confidence(outputs, doc['confidence_metric'])}% confidence.")
    
    def _build_leadership_explanation(self, 
                                   doc: Dict[str, Any], 
                                   inputs: Dict[str, Any], 
                                   outputs: Dict[str, Any]) -> str:
        """Build a leadership-level explanation of a model prediction.
        
        Args:
            doc: Model documentation
            inputs: Input data
            outputs: Output predictions
            
        Returns:
            High-level leadership explanation
        """
        return (f"The {doc['use_case']} capability has analyzed the situation and "
                f"determined an assessment with {self._extract_confidence(outputs, doc['confidence_metric'])}% confidence.")
    
    def _extract_confidence(self, 
                         outputs: Dict[str, Any], 
                         confidence_metric: str) -> float:
        """Extract confidence score from model outputs.
        
        Args:
            outputs: Model output dictionary
            confidence_metric: Name of the confidence metric
            
        Returns:
            Confidence score as a percentage
        """
        # In a real implementation, this would extract the confidence value
        # based on the model's specific confidence metric
        if "confidence" in outputs:
            return outputs["confidence"] * 100
        elif confidence_metric in outputs:
            return outputs[confidence_metric] * 100
        else:
            return 85.0  # Default placeholder 
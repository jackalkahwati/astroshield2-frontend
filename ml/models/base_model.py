from typing import Dict, Any
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def evaluate(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate input data and return predictions
        """
        try:
            with torch.no_grad():
                # Convert input data to tensor
                input_tensor = self._preprocess_data(data)
                
                # Get model predictions
                output = self.forward(input_tensor)
                
                # Process output
                result = self._postprocess_output(output)
                
                return {
                    'score': float(result['score']),
                    'confidence': float(result['confidence']),
                    'details': result['details']
                }
        except Exception as e:
            return {
                'score': 0.0,
                'confidence': 0.0,
                'details': {'error': str(e)}
            }

    def _preprocess_data(self, data: Dict[str, Any]) -> torch.Tensor:
        """
        Convert input data to tensor format
        To be implemented by child classes
        """
        raise NotImplementedError

    def _postprocess_output(self, output: torch.Tensor) -> Dict[str, Any]:
        """
        Convert model output to final prediction format
        To be implemented by child classes
        """
        raise NotImplementedError

    def save(self, path: str):
        """Save model weights"""
        torch.save(self.state_dict(), path)

    def load(self, path: str):
        """Load model weights"""
        self.load_state_dict(torch.load(path, map_location=self.device)) 
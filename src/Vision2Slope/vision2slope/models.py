"""
Segmentation model wrapper for Vision2Slope pipeline.
"""

import logging
import torch
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

from .core.interfaces import SegmentationProvider


class SegmentationModel(SegmentationProvider):
    """Wrapper for semantic segmentation model."""
    
    def __init__(self, model_name: str):
        """
        Initialize segmentation model.
        
        Args:
            model_name: HuggingFace model identifier
        """
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(__name__)
        self._load_model()
    
    def _load_model(self):
        """Load the pre-trained segmentation model."""
        try:
            self.logger.info(f"Loading segmentation model: {self.model_name}")
            self.processor = AutoImageProcessor.from_pretrained(self.model_name)
            self.model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            self.logger.info(f"Model loaded successfully on device: {self.device}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
    
    def segment(self, image: Image.Image) -> np.ndarray:
        """
        Perform semantic segmentation on an image.
        
        Args:
            image: PIL Image object
            
        Returns:
            Semantic segmentation map as numpy array
        """
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            predicted_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[image.size[::-1]]
            )[0].cpu().numpy()
            
            return predicted_map
            
        except Exception as e:
            self.logger.error(f"Segmentation failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """
        Get information about the segmentation model.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            "model_name": self.model_name,
            "device": str(self.device),
            "model_type": "Mask2Former"
        }

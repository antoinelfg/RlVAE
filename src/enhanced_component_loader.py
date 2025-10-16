
"""
Enhanced Component Loader for Three-Stage Pipeline
==================================================

Reduces multiple initialization phases by providing a unified component loading system.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from omegaconf import DictConfig
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class EnhancedComponentLoader:
    """Enhanced component loader that reduces multiple init phases."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.loaded_components = {}
        self.load_history = []
        self.validation_results = {}
    
    def load_all_components(self, stage_a_data: Dict[str, Any], 
                           stage_b_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load all components in a single phase to reduce init phases."""
        
        logger.info("🔄 Loading all components in single phase...")
        
        components = {}
        
        # Load Stage A components (encoder/decoder)
        if stage_a_data:
            components.update(self._load_stage_a_components(stage_a_data))
        
        # Load Stage B components (metric)
        if stage_b_data:
            components.update(self._load_stage_b_components(stage_b_data))
        
        # Validate component compatibility
        validation = self._validate_all_components(components)
        self.validation_results = validation
        
        if not validation['compatible']:
            logger.warning(f"⚠️ Component compatibility issues: {validation['issues']}")
        
        # Store loaded components
        self.loaded_components = components
        self.load_history.append({
            'timestamp': datetime.now().isoformat(),
            'components_loaded': list(components.keys()),
            'validation_passed': validation['compatible']
        })
        
        logger.info(f"✅ Loaded {len(components)} components in single phase")
        return components
    
    def _load_stage_a_components(self, stage_a_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load Stage A components (encoder/decoder)."""
        
        components = {}
        
        # Load encoder
        if 'encoder_path' in stage_a_data:
            try:
                encoder_state = torch.load(stage_a_data['encoder_path'], map_location='cpu', weights_only=False)
                components['encoder'] = encoder_state
                logger.info(f"✅ Loaded encoder from: {stage_a_data['encoder_path']}")
            except Exception as e:
                logger.error(f"❌ Failed to load encoder: {e}")
                raise
        
        # Load decoder
        if 'decoder_path' in stage_a_data:
            try:
                decoder_state = torch.load(stage_a_data['decoder_path'], map_location='cpu', weights_only=False)
                components['decoder'] = decoder_state
                logger.info(f"✅ Loaded decoder from: {stage_a_data['decoder_path']}")
            except Exception as e:
                logger.error(f"❌ Failed to load decoder: {e}")
                raise
        
        return components
    
    def _load_stage_b_components(self, stage_b_data: Dict[str, Any]) -> Dict[str, Any]:
        """Load Stage B components (metric)."""
        
        components = {}
        
        # Load metric
        if 'metric_path' in stage_b_data:
            try:
                metric_state = torch.load(stage_b_data['metric_path'], map_location='cpu', weights_only=False)
                components['metric'] = metric_state
                logger.info(f"✅ Loaded metric from: {stage_b_data['metric_path']}")
            except Exception as e:
                logger.error(f"❌ Failed to load metric: {e}")
                raise
        
        return components
    
    def _validate_all_components(self, components: Dict[str, Any]) -> Dict[str, Any]:
        """Validate all loaded components for compatibility."""
        
        issues = []
        
        # Check if all required components are present
        required_components = ['encoder', 'decoder', 'metric']
        missing_components = [comp for comp in required_components if comp not in components]
        
        if missing_components:
            issues.append(f"Missing components: {missing_components}")
        
        # Validate component architectures
        for comp_name, comp_state in components.items():
            if not self._validate_component_architecture(comp_name, comp_state):
                issues.append(f"Invalid architecture for {comp_name}")
        
        return {
            'compatible': len(issues) == 0,
            'issues': issues
        }
    
    def _validate_component_architecture(self, comp_name: str, comp_state: Any) -> bool:
        """Validate component architecture."""
        
        try:
            if hasattr(comp_state, 'state_dict'):
                state_dict = comp_state.state_dict()
            else:
                state_dict = comp_state
            
            # Basic validation
            if not state_dict or len(state_dict) == 0:
                return False
            
            # Check for reasonable number of parameters
            total_params = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            if total_params < 100:  # Too few parameters
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_loading_summary(self) -> Dict[str, Any]:
        """Get summary of component loading."""
        return {
            'components_loaded': list(self.loaded_components.keys()),
            'validation_results': self.validation_results,
            'load_history': self.load_history
        }

"""
Configuration Synchronizer
==========================

Ensures configuration consistency across pipeline stages and handles
stage-specific overrides properly.
"""

from typing import Dict, Any, List, Optional
from omegaconf import DictConfig, OmegaConf
import warnings


class ConfigSynchronizer:
    """
    Synchronizes configuration across three-stage pipeline.
    
    This synchronizer ensures:
    - Critical parameters are propagated across stages
    - Stage-specific overrides are handled properly
    - Configuration history is maintained
    - Stage transitions are validated
    """
    
    # Parameters that must be consistent across all stages
    CRITICAL_PARAMS = [
        'latent_dim',
        'input_dim', 
        'data.name',
        'seed'
    ]
    
    # Parameters that should be propagated but can be overridden
    PROPAGATED_PARAMS = [
        'posterior_type',
        'architecture',
        'temperature',
        'regularization',
        'beta',
        'riemannian_beta'
    ]
    
    # Stage-specific parameter mappings
    STAGE_MAPPINGS = {
        'stage_a': {
            'model_type': 'vanilla_vae',  # Stage A typically uses vanilla VAE
            'required_outputs': ['encoder', 'decoder', 'latent_samples']
        },
        'stage_b': {
            'model_type': 'metric_learning',  # Stage B learns the metric
            'required_outputs': ['metric', 'centroids']
        },
        'stage_c': {
            'model_type': 'rlvae',  # Stage C uses full RLVAE
            'required_inputs': ['encoder', 'decoder', 'metric'],
            'required_outputs': ['full_model']
        }
    }
    
    def __init__(self):
        self.config_history = []
        self.stage_configs = {}
    
    def sync_pipeline_config(self, config: DictConfig) -> DictConfig:
        """
        Synchronize configuration for three-stage pipeline.
        
        Args:
            config: Base pipeline configuration
            
        Returns:
            Synchronized configuration with proper stage settings
        """
        # Make a copy to avoid modifying original
        config = OmegaConf.create(config)
        
        # Store original config
        self.config_history.append(OmegaConf.create(config))
        
        # Ensure experiment configuration exists
        if 'experiment' not in config:
            config.experiment = {}
        
        # Synchronize critical parameters across stages
        self._sync_critical_params(config)
        
        # Propagate common parameters to stages
        self._propagate_stage_params(config)
        
        # Validate stage transitions
        self._validate_stage_config(config)
        
        # Fix struct mode issues
        self._fix_struct_mode_issues(config)
        
        return config
    
    def _sync_critical_params(self, config: DictConfig):
        """Ensure critical parameters are consistent across all stages."""
        base_values = {}
        
        # Extract critical parameters from base config
        for param in self.CRITICAL_PARAMS:
            if '.' in param:
                # Handle nested parameters like 'data.name'
                parts = param.split('.')
                value = config
                for part in parts:
                    if part in value:
                        value = value[part]
                    else:
                        value = None
                        break
                base_values[param] = value
            else:
                base_values[param] = config.get(param)
        
        # Propagate to all stages
        for stage in ['stage_a', 'stage_b', 'stage_c']:
            if stage not in config.experiment:
                # Temporarily disable struct to allow creating stage dicts
                original_struct = OmegaConf.is_struct(config.experiment)
                OmegaConf.set_struct(config.experiment, False)
                config.experiment[stage] = {}
                OmegaConf.set_struct(config.experiment, original_struct)
            
            for param, value in base_values.items():
                if value is not None:
                    if '.' in param:
                        # Handle nested parameters
                        parts = param.split('.')
                        target = config.experiment[stage]
                        
                        # Temporarily disable struct mode for nested parameter creation
                        original_struct = OmegaConf.is_struct(target)
                        OmegaConf.set_struct(target, False)
                        
                        for part in parts[:-1]:
                            if part not in target:
                                target[part] = {}
                            target = target[part]
                        target[parts[-1]] = value
                        
                        # Restore original struct mode
                        OmegaConf.set_struct(config.experiment[stage], original_struct)
                    else:
                        # Handle simple parameter assignment with struct mode
                        original_struct = OmegaConf.is_struct(config.experiment[stage])
                        OmegaConf.set_struct(config.experiment[stage], False)
                        config.experiment[stage][param] = value
                        OmegaConf.set_struct(config.experiment[stage], original_struct)
    
    def _propagate_stage_params(self, config: DictConfig):
        """Propagate common parameters to stages while allowing overrides."""
        base_values = {}
        
        # Extract propagated parameters from base config
        for param in self.PROPAGATED_PARAMS:
            if param in config:
                base_values[param] = config[param]
            elif 'model' in config and param in config.model:
                base_values[param] = config.model[param]
        
        # Propagate to stages that don't already have these parameters
        for stage in ['stage_a', 'stage_b', 'stage_c']:
            if stage not in config.experiment:
                continue
                
            stage_config = config.experiment[stage]
            
            for param, value in base_values.items():
                # Only set if not already specified in stage config
                if param not in stage_config:
                    # Handle struct mode for parameter assignment
                    original_struct = OmegaConf.is_struct(stage_config)
                    OmegaConf.set_struct(stage_config, False)
                    stage_config[param] = value
                    OmegaConf.set_struct(stage_config, original_struct)
                    
                # Also propagate to model config within stage
                if 'model' not in stage_config:
                    original_struct = OmegaConf.is_struct(stage_config)
                    OmegaConf.set_struct(stage_config, False)
                    stage_config['model'] = {}
                    OmegaConf.set_struct(stage_config, original_struct)
                if param not in stage_config.model:
                    # Check if stage_config.model is a DictConfig before handling struct mode
                    if hasattr(stage_config.model, '_get_flag'):  # Check if it's a DictConfig
                        original_struct = OmegaConf.is_struct(stage_config.model)
                        OmegaConf.set_struct(stage_config.model, False)
                        stage_config.model[param] = value
                        OmegaConf.set_struct(stage_config.model, original_struct)
                    else:
                        # If it's not a DictConfig (e.g., string), convert it first
                        from omegaconf import DictConfig
                        stage_config.model = DictConfig({param: value})
    
    def _validate_stage_config(self, config: DictConfig):
        """Validate stage-specific configurations."""
        experiment = config.experiment
        
        # Validate Stage A
        if 'stage_a' in experiment:
            stage_a = experiment.stage_a
            try:
                from omegaconf import OmegaConf
                if 'epochs' not in stage_a:
                    original_struct = OmegaConf.is_struct(stage_a)
                    OmegaConf.set_struct(stage_a, False)
                    stage_a.epochs = 50  # Default
                    OmegaConf.set_struct(stage_a, original_struct)
                if 'model' not in stage_a:
                    original_struct = OmegaConf.is_struct(stage_a)
                    OmegaConf.set_struct(stage_a, False)
                    stage_a.model = 'vanilla_vae'
                    OmegaConf.set_struct(stage_a, original_struct)
            except Exception:
                pass
        
        # Validate Stage B
        if 'stage_b' in experiment:
            stage_b = experiment.stage_b
            try:
                from omegaconf import OmegaConf
                if 'implementation' not in stage_b:
                    original_struct = OmegaConf.is_struct(stage_b)
                    OmegaConf.set_struct(stage_b, False)
                    stage_b.implementation = 'rhvae'
                    OmegaConf.set_struct(stage_b, original_struct)
                if 'n_centroids' not in stage_b:
                    original_struct = OmegaConf.is_struct(stage_b)
                    OmegaConf.set_struct(stage_b, False)
                    stage_b.n_centroids = 300
                    OmegaConf.set_struct(stage_b, original_struct)
            except Exception:
                pass
        
        # Validate Stage C
        if 'stage_c' in experiment:
            stage_c = experiment.stage_c
            try:
                from omegaconf import OmegaConf
                if 'epochs' not in stage_c:
                    original_struct = OmegaConf.is_struct(stage_c)
                    OmegaConf.set_struct(stage_c, False)
                    stage_c.epochs = 30  # Default
                    OmegaConf.set_struct(stage_c, original_struct)
                if 'model' not in stage_c:
                    original_struct = OmegaConf.is_struct(stage_c)
                    OmegaConf.set_struct(stage_c, False)
                    stage_c.model = 'modular_rlvae'
                    OmegaConf.set_struct(stage_c, original_struct)
            except Exception:
                pass
            
            # Ensure posterior type is properly set
            if 'posterior_type' not in stage_c and 'model' in config:
                try:
                    from omegaconf import OmegaConf
                    original_struct = OmegaConf.is_struct(stage_c)
                    OmegaConf.set_struct(stage_c, False)
                    if 'posterior_type' in config.model:
                        stage_c.posterior_type = config.model.posterior_type
                    elif 'posterior' in config.model and 'type' in config.model.posterior:
                        stage_c.posterior_type = config.model.posterior.type
                    OmegaConf.set_struct(stage_c, original_struct)
                except Exception:
                    pass
    
    def _fix_struct_mode_issues(self, config: DictConfig):
        """Fix Hydra struct mode issues by ensuring all required keys exist."""
        # Ensure model config has all required fields
        from omegaconf import OmegaConf
        if 'model' not in config:
            original_struct = OmegaConf.is_struct(config)
            OmegaConf.set_struct(config, False)
            config.model = {}
            OmegaConf.set_struct(config, original_struct)
        
        model_config = config.model
        
        # Add posterior_type if missing
        if 'posterior_type' not in model_config:
            original_struct = OmegaConf.is_struct(model_config)
            OmegaConf.set_struct(model_config, False)
            if 'posterior' in model_config and isinstance(model_config.posterior, dict):
                if 'type' in model_config.posterior:
                    model_config.posterior_type = model_config.posterior.type
                else:
                    model_config.posterior_type = 'riemannian_metric'
            else:
                model_config.posterior_type = 'riemannian_metric'
            OmegaConf.set_struct(model_config, original_struct)
        
        # Ensure posterior config exists and matches posterior_type
        if 'posterior' not in model_config:
            original_struct = OmegaConf.is_struct(model_config)
            OmegaConf.set_struct(model_config, False)
            model_config.posterior = {'type': model_config.posterior_type}
            OmegaConf.set_struct(model_config, original_struct)
        elif isinstance(model_config.posterior, dict):
            if 'type' not in model_config.posterior:
                original_struct = OmegaConf.is_struct(model_config.posterior)
                OmegaConf.set_struct(model_config.posterior, False)
                model_config.posterior.type = model_config.posterior_type
                OmegaConf.set_struct(model_config.posterior, original_struct)

        # Mirror top-level pretrained.* overrides into model.pretrained.* if present
        if 'pretrained' in config:
            if 'pretrained' not in model_config:
                original_struct = OmegaConf.is_struct(model_config)
                OmegaConf.set_struct(model_config, False)
                model_config.pretrained = {}
                OmegaConf.set_struct(model_config, original_struct)
            # Copy known keys if not present
            for key in [
                'encoder_path', 'decoder_path', 'metric_path'
            ]:
                try:
                    top_val = config.pretrained.get(key, None)
                    if top_val is not None:
                        original_struct = OmegaConf.is_struct(model_config.pretrained)
                        OmegaConf.set_struct(model_config.pretrained, False)
                        model_config.pretrained[key] = top_val
                        OmegaConf.set_struct(model_config.pretrained, original_struct)
                except Exception:
                    pass
        
        # Ensure training config has model section
        if 'training' not in config:
            original_struct = OmegaConf.is_struct(config)
            OmegaConf.set_struct(config, False)
            config.training = {}
            OmegaConf.set_struct(config, original_struct)
        if 'model' not in config.training:
            original_struct = OmegaConf.is_struct(config.training)
            OmegaConf.set_struct(config.training, False)
            config.training.model = {}
            OmegaConf.set_struct(config.training, original_struct)
        
        # Sync posterior_type to training config
        if 'posterior_type' not in config.training.model:
            original_struct = OmegaConf.is_struct(config.training.model)
            OmegaConf.set_struct(config.training.model, False)
            config.training.model.posterior_type = model_config.posterior_type
            OmegaConf.set_struct(config.training.model, original_struct)
        
        # Ensure posterior config in training matches
        if 'posterior' not in config.training.model:
            original_struct = OmegaConf.is_struct(config.training.model)
            OmegaConf.set_struct(config.training.model, False)
            config.training.model.posterior = {'type': model_config.posterior_type}
            OmegaConf.set_struct(config.training.model, original_struct)
    
    def get_stage_config(self, config: DictConfig, stage: str) -> DictConfig:
        """
        Get configuration for a specific stage.
        
        Args:
            config: Full pipeline configuration
            stage: Stage name ('stage_a', 'stage_b', 'stage_c')
            
        Returns:
            Stage-specific configuration
        """
        if stage not in config.experiment:
            raise ValueError(f"Stage '{stage}' not found in experiment configuration")
        
        # Create stage config by merging base config with stage-specific overrides
        stage_config = OmegaConf.create(config)
        stage_overrides = config.experiment[stage]
        
        # Apply stage-specific overrides
        stage_config = OmegaConf.merge(stage_config, {'model': stage_overrides})
        
        # Store for history
        self.stage_configs[stage] = stage_config
        
        return stage_config
    
    def validate_stage_transition(self, from_stage: str, to_stage: str, 
                                 from_outputs: Dict[str, Any]) -> List[str]:
        """
        Validate that a stage transition is valid.
        
        Args:
            from_stage: Source stage
            to_stage: Target stage
            from_outputs: Outputs from source stage
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Check if target stage has required inputs
        if to_stage in self.STAGE_MAPPINGS:
            required_inputs = self.STAGE_MAPPINGS[to_stage].get('required_inputs', [])
            for required_input in required_inputs:
                if required_input not in from_outputs:
                    errors.append(f"Stage {to_stage} requires '{required_input}' from {from_stage}")
        
        return errors


def sync_pipeline_config(config: DictConfig) -> DictConfig:
    """
    Convenience function to synchronize pipeline configuration.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Synchronized configuration
    """
    synchronizer = ConfigSynchronizer()
    return synchronizer.sync_pipeline_config(config)


def create_stage_config(base_config: DictConfig, stage: str) -> DictConfig:
    """
    Create configuration for a specific pipeline stage.
    
    Args:
        base_config: Base pipeline configuration
        stage: Stage name
        
    Returns:
        Stage-specific configuration
    """
    synchronizer = ConfigSynchronizer()
    synced_config = synchronizer.sync_pipeline_config(base_config)
    return synchronizer.get_stage_config(synced_config, stage)

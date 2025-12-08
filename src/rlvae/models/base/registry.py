"""
Registry system for model components.

This module provides a simple string-to-class registry with decorators
and Hydra integration for component instantiation.
"""

from typing import Dict, Type, Any, Optional, Callable
from omegaconf import DictConfig
import hydra
from functools import wraps


class ComponentRegistry:
    """Registry for model components."""
    
    def __init__(self):
        self._components: Dict[str, Type] = {}
    
    def register(self, name: str) -> Callable:
        """Decorator to register a component class."""
        def decorator(cls: Type) -> Type:
            if name in self._components:
                raise ValueError(f"Component '{name}' is already registered")
            self._components[name] = cls
            return cls
        return decorator
    
    def get(self, name: str) -> Optional[Type]:
        """Get a component class by name."""
        return self._components.get(name)
    
    def list(self) -> Dict[str, Type]:
        """List all registered components."""
        return self._components.copy()
    
    def build(self, config: DictConfig) -> Any:
        """Build a component from Hydra config."""
        if not isinstance(config, DictConfig):
            raise ValueError("Config must be a DictConfig")
        
        if '_target_' not in config:
            raise ValueError("Config must contain '_target_' field")
        
        target = config._target_
        
        # Try to instantiate directly with Hydra
        try:
            return hydra.utils.instantiate(config)
        except Exception as e:
            # Fallback to registry lookup
            if target in self._components:
                cls = self._components[target]
                # Convert config to dict and remove _target_
                config_dict = dict(config)
                config_dict.pop('_target_', None)
                return cls(**config_dict)
            else:
                raise ValueError(f"Unknown component target: {target}") from e


# Global registry instance
_registry = ComponentRegistry()


def register(name: str) -> Callable:
    """Decorator to register a component."""
    return _registry.register(name)


def get_component(name: str) -> Optional[Type]:
    """Get a component by name."""
    return _registry.get(name)


def list_components() -> Dict[str, Type]:
    """List all registered components."""
    return _registry.list()


def build_component(config: DictConfig) -> Any:
    """Build a component from config."""
    return _registry.build(config)


# Convenience function for backward compatibility
def build_from_config(config: DictConfig) -> Any:
    """Build component from config (alias for build_component)."""
    return build_component(config)

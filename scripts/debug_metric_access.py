#!/usr/bin/env python3
"""
Debug Metric Tensor Access Issues
=================================

Investigate why metric tensor is not available during forward passes.
"""

import os
import sys
from pathlib import Path
import torch
import numpy as np
from omegaconf import DictConfig
import logging

# Setup paths
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir.parent / "src"
lib_src_dir = src_dir / "lib" / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(lib_src_dir) not in sys.path:
    sys.path.insert(0, str(lib_src_dir))

# Project imports
from models.modular_rlvae import ModularRiemannianFlowVAE
from data.cyclic_dataset import CyclicSpritesDataModule

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_model():
    """Load model exactly as before."""
    checkpoint_path = "outputs/checkpoints/epoch=197-val_loss=6.402.ckpt"
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    model_hparams = checkpoint['hyper_parameters']['model']
    
    config = DictConfig({
        'input_dim': model_hparams['input_dim'],
        'latent_dim': model_hparams['latent_dim'],
        'n_flows': model_hparams['n_flows'],
        'flow_hidden_size': model_hparams['flow_hidden_size'],
        'flow_n_blocks': model_hparams['flow_n_blocks'],
        'flow_n_hidden': model_hparams['flow_n_hidden'],
        'epsilon': model_hparams['epsilon'],
        'encoder': model_hparams['encoder'],
        'decoder': model_hparams['decoder'],
        'beta': model_hparams['beta'],
        'riemannian_beta': model_hparams['riemannian_beta'],
        'posterior': model_hparams['posterior'],
        'sampling': model_hparams['sampling'],
        'loop': model_hparams['loop'],
        'metric': model_hparams['metric'],
        'pretrained': {'encoder_path': None, 'decoder_path': None, 'metric_path': None},
        'sequence_length': model_hparams['sequence_length']
    })
    
    model = ModularRiemannianFlowVAE(config)
    
    # Load state dict
    state_dict = checkpoint['state_dict']
    clean_state_dict = {k.replace('model.', '') if k.startswith('model.') else k: v 
                       for k, v in state_dict.items()}
    
    # Resize metric tensor
    for name, param in clean_state_dict.items():
        if 'modular_metric.centroids' in name:
            model.modular_metric.centroids = torch.nn.Parameter(torch.zeros_like(param))
        elif 'modular_metric.metric_matrices' in name:
            model.modular_metric.metric_matrices = torch.nn.Parameter(torch.zeros_like(param))
    
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    
    return model

def debug_metric_access():
    """Debug exactly what happens with metric access."""
    logger.info("🔍 Starting metric access debugging")
    
    model = load_model()
    
    # First, verify metric tensor is loaded
    logger.info("📊 Checking metric tensor after loading:")
    logger.info(f"  - Has modular_metric: {hasattr(model, 'modular_metric')}")
    logger.info(f"  - modular_metric is not None: {model.modular_metric is not None}")
    
    if hasattr(model, 'modular_metric') and model.modular_metric is not None:
        centroids = model.modular_metric.centroids
        matrices = model.modular_metric.metric_matrices
        logger.info(f"  - Centroids shape: {centroids.shape}")
        logger.info(f"  - Matrices shape: {matrices.shape}")
        logger.info(f"  - Centroids requires_grad: {centroids.requires_grad}")
        logger.info(f"  - Matrices requires_grad: {matrices.requires_grad}")
    
    # Check what the model thinks about metric availability
    logger.info("🔍 Checking model's internal metric availability checks:")
    
    # Let's see what posterior type is set
    if hasattr(model, 'posterior_type'):
        logger.info(f"  - Model posterior_type: {model.posterior_type}")
    
    # Check if there are any specific conditions that make metric "unavailable"
    # during forward pass in the RLVAE code
    
    # Create sample data
    data_config = DictConfig({
        'train_path': 'data/sprites/ColoredCircles_train.pt',
        'test_path': 'data/sprites/ColoredCircles_test.pt',
        'train_meta_path': 'data/sprites/ColoredCircles_train_params.pt',
        'test_meta_path': 'data/sprites/ColoredCircles_test_params.pt',
        'sequence_length': 10,
        'image_size': [28, 28],
        'channels': 3,
        'batch_size': 1,
        'num_workers': 0,
        'pin_memory': False,
        'max_test_samples': 10,
        'verify_cyclicity': False
    })
    
    data_module = CyclicSpritesDataModule(data_config)
    data_module.setup('test')
    test_loader = data_module.test_dataloader()
    
    # Get a sample
    batch = next(iter(test_loader))
    if len(batch.shape) == 4:
        batch = batch.unsqueeze(0)
    
    logger.info(f"📊 Sample batch shape: {batch.shape}")
    
    # Debug step by step forward pass
    logger.info("🔍 Debugging step-by-step forward pass:")
    
    with torch.no_grad():
        try:
            # Step 1: Check what happens in the model's forward method
            logger.info("Step 1: Calling model forward...")
            
            # Let's manually trace through what the forward method does
            # Looking at the model structure
            logger.info(f"  - Model type: {type(model)}")
            logger.info(f"  - Model has posterior_type: {hasattr(model, 'posterior_type')}")
            
            if hasattr(model, 'posterior_type'):
                logger.info(f"  - posterior_type value: {model.posterior_type}")
            
            # Check modular_metric before forward
            logger.info("Before forward:")
            logger.info(f"  - modular_metric exists: {hasattr(model, 'modular_metric')}")
            if hasattr(model, 'modular_metric'):
                logger.info(f"  - modular_metric is not None: {model.modular_metric is not None}")
                if model.modular_metric is not None:
                    logger.info(f"  - centroids shape: {model.modular_metric.centroids.shape}")
            
            # Try the forward pass and catch where the warning comes from
            output = model(batch)
            
            logger.info("✅ Forward pass completed")
            logger.info(f"  - Output keys: {list(output.keys()) if isinstance(output, dict) else 'Not dict'}")
            
        except Exception as e:
            logger.error(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()

def check_model_internals():
    """Check the internal structure of the model to understand metric access."""
    logger.info("🔍 Checking model internals")
    
    model = load_model()
    
    # List all attributes
    logger.info("📊 Model attributes:")
    for attr in dir(model):
        if not attr.startswith('_'):
            try:
                value = getattr(model, attr)
                if not callable(value):
                    logger.info(f"  - {attr}: {type(value)} = {value}")
            except:
                logger.info(f"  - {attr}: <could not access>")
    
    # Check specific metric-related attributes
    logger.info("📊 Metric-related attributes:")
    metric_attrs = ['modular_metric', 'posterior_type', 'use_riemannian', 'sampling_method']
    for attr in metric_attrs:
        if hasattr(model, attr):
            value = getattr(model, attr)
            logger.info(f"  - {attr}: {value}")
        else:
            logger.info(f"  - {attr}: <not found>")

if __name__ == "__main__":
    debug_metric_access()
    print("\n" + "="*50 + "\n")
    check_model_internals() 
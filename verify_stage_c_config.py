#!/usr/bin/env python3
"""
Verify that all required Stage C pretrained paths exist before training.
"""

from pathlib import Path
import yaml
import sys

def check_config():
    """Check if all pretrained paths in config.yaml exist."""
    config_path = Path("conf/config.yaml")
    
    if not config_path.exists():
        print(f"❌ Config file not found: {config_path}")
        return False
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Extract pretrained paths (handle different config structures)
    try:
        model_cfg = config.get('model', {})
        if isinstance(model_cfg, dict):
            pretrained = model_cfg.get('pretrained', {})
            if isinstance(pretrained, dict):
                encoder_path = pretrained.get('encoder_path')
                decoder_path = pretrained.get('decoder_path')
            else:
                encoder_path = None
                decoder_path = None
            
            metric = model_cfg.get('metric', {})
            if isinstance(metric, dict):
                metric_path = metric.get('path')
            else:
                metric_path = None
        else:
            encoder_path = None
            decoder_path = None
            metric_path = None
    except Exception as e:
        print(f"⚠️  Error parsing config: {e}")
        encoder_path = None
        decoder_path = None
        metric_path = None
    
    print("🔍 Verifying Stage C Configuration")
    print("=" * 70)
    print()
    
    all_ok = True
    
    # Check encoder
    print("1. Encoder:")
    if encoder_path and encoder_path != 'null':
        path = Path(encoder_path)
        if path.exists():
            print(f"   ✅ {encoder_path}")
            print(f"      Size: {path.stat().st_size / (1024*1024):.1f} MB")
        else:
            print(f"   ❌ NOT FOUND: {encoder_path}")
            all_ok = False
    else:
        print(f"   ⚠️  Not configured (encoder_path: {encoder_path})")
        all_ok = False
    print()
    
    # Check decoder
    print("2. Decoder:")
    if decoder_path and decoder_path != 'null':
        path = Path(decoder_path)
        if path.exists():
            print(f"   ✅ {decoder_path}")
            print(f"      Size: {path.stat().st_size / (1024*1024):.1f} MB")
        else:
            print(f"   ❌ NOT FOUND: {decoder_path}")
            all_ok = False
    else:
        print(f"   ⚠️  Not configured (decoder_path: {decoder_path})")
        all_ok = False
    print()
    
    # Check metric
    print("3. Metric:")
    if metric_path and metric_path != 'null':
        path = Path(metric_path)
        if path.exists():
            print(f"   ✅ {metric_path}")
            print(f"      Size: {path.stat().st_size / 1024:.1f} KB")
            
            # Check if it's rescaled
            if 'rescaled' in metric_path:
                print(f"      🔧 Using RESCALED metric!")
        else:
            print(f"   ❌ NOT FOUND: {metric_path}")
            all_ok = False
    else:
        print(f"   ⚠️  Not configured (metric_path: {metric_path})")
    print()
    
    print("=" * 70)
    if all_ok:
        print("✅ All required Stage C paths are configured and exist!")
        print()
        print("Ready to run:")
        print("  python run_experiment.py stage=C")
        return True
    else:
        print("❌ Some paths are missing or not configured!")
        print()
        print("Fix by updating conf/config.yaml:")
        print("  model:")
        print("    pretrained:")
        print("      encoder_path: outputs/stages/A_VANILLA_MLP_2_SPRITES/encoder_diverse_mlp_ld2_*.pt")
        print("      decoder_path: outputs/stages/A_VANILLA_MLP_2_SPRITES/decoder_diverse_mlp_ld2_*.pt")
        print("    metric:")
        print("      path: outputs/stages/B_RHVAE_MLP_2_SPRITES/metric_rescaled.pt")
        return False

if __name__ == "__main__":
    success = check_config()
    sys.exit(0 if success else 1)


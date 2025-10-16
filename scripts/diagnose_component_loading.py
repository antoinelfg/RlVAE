#!/usr/bin/env python3
"""
Component Loading Diagnostics
============================

Diagnose and fix component loading issues in the three-stage pipeline.
This script identifies and corrects problems with Stage A component loading
into Stage C (RLVAE).
"""

import os
import sys
from pathlib import Path
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComponentLoadingDiagnostics:
    """Diagnose and fix component loading issues."""
    
    def __init__(self, project_root: Optional[Path] = None):
        """Initialize diagnostics with project root."""
        if project_root is None:
            project_root = Path.cwd()
            if not (project_root / 'src').exists():
                project_root = project_root.parent
        
        self.project_root = project_root
        self.stages_dir = project_root / 'outputs' / 'stages'
        self.config_dir = project_root / 'conf'
        
        logger.info(f"🔧 Component loading diagnostics initialized")
        logger.info(f"📁 Project root: {self.project_root}")
        logger.info(f"📁 Stages dir: {self.stages_dir}")
    
    def analyze_stage_components(self, architecture: str, latent_dim: int) -> Dict[str, Any]:
        """Analyze available Stage A components for given architecture and latent_dim."""
        logger.info(f"🔍 Analyzing Stage A components for {architecture}_ld{latent_dim}")
        
        # Find all possible Stage A folders
        possible_folders = [
            f"A_RHVAE_{architecture.upper()}_{latent_dim}_SPRITES",
            f"A_VANILLA_{architecture.upper()}_{latent_dim}_SPRITES",
            f"A_VANILLA_{architecture.upper()}_GRAY_{latent_dim}_SPRITES"
        ]
        
        found_components = {}
        
        for folder_name in possible_folders:
            stage_a_dir = self.stages_dir / folder_name
            if stage_a_dir.exists():
                logger.info(f"📂 Found Stage A folder: {folder_name}")
                
                # Analyze components
                components = self._analyze_stage_a_folder(stage_a_dir)
                if components:
                    found_components[folder_name] = components
        
        return found_components
    
    def _analyze_stage_a_folder(self, stage_a_dir: Path) -> Optional[Dict[str, Any]]:
        """Analyze a specific Stage A folder for components."""
        try:
            # Load config
            config_path = stage_a_dir / 'config.yaml'
            if not config_path.exists():
                logger.warning(f"⚠️ No config.yaml in {stage_a_dir}")
                return None
            
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            # Find encoder/decoder files
            encoder_files = list(stage_a_dir.glob('encoder*.pt')) + list(stage_a_dir.glob('encoder*.pkl'))
            decoder_files = list(stage_a_dir.glob('decoder*.pt')) + list(stage_a_dir.glob('decoder*.pkl'))
            
            if not encoder_files or not decoder_files:
                logger.warning(f"⚠️ Missing encoder/decoder files in {stage_a_dir}")
                return None
            
            # Get most recent files
            encoder_path = max(encoder_files, key=lambda p: p.stat().st_mtime)
            decoder_path = max(decoder_files, key=lambda p: p.stat().st_mtime)
            
            # Validate component compatibility
            encoder_info = self._validate_component(encoder_path, 'encoder')
            decoder_info = self._validate_component(decoder_path, 'decoder')
            
            return {
                'config': config,
                'encoder_path': encoder_path,
                'decoder_path': decoder_path,
                'encoder_info': encoder_info,
                'decoder_info': decoder_info,
                'timestamp': datetime.fromtimestamp(encoder_path.stat().st_mtime).isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to analyze {stage_a_dir}: {e}")
            return None
    
    def _validate_component(self, component_path: Path, component_type: str) -> Dict[str, Any]:
        """Validate a component file and extract metadata."""
        try:
            # Load the component
            component_data = torch.load(component_path, map_location='cpu', weights_only=False)
            
            # Extract state dict
            if hasattr(component_data, 'state_dict'):
                state_dict = component_data.state_dict()
            else:
                state_dict = component_data
            
            # Analyze the state dict
            info = {
                'path': str(component_path),
                'size_mb': component_path.stat().st_size / (1024 * 1024),
                'num_parameters': sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor)),
                'keys': list(state_dict.keys())[:10],  # First 10 keys
                'compatible': True,
                'issues': []
            }
            
            # Check for common issues
            if not state_dict:
                info['compatible'] = False
                info['issues'].append("Empty state dict")
            
            # Check for architecture-specific patterns
            if component_type == 'encoder':
                if not any('encoder' in key.lower() or 'fc' in key.lower() for key in state_dict.keys()):
                    info['issues'].append("No encoder-like layers found")
            
            elif component_type == 'decoder':
                if not any('decoder' in key.lower() or 'fc' in key.lower() for key in state_dict.keys()):
                    info['issues'].append("No decoder-like layers found")
            
            return info
            
        except Exception as e:
            return {
                'path': str(component_path),
                'compatible': False,
                'issues': [f"Failed to load: {e}"]
            }
    
    def diagnose_stage_c_loading(self, architecture: str, latent_dim: int) -> Dict[str, Any]:
        """Diagnose Stage C component loading issues."""
        logger.info(f"🔍 Diagnosing Stage C loading for {architecture}_ld{latent_dim}")
        
        # Find Stage C folder
        stage_c_folder = f"C_RLVAE_{architecture.upper()}_{latent_dim}_SPRITES"
        stage_c_dir = self.stages_dir / stage_c_folder
        
        if not stage_c_dir.exists():
            logger.warning(f"⚠️ Stage C folder not found: {stage_c_folder}")
            return {'status': 'not_found', 'stage_c_dir': stage_c_dir}
        
        # Load Stage C config
        config_path = stage_c_dir / 'config.yaml'
        if not config_path.exists():
            logger.warning(f"⚠️ No Stage C config found")
            return {'status': 'no_config', 'stage_c_dir': stage_c_dir}
        
        with open(config_path, 'r') as f:
            stage_c_config = yaml.safe_load(f)
        
        # Analyze what Stage A components should be used
        stage_a_components = self.analyze_stage_components(architecture, latent_dim)
        
        # Check for loading issues
        issues = []
        recommendations = []
        
        if not stage_a_components:
            issues.append("No Stage A components found")
            recommendations.append("Run Stage A first to generate required components")
        else:
            # Check if the right components are being used
            stage_a_source = stage_c_config.get('stage_a_source', '')
            if stage_a_source:
                expected_folder = Path(stage_a_source).name
                if expected_folder not in stage_a_components:
                    issues.append(f"Stage C expects {expected_folder} but it's not available")
                    recommendations.append(f"Use available components: {list(stage_a_components.keys())}")
        
        return {
            'status': 'analyzed',
            'stage_c_dir': stage_c_dir,
            'stage_c_config': stage_c_config,
            'stage_a_components': stage_a_components,
            'issues': issues,
            'recommendations': recommendations
        }
    
    def fix_component_loading(self, architecture: str, latent_dim: int, 
                            preferred_model_type: str = 'RHVAE') -> Dict[str, Any]:
        """Fix component loading issues by ensuring correct components are used."""
        logger.info(f"🔧 Fixing component loading for {architecture}_ld{latent_dim}")
        
        # Analyze current state
        diagnosis = self.diagnose_stage_c_loading(architecture, latent_dim)
        
        if diagnosis['status'] != 'analyzed':
            return {'status': 'failed', 'reason': f"Cannot fix: {diagnosis['status']}"}
        
        # Find the best Stage A components
        stage_a_components = diagnosis['stage_a_components']
        
        if not stage_a_components:
            return {'status': 'failed', 'reason': 'No Stage A components available'}
        
        # Select the best components based on preference
        best_components = None
        for folder_name, components in stage_a_components.items():
            if preferred_model_type.upper() in folder_name:
                best_components = components
                break
        
        if not best_components:
            # Fallback to any available components
            best_components = list(stage_a_components.values())[0]
            logger.warning(f"⚠️ Using fallback components from {list(stage_a_components.keys())[0]}")
        
        # Create corrected Stage C config
        stage_c_config = diagnosis['stage_c_config'].copy()
        stage_c_config['stage_a_source'] = str(best_components['encoder_path'].parent)
        
        # Save corrected config
        config_path = diagnosis['stage_c_dir'] / 'config.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(stage_c_config, f, default_flow_style=False)
        
        logger.info(f"✅ Fixed Stage C config to use: {stage_c_config['stage_a_source']}")
        
        return {
            'status': 'fixed',
            'stage_a_source': stage_c_config['stage_a_source'],
            'components_used': {
                'encoder': str(best_components['encoder_path']),
                'decoder': str(best_components['decoder_path'])
            }
        }
    
    def generate_report(self, architecture: str, latent_dim: int) -> str:
        """Generate a comprehensive diagnostic report."""
        logger.info(f"📊 Generating diagnostic report for {architecture}_ld{latent_dim}")
        
        # Analyze components
        stage_a_components = self.analyze_stage_components(architecture, latent_dim)
        stage_c_diagnosis = self.diagnose_stage_c_loading(architecture, latent_dim)
        
        # Generate report
        report = []
        report.append(f"# Component Loading Diagnostic Report")
        report.append(f"**Generated:** {datetime.now().isoformat()}")
        report.append(f"**Architecture:** {architecture}")
        report.append(f"**Latent Dimension:** {latent_dim}")
        report.append("")
        
        # Stage A Components
        report.append("## Stage A Components Found")
        if stage_a_components:
            for folder_name, components in stage_a_components.items():
                report.append(f"### {folder_name}")
                report.append(f"- **Encoder:** {components['encoder_path'].name}")
                report.append(f"- **Decoder:** {components['decoder_path'].name}")
                report.append(f"- **Timestamp:** {components['timestamp']}")
                report.append(f"- **Model Type:** {components['config'].get('model_type', 'Unknown')}")
                report.append(f"- **Latent Dim:** {components['config'].get('latent_dim', 'Unknown')}")
                report.append("")
        else:
            report.append("❌ No Stage A components found")
            report.append("")
        
        # Stage C Analysis
        report.append("## Stage C Analysis")
        if stage_c_diagnosis['status'] == 'analyzed':
            report.append(f"**Stage C Directory:** {stage_c_diagnosis['stage_c_dir']}")
            report.append(f"**Expected Stage A Source:** {stage_c_diagnosis['stage_c_config'].get('stage_a_source', 'Not specified')}")
            
            if stage_c_diagnosis['issues']:
                report.append("### Issues Found:")
                for issue in stage_c_diagnosis['issues']:
                    report.append(f"- ❌ {issue}")
                report.append("")
            
            if stage_c_diagnosis['recommendations']:
                report.append("### Recommendations:")
                for rec in stage_c_diagnosis['recommendations']:
                    report.append(f"- 💡 {rec}")
                report.append("")
        else:
            report.append(f"❌ Stage C analysis failed: {stage_c_diagnosis['status']}")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main diagnostic function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Diagnose component loading issues')
    parser.add_argument('--architecture', type=str, default='mlp', help='Architecture type')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent dimension')
    parser.add_argument('--fix', action='store_true', help='Attempt to fix issues automatically')
    parser.add_argument('--preferred_model', type=str, default='RHVAE', 
                       choices=['RHVAE', 'VANILLA'], help='Preferred Stage A model type')
    parser.add_argument('--report', action='store_true', help='Generate diagnostic report')
    
    args = parser.parse_args()
    
    # Initialize diagnostics
    diagnostics = ComponentLoadingDiagnostics()
    
    # Generate report
    if args.report:
        report = diagnostics.generate_report(args.architecture, args.latent_dim)
        print(report)
        
        # Save report
        report_path = diagnostics.project_root / f'component_loading_report_{args.architecture}_ld{args.latent_dim}.md'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"📄 Report saved to: {report_path}")
    
    # Fix issues if requested
    if args.fix:
        result = diagnostics.fix_component_loading(
            args.architecture, 
            args.latent_dim, 
            args.preferred_model
        )
        
        if result['status'] == 'fixed':
            logger.info("✅ Component loading issues fixed!")
            logger.info(f"📁 Using Stage A source: {result['stage_a_source']}")
        else:
            logger.error(f"❌ Failed to fix issues: {result.get('reason', 'Unknown error')}")

if __name__ == "__main__":
    main()

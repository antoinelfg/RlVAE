#!/usr/bin/env python3
"""
WandB Sweep Runner for RLVAE Hyperparameter Optimization
========================================================

This script runs WandB sweeps for hyperparameter optimization of the RLVAE pipeline.
It integrates seamlessly with Hydra configuration management.

Usage Examples:
--------------

1. Run comprehensive sweep:
   python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep

2. Run architecture optimization:
   python scripts/run_sweep.py --sweep-config architecture_optimization --agent-count 4

3. Run learning rate optimization:
   python scripts/run_sweep.py --sweep-config learning_rate_optimization --max-runs 50

4. Dry run to validate configuration:
   python scripts/run_sweep.py --sweep-config comprehensive_hyperparameter_sweep --dry-run

Features:
--------
- Automatic WandB sweep initialization
- Integration with Hydra configuration system
- Support for multiple sweep agents
- Comprehensive logging and error handling
- Resume capability for interrupted sweeps
- Real-time progress monitoring
"""

import os
import sys
import argparse
import json
import yaml
import time
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

# Add project root to path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

try:
    import wandb
    from omegaconf import DictConfig, OmegaConf
    import hydra
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
except ImportError as e:
    print(f"❌ Missing required packages: {e}")
    print("Please install: wandb omegaconf hydra-core")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sweep_runner.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


class SweepRunner:
    """Main class for running WandB sweeps with Hydra integration."""
    
    def __init__(
        self, 
        sweep_config_name: str,
        project_name: Optional[str] = None,
        entity: Optional[str] = None,
        dry_run: bool = False
    ):
        self.sweep_config_name = sweep_config_name
        self.project_name = project_name or "rlvae-hyperparameter-optimization"
        self.entity = entity
        self.dry_run = dry_run
        
        # Load sweep configuration
        self.sweep_config = self._load_sweep_config()
        
        # Initialize WandB
        self.sweep_id = None
        self.agents = []
        
        logger.info(f"🚀 SweepRunner initialized")
        logger.info(f"   Sweep config: {sweep_config_name}")
        logger.info(f"   Project: {self.project_name}")
        logger.info(f"   Dry run: {dry_run}")
    
    def _load_sweep_config(self) -> Dict[str, Any]:
        """Load sweep configuration from Hydra config."""
        config_path = project_root / "conf" / "sweep" / f"{self.sweep_config_name}.yaml"
        
        if not config_path.exists():
            raise FileNotFoundError(f"Sweep config not found: {config_path}")
        
        logger.info(f"📋 Loading sweep config: {config_path}")
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Remove @package directive if present
        if '@package' in config:
            del config['@package']
        
        return config
    
    def _create_wandb_sweep_config(self) -> Dict[str, Any]:
        """Convert Hydra sweep config to WandB format."""
        wandb_config = {
            'name': self.sweep_config.get('name', 'rlvae_sweep'),
            'description': self.sweep_config.get('description', 'RLVAE hyperparameter sweep'),
            'method': self.sweep_config.get('method', 'random'),
            'parameters': self.sweep_config.get('parameters', {}),
            'metric': self.sweep_config.get('objective', self.sweep_config.get('metric', {'metric': 'val_loss', 'goal': 'minimize'}))
        }
        
        # Add early termination if specified
        if 'early_terminate' in self.sweep_config:
            wandb_config['early_terminate'] = self.sweep_config['early_terminate']
        
        # Add run cap if specified
        if 'run_cap' in self.sweep_config:
            wandb_config['run_cap'] = self.sweep_config['run_cap']
        
        logger.info(f"🔧 WandB sweep config created:")
        logger.info(f"   Method: {wandb_config['method']}")
        logger.info(f"   Parameters: {len(wandb_config['parameters'])} parameters")
        logger.info(f"   Metric: {wandb_config['metric']}")
        
        return wandb_config
    
    def create_sweep(self) -> str:
        """Create WandB sweep and return sweep ID."""
        if self.dry_run:
            logger.info("🧪 DRY RUN: Would create WandB sweep with config:")
            logger.info(json.dumps(self._create_wandb_sweep_config(), indent=2))
            return "dry_run_sweep_id"
        
        wandb_config = self._create_wandb_sweep_config()
        
        logger.info("🎯 Creating WandB sweep...")
        
        try:
            self.sweep_id = wandb.sweep(
                sweep=wandb_config,
                project=self.project_name,
                entity=self.entity
            )
            
            logger.info(f"✅ Sweep created successfully!")
            logger.info(f"   Sweep ID: {self.sweep_id}")
            logger.info(f"   URL: https://wandb.ai/{self.entity or 'your-entity'}/{self.project_name}/sweeps/{self.sweep_id}")
            
            return self.sweep_id
            
        except Exception as e:
            logger.error(f"❌ Failed to create sweep: {e}")
            raise
    
    def _train_function(self):
        """Training function for WandB sweep agent."""
        # This function will be called by the WandB agent
        # with different hyperparameter configurations
        
        # Initialize WandB run with extended timeout for the sweep
        run = wandb.init(settings=wandb.Settings(init_timeout=300))
        
        # Get hyperparameters from WandB
        config = wandb.config
        
        logger.info(f"🏃 Starting sweep run with config:")
        for key, value in config.items():
            logger.info(f"   {key}: {value}")
        
        try:
            # Create Hydra overrides from WandB config
            overrides = []
            
            # Parameter mapping from WandB names to Hydra paths
            param_mapping = {
                # Training parameters
                'lr': 'training.optimizer.lr',
                'weight_decay': 'training.optimizer.weight_decay',
                'batch_size': 'training.data.batch_size',
                'max_epochs': 'training.trainer.max_epochs',
                'n_train_samples': 'training.n_train_samples',
                'n_val_samples': 'training.n_val_samples',
                
                # Model parameters
                'beta': 'model.beta',
                'riemannian_beta': 'model.riemannian_beta',
                'n_flows': 'model.n_flows',
                'flow_hidden_size': 'model.flow_hidden_size',
                'flow_n_blocks': 'model.flow_n_blocks',
                'sampling_method': 'model.sampling.method',
                'loop_mode': 'model.loop.mode',
                'loop_penalty': 'model.loop.penalty',
                'posterior_type': 'model.posterior.type',
                'temperature_override': 'model.metric.temperature_override',
                
                # Stage 1 parameters (experiment specific)
                'stage1_architecture': 'experiment.stage1.architecture',
                'stage1_latent_dim': 'experiment.stage1.latent_dim',
                'stage1_epochs': 'experiment.stage1.epochs',
                'stage1_temperature': 'experiment.stage1.temperature',
                'stage1_regularization': 'experiment.stage1.regularization',
                'stage1_preset': 'experiment.stage1.preset',
                
                # Stage 2 parameters (experiment specific)
                'stage2_model': 'experiment.stage2.model',
            }
            
            for key, value in config.items():
                # Map WandB parameter names to Hydra paths
                hydra_key = param_mapping.get(key, key)
                
                if isinstance(value, (str, int, float, bool)):
                    overrides.append(f"{hydra_key}={value}")
                else:
                    # Handle complex parameters
                    overrides.append(f"{hydra_key}={str(value)}")
            
            # Add fixed parameters
            fixed_params = self.sweep_config.get('parameters_fixed', {})
            for key, value in fixed_params.items():
                overrides.append(f"{key}={value}")
            
            # Always use pipeline experiment
            overrides.append("experiment=global_vanilla_rlvae_pipeline")
            
            # CRITICAL FIX: Disable WandB in subprocess to avoid double initialization
            overrides.append("wandb.mode=disabled")
            
            logger.info(f"🔧 Hydra overrides: {overrides}")
            
            # Create environment for subprocess - inherit current env but add sweep info
            env = os.environ.copy()
            # Let the subprocess know it's running in a sweep context
            env['WANDB_RUN_GROUP'] = f"sweep_{self.sweep_id}"
            env['WANDB_JOB_TYPE'] = "sweep_run"
            
            # Run the experiment using subprocess to avoid Hydra conflicts
            cmd = [
                sys.executable, "run_experiment.py"
            ] + overrides
            
            logger.info(f"🚀 Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                cwd=project_root,
                env=env,  # Pass the modified environment
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("✅ Experiment completed successfully")
                
                # Parse metrics from subprocess output with improved patterns
                output_lines = result.stdout.split('\n')
                metrics = {}
                
                logger.info("🔍 Parsing metrics from subprocess output...")
                
                for line in output_lines:
                    line = line.strip()
                    
                    # Lightning trainer validation metrics (these are most reliable)
                    if 'val_loss=' in line and 'epoch=' in line:
                        try:
                            # Parse Lightning progress bar output like: "Epoch 49: 100%|██| 12/12 [00:XX<00:00,  X.XXit/s, train_loss=X.XXX, val_loss=X.XXX]"
                            import re
                            val_loss_match = re.search(r'val_loss=([0-9.]+)', line)
                            train_loss_match = re.search(r'train_loss=([0-9.]+)', line)
                            epoch_match = re.search(r'epoch=([0-9]+)', line)
                            
                            if val_loss_match:
                                metrics['val_loss'] = float(val_loss_match.group(1))
                            if train_loss_match:
                                metrics['train_loss'] = float(train_loss_match.group(1))
                            if epoch_match:
                                metrics['epoch'] = int(epoch_match.group(1))
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Failed to parse Lightning progress line: {e}")
                    
                    # Test results (final evaluation)
                    elif 'test_loss' in line.lower() or 'test results' in line.lower():
                        try:
                            import re
                            test_loss_match = re.search(r'(?:test_loss|test/loss)[:\s=]+([0-9.]+)', line, re.IGNORECASE)
                            if test_loss_match:
                                metrics['test_loss'] = float(test_loss_match.group(1))
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Failed to parse test loss: {e}")
                    
                    # Pipeline stage completion messages
                    elif '[Stage 1] Epoch' in line and 'Val Loss:' in line:
                        try:
                            # Parse: "[Stage 1] Epoch X/Y - Train Loss: X.XXXX | Val Loss: X.XXXX"
                            import re
                            val_loss_match = re.search(r'Val Loss:\s*([0-9.]+)', line)
                            train_loss_match = re.search(r'Train Loss:\s*([0-9.]+)', line)
                            if val_loss_match:
                                metrics['stage1_val_loss'] = float(val_loss_match.group(1))
                            if train_loss_match:
                                metrics['stage1_train_loss'] = float(train_loss_match.group(1))
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Failed to parse stage 1 metrics: {e}")
                    
                    # Look for completion messages
                    elif '✅' in line and ('completed' in line.lower() or 'finished' in line.lower()):
                        logger.info(f"📋 Found completion message: {line}")
                    
                    # Log any error or warning messages for debugging
                    elif any(marker in line for marker in ['❌', '⚠️', 'ERROR', 'WARNING', 'Failed']):
                        logger.warning(f"⚠️ Found warning/error in output: {line}")
                
                # Also look for final summary metrics in a more general way
                summary_section = False
                for line in output_lines:
                    line = line.strip()
                    
                    # Look for summary sections
                    if 'experiment completed' in line.lower() or 'final' in line.lower():
                        summary_section = True
                    elif summary_section and any(keyword in line.lower() for keyword in ['loss:', 'accuracy:', 'mse:', 'mae:']):
                        try:
                            # Parse "metric_name: value" patterns
                            import re
                            metric_match = re.search(r'([a-zA-Z_]+):\s*([0-9.]+)', line)
                            if metric_match:
                                metric_name = metric_match.group(1).lower()
                                metric_value = float(metric_match.group(2))
                                if 'final' not in metric_name:
                                    metric_name = f"final_{metric_name}"
                                metrics[metric_name] = metric_value
                        except (ValueError, AttributeError) as e:
                            logger.debug(f"Failed to parse summary metric: {e}")
                
                # Add subprocess metadata
                metrics['subprocess_returncode'] = 0
                metrics['subprocess_success'] = True
                
                # Log metrics to the sweep run
                if metrics:
                    wandb.log(metrics)
                    logger.info(f"📊 Logged {len(metrics)} metrics to sweep: {metrics}")
                else:
                    logger.warning("⚠️ No metrics found in subprocess output")
                    logger.info("📝 First 20 lines of subprocess output for debugging:")
                    for i, line in enumerate(output_lines[:20]):
                        logger.info(f"  {i+1:2d}: {line}")
                    
                    # Log a basic success metric even if we can't parse specific metrics
                    wandb.log({
                        'subprocess_returncode': 0,
                        'subprocess_success': True,
                        'parsing_failed': True
                    })
                    
            else:
                logger.error(f"❌ Experiment failed with return code {result.returncode}")
                logger.error(f"STDOUT: {result.stdout}")
                logger.error(f"STDERR: {result.stderr}")
                
                # Log error to WandB
                wandb.log({
                    "error": f"Experiment failed: {result.stderr}", 
                    "returncode": result.returncode,
                    "subprocess_success": False
                })
                
        except subprocess.TimeoutExpired:
            logger.error("⏰ Experiment timed out")
            wandb.log({"error": "Experiment timed out"})
            
        except Exception as e:
            logger.error(f"❌ Experiment error: {e}")
            wandb.log({"error": str(e)})
            
        finally:
            # Finish WandB run
            wandb.finish()
    
    def run_agent(self, count: Optional[int] = None):
        """Run WandB sweep agent."""
        if not self.sweep_id:
            raise ValueError("No sweep ID available. Create sweep first.")
        
        if self.dry_run:
            logger.info(f"🧪 DRY RUN: Would run sweep agent for sweep {self.sweep_id}")
            return
        
        logger.info(f"🤖 Starting sweep agent...")
        logger.info(f"   Sweep ID: {self.sweep_id}")
        logger.info(f"   Count: {count or 'unlimited'}")
        
        try:
            wandb.agent(
                sweep_id=self.sweep_id,
                function=self._train_function,
                count=count,
                project=self.project_name,
                entity=self.entity
            )
            
            logger.info("✅ Sweep agent completed")
            
        except KeyboardInterrupt:
            logger.info("🛑 Sweep agent interrupted by user")
            
        except Exception as e:
            logger.error(f"❌ Sweep agent error: {e}")
            raise
    
    def run_multiple_agents(self, agent_count: int = 2, runs_per_agent: Optional[int] = None):
        """Run multiple sweep agents in parallel."""
        if self.dry_run:
            logger.info(f"🧪 DRY RUN: Would run {agent_count} parallel agents")
            return
        
        logger.info(f"🔥 Starting {agent_count} parallel sweep agents...")
        
        import multiprocessing as mp
        
        def run_agent_process(agent_id: int):
            """Run a single agent process."""
            logger.info(f"🤖 Agent {agent_id} starting...")
            
            try:
                wandb.agent(
                    sweep_id=self.sweep_id,
                    function=self._train_function,
                    count=runs_per_agent,
                    project=self.project_name,
                    entity=self.entity
                )
                logger.info(f"✅ Agent {agent_id} completed")
                
            except Exception as e:
                logger.error(f"❌ Agent {agent_id} error: {e}")
        
        # Start agent processes
        processes = []
        for i in range(agent_count):
            p = mp.Process(target=run_agent_process, args=(i,))
            p.start()
            processes.append(p)
        
        # Wait for all agents to complete
        try:
            for p in processes:
                p.join()
            logger.info("✅ All sweep agents completed")
            
        except KeyboardInterrupt:
            logger.info("🛑 Terminating all agents...")
            for p in processes:
                p.terminate()
                p.join()
    
    def get_sweep_status(self) -> Dict[str, Any]:
        """Get current sweep status."""
        if not self.sweep_id or self.dry_run:
            return {"status": "dry_run"}
        
        try:
            api = wandb.Api()
            sweep = api.sweep(f"{self.entity or 'your-entity'}/{self.project_name}/sweeps/{self.sweep_id}")
            
            return {
                "sweep_id": self.sweep_id,
                "state": sweep.state,
                "run_count": len(sweep.runs),
                "best_loss": sweep.best_run().summary.get('val_loss', None) if sweep.best_run() else None,
                "url": sweep.url
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get sweep status: {e}")
            return {"error": str(e)}


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run WandB hyperparameter sweeps for RLVAE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--sweep-config", "-c",
        type=str,
        required=True,
        help="Name of sweep configuration (without .yaml extension)"
    )
    
    parser.add_argument(
        "--project", "-p",
        type=str,
        default="rlvae-hyperparameter-optimization",
        help="WandB project name"
    )
    
    parser.add_argument(
        "--entity", "-e",
        type=str,
        help="WandB entity (username or team)"
    )
    
    parser.add_argument(
        "--agent-count", "-n",
        type=int,
        default=1,
        help="Number of parallel agents to run"
    )
    
    parser.add_argument(
        "--max-runs", "-m",
        type=int,
        help="Maximum number of runs per agent"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running sweep"
    )
    
    parser.add_argument(
        "--sweep-id",
        type=str,
        help="Resume existing sweep by ID"
    )
    
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Only show sweep status (requires --sweep-id)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info(f"🔬 Starting RLVAE Hyperparameter Sweep")
    logger.info(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"   Sweep config: {args.sweep_config}")
    logger.info(f"   Project: {args.project}")
    
    try:
        # Create sweep runner
        runner = SweepRunner(
            sweep_config_name=args.sweep_config,
            project_name=args.project,
            entity=args.entity,
            dry_run=args.dry_run
        )
        
        # Handle existing sweep
        if args.sweep_id:
            runner.sweep_id = args.sweep_id
            
            if args.status_only:
                status = runner.get_sweep_status()
                logger.info(f"📊 Sweep Status:")
                for key, value in status.items():
                    logger.info(f"   {key}: {value}")
                return
        else:
            # Create new sweep
            sweep_id = runner.create_sweep()
            logger.info(f"🎯 Sweep created: {sweep_id}")
        
        # Run sweep agents (skip for dry run)
        if not args.dry_run:
            if args.agent_count > 1:
                runner.run_multiple_agents(
                    agent_count=args.agent_count,
                    runs_per_agent=args.max_runs
                )
            else:
                runner.run_agent(count=args.max_runs)
        else:
            logger.info("🧪 DRY RUN: Skipping agent execution")
        
        # Final status
        if not args.dry_run:
            final_status = runner.get_sweep_status()
            logger.info(f"🏁 Final Sweep Status:")
            for key, value in final_status.items():
                logger.info(f"   {key}: {value}")
    
    except KeyboardInterrupt:
        logger.info("🛑 Sweep interrupted by user")
    except Exception as e:
        logger.error(f"❌ Sweep failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 
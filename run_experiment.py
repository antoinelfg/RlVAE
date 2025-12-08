from src.enhanced_component_loader import EnhancedComponentLoader
"""
RlVAE Experiment Runner
======================

Hydra-powered experiment runner for systematic comparison of Riemannian VAE variants.

Usage Examples:
--------------

1. Single experiment:
   python run_experiment.py

2. Quick development test:
   python run_experiment.py training=quick visualization=minimal

3. Compare models:
   python run_experiment.py experiment=comparison_study

4. Hyperparameter sweep:
   python run_experiment.py experiment=hyperparameter_sweep -m

5. Custom configuration:
   python run_experiment.py model=vanilla_vae training.n_epochs=50 wandb.mode=offline

6. Override specific parameters:
   python run_experiment.py model.riemannian_beta=10.0 training.learning_rate=1e-3
"""

import os
import sys
import math
from pathlib import Path

# Add src to path
current_dir = Path(__file__).parent.absolute()
src_dir = current_dir / "src"
lib_src_dir = src_dir / "lib" / "src"

if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))
if str(lib_src_dir) not in sys.path:
    sys.path.insert(0, str(lib_src_dir))

import torch
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import wandb
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from datetime import datetime
from typing import Dict, Any, List, Optional
import yaml

# Local imports
from utils.reproducibility import configure_for_experiment
try:
    from rlvae.models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory, MetricsCollector
except Exception:
    from models.modular_rlvae import ModularRiemannianFlowVAE, ModelFactory, MetricsCollector
from data.datamodule_factory import build_data_module
try:
    from rlvae.training.lightning_trainer import LightningRlVAETrainer
except Exception:
    from training.lightning_trainer import LightningRlVAETrainer
from visualizations.manager import VisualizationManager
try:
    from rlvae.models.components.metric_utils import compute_metric_weights, normalize_metric_atoms, half_logdet_volume
except Exception:
    from models.components.metric_utils import compute_metric_weights, normalize_metric_atoms
try:
    from rlvae.utils.stagec_debugger import stagec_debugger
except Exception:
    from utils.stagec_debugger import stagec_debugger


class ExperimentRunner:
    """Main experiment runner with Hydra configuration."""
    
    @staticmethod
    def _set_config_value(cfg: DictConfig, path: str, value: Any) -> None:
        """Update a configuration value and keep settings.* in sync."""
        OmegaConf.update(cfg, path, value, force_add=True)
        if hasattr(cfg, 'settings'):
            root, *rest = path.split('.', 1)
            if hasattr(cfg.settings, root):
                settings_path = f"settings.{root}"
                if rest:
                    settings_path = f"{settings_path}.{rest[0]}"
                OmegaConf.update(cfg, settings_path, value, force_add=True)

    @staticmethod
    def _set_config_value_if_missing(cfg: DictConfig, path: str, value: Any) -> bool:
        """
        Update a configuration value only when it is missing or None so that
        user-provided overrides (e.g., in the main mega config) take precedence.
        
        Returns:
            bool: True if the value was written, False otherwise.
        """
        try:
            if OmegaConf.is_missing(cfg, path):
                ExperimentRunner._set_config_value(cfg, path, value)
                return True
        except Exception:
            pass
        try:
            current = OmegaConf.select(cfg, path)
        except Exception:
            current = None
        if current is None:
            ExperimentRunner._set_config_value(cfg, path, value)
            return True
        # Ensure compatibility mirrors exist even when we didn't mutate the value.
        if hasattr(cfg, 'settings'):
            root, *rest = path.split('.', 1)
            if hasattr(cfg.settings, root):
                settings_path = f"settings.{root}"
                if rest:
                    settings_path = f"{settings_path}.{rest[0]}"
                try:
                    OmegaConf.select(cfg, settings_path)
                except Exception:
                    # Populate the mirror with the current value without overriding.
                    try:
                        cloned = OmegaConf.to_container(current, resolve=False)
                    except Exception:
                        cloned = current
                    OmegaConf.update(cfg, settings_path, cloned, force_add=True)
        return False

    def _ensure_training_aliases(self, config: Optional[DictConfig] = None) -> None:
        """Ensure settings.training.* mirrors exist even in single-stage runs."""
        cfg = config if config is not None else self.config
        if cfg is None or not hasattr(cfg, 'training') or not hasattr(cfg, 'settings'):
            return
        try:
            training_settings = getattr(cfg.settings, 'training')
        except Exception:
            training_settings = None
        if training_settings is None:
            OmegaConf.update(cfg, "settings.training", {}, force_add=True)
        alias_keys = (
            'trainer',
            'optimizer',
            'scheduler',
            'early_stopping',
            'logging',
            'monitoring',
        )
        for key in alias_keys:
            try:
                cfg_value = getattr(cfg.training, key)
            except Exception:
                continue
            if cfg_value is None:
                continue
            settings_path = f"settings.training.{key}"
            try:
                existing = OmegaConf.select(cfg, settings_path)
                if existing is not None:
                    continue
            except Exception:
                pass
            try:
                    cloned = OmegaConf.to_container(cfg_value, resolve=False)
            except Exception:
                cloned = cfg_value
            OmegaConf.update(cfg, settings_path, cloned, force_add=True)

    def _maybe_run_data_analysis(self, data_module=None, wandb_run=None) -> None:
        """Optionally build a quick ellipse data overview panel before training."""
        cfg_da = getattr(self.config, "data_analysis", None)
        if cfg_da is None or not bool(getattr(cfg_da, "enabled", False)):
            return
        if "data_analysis" in self.results:
            return
        try:
            from data.ellipse_analysis import generate_ellipse_data_report

            dm = data_module
            if dm is None:
                dm = build_data_module(self.config.data)
                dm.setup("fit", getattr(self.config, "training", None))

            root_dir = getattr(cfg_da, "output_dir", None)
            if root_dir is None or ("${" in str(root_dir)):
                try:
                    root_dir = getattr(self.config.output, "run_dir", None)
                except Exception:
                    root_dir = None
            base_dir = Path(root_dir) if root_dir else self.output_dir
            out_dir = base_dir / "data_analysis"

            report = generate_ellipse_data_report(
                cfg=self.config,
                data_module=dm,
                output_dir=out_dir,
                max_sequences=int(getattr(cfg_da, "max_sequences", 256)),
                pairwise_pairs=int(getattr(cfg_da, "pairwise_pairs", 120)),
                preview_sequences=int(getattr(cfg_da, "preview_sequences", 9)),
                log_to_wandb=bool(getattr(cfg_da, "log_to_wandb", False) and wandb_run is not None),
            )
            self.results["data_analysis"] = report
        except Exception as e:
            print(f"[DATA ANALYSIS] ⚠️ Dataset overview failed: {e}")

    @staticmethod
    def _get_wandb_artifacts_cfg(cfg: DictConfig) -> Optional[DictConfig]:
        """Safely retrieve the wandb.artifacts subsection if it exists."""
        if cfg is None:
            return None
        try:
            wandb_cfg = cfg.wandb
        except Exception:
            return None
        if wandb_cfg is None:
            return None
        if isinstance(wandb_cfg, DictConfig):
            return wandb_cfg.get('artifacts', None)
        return getattr(wandb_cfg, 'artifacts', None)

    @staticmethod
    def _wandb_artifacts_enabled(cfg: DictConfig) -> bool:
        """Return True when wandb artifact logging is enabled in the config."""
        art_cfg = ExperimentRunner._get_wandb_artifacts_cfg(cfg)
        if art_cfg is None:
            return False
        if isinstance(art_cfg, DictConfig):
            return bool(art_cfg.get('enabled', False))
        return bool(getattr(art_cfg, 'enabled', False))

    @staticmethod
    def _resolve_wandb_alias(cfg: DictConfig, alias_key: str, default: str) -> str:
        """Resolve a wandb artifact alias with robust fallbacks."""
        art_cfg = ExperimentRunner._get_wandb_artifacts_cfg(cfg)
        if art_cfg is None:
            return default
        if isinstance(art_cfg, DictConfig):
            aliases_cfg = art_cfg.get('aliases', None)
        else:
            aliases_cfg = getattr(art_cfg, 'aliases', None)
        if aliases_cfg is None:
            return default
        if isinstance(aliases_cfg, DictConfig):
            return aliases_cfg.get(alias_key, default)
        if isinstance(aliases_cfg, dict):
            return aliases_cfg.get(alias_key, default)
        return default

    def __init__(self, config: DictConfig):
        # Keep a reference to the original Hydra config for display, but operate on
        # a detached copy so that subsequent mutations do not leak back to Hydra.
        self.original_config = config
        self.config = OmegaConf.create(OmegaConf.to_container(config, resolve=False))
        self.output_dir = Path(self.config.output_dir)
        self.results = {}

        # Configuration is expected to be fully resolved (single source of truth
        # in settings.*). Avoid mutating it here to keep determinism.

        # Setup device
        if self.config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.config.device)

        print(f"🚀 RlVAE Experiment Runner")
        try:
            exp_name = getattr(self.config, 'experiment_name', None)
            if exp_name is None and hasattr(self.config, 'experiment'):
                exp_name = getattr(self.config.experiment, 'name', 'unnamed')
        except Exception:
            exp_name = 'unnamed'
        print(f"📅 Experiment: {exp_name}")
        print(f"💻 Device: {self.device}")
        print(f"📁 Output: {self.output_dir}")

        # Set comprehensive random seed for reproducibility
        seed = self.config.get('seed', 42)
        experiment_type = self.config.get('experiment_type', 'research')
        configure_for_experiment(seed, experiment_type, self.device)
        print(f"🎲 Comprehensive seeding configured (seed: {seed})")
    
    def run(self):
        """Run the experiment based on configuration."""
        experiment_type = self.config.experiment.type
        print(f"\n🧪 Running {experiment_type} experiment: {self.config.experiment.name}")
        if experiment_type in ("single", "monolith"):
            self.run_single_experiment()
            # Stage C: add recon vs real panel using validation batch
            try:
                import torchvision.utils as vutils
                data_module = build_data_module(self.config.data)
                data_module.setup("fit", self.config.training)
                loader_c = data_module.val_dataloader()
                batch_c = next(iter(loader_c))[:8].to(self.device)
                model_wrapper = LightningRlVAETrainer(self.config, data_module=data_module)
                model_wrapper.model.eval()
                with torch.no_grad():
                    outc = model_wrapper.model(batch_c)
                recon_c = None
                if isinstance(outc, dict):
                    recon_c = outc.get('reconstruction', None) or outc.get('recon_x', None)
                elif hasattr(outc, 'reconstruction'):
                    recon_c = outc.reconstruction
                elif hasattr(outc, 'recon_x'):
                    recon_c = outc.recon_x
                if recon_c is not None:
                    recon_c = recon_c.clamp(0, 1)
                    grid_c = vutils.make_grid(torch.cat([batch_c, recon_c], dim=0), nrow=8, normalize=False)
                    wandb.log({"stageC/recon_vs_real": wandb.Image(grid_c)})
            except Exception as e:
                print(f"[Stage C] ⚠️ recon_vs_real logging failed: {e}")
        elif experiment_type == "comparison":
            self.run_comparison_study()
        elif experiment_type == "sweep":
            self.run_hyperparameter_sweep()
        elif experiment_type == "pipeline":
            self.run_pipeline_experiment()
        elif experiment_type == "three_stage":
            self.run_three_stage_experiment()
        else:
            raise ValueError(f"Unknown experiment type: {experiment_type}")

    # --- Monolith config updater -----------------------------------------
    def _update_monolith_stagec_config(
        self,
        *,
        encoder_path: str,
        decoder_path: str,
        metric_path: str,
        latent_dim: int | None = None,
        input_dim: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Update conf/monolith_stagec.yaml with freshly produced Stage‑A/B artifacts.

        This keeps a single source of truth for standalone Stage‑C runs.
        """
        try:
            from omegaconf import OmegaConf
            monolith_cfg_path = Path("conf/monolith_stagec.yaml")
            if not monolith_cfg_path.exists():
                print(f"[MONOLITH] No conf/monolith_stagec.yaml found; skipping updater")
                return
            cfg = OmegaConf.load(monolith_cfg_path)
            # Ensure nested keys
            if 'model' not in cfg:
                cfg.model = {}
            if 'pretrained' not in cfg.model:
                cfg.model.pretrained = {}
            # Update paths
            cfg.model.pretrained.encoder_path = str(encoder_path)
            cfg.model.pretrained.decoder_path = str(decoder_path)
            cfg.model.pretrained.metric_path = str(metric_path)
            if latent_dim is not None:
                cfg.model.latent_dim = int(latent_dim)
            if input_dim is not None:
                cfg.model.input_dim = list(input_dim)
            OmegaConf.save(cfg, monolith_cfg_path)
            print(f"[MONOLITH] ✅ Updated monolith config with Stage‑A/B artifacts:\n"
                  f"  - encoder: {encoder_path}\n  - decoder: {decoder_path}\n  - metric:  {metric_path}")
        except Exception as e:
            print(f"[MONOLITH] ⚠️ Failed to update monolith_stagec.yaml: {e}")
    
    def run_single_experiment(self):
        """Run a single experiment with current configuration."""
        
        # Ensure alias mirrors exist (important when running single stages)
        self._ensure_training_aliases()

        # Initialize wandb
        wandb_logger = self._setup_wandb("single_run")
        
        # Create data module
        data_module = build_data_module(self.config.data)
        data_module.setup("fit", self.config.training)
        self._maybe_run_data_analysis(data_module=data_module, wandb_run=getattr(wandb_logger, "experiment", None))
        
        # Create model
        model_wrapper = LightningRlVAETrainer(
            self.config,
            data_module=data_module
        )
        # === Config-driven generator, inference, evaluator ===
        self.generator = model_wrapper.model.create_generator(self.config.get('generation', None))
        self.inference_pipeline = model_wrapper.model.create_inference_pipeline(self.config.get('inference', None))
        self.evaluator = model_wrapper.model.create_evaluator() if hasattr(model_wrapper.model, 'create_evaluator') else None
        # === End config-driven instantiation ===
        
        # Setup trainer
        trainer = self._create_trainer(wandb_logger)
        
        # Train
        print(f"🚀 Starting training...")
        trainer.fit(model_wrapper, data_module)
        
        # Test
        print(f"🧪 Running test...")
        test_results = trainer.test(model_wrapper, data_module)
        
        # Save results
        combined_results = dict(getattr(self, "results", {}))
        combined_results.update({
            'test_results': test_results[0] if test_results else {},
            'model_summary': model_wrapper.model.get_model_summary()
        })
        self.results = combined_results
        
        print(f"✅ Single experiment completed!")
        self._save_results()
    
    def run_comparison_study(self):
        """Run comparison between multiple model variants."""
        
        print(f"🔬 Running comparison study...")
        models_to_compare = self.config.experiment.models
        comparison_metrics = self.config.experiment.comparison_metrics
        
        metrics_collector = MetricsCollector()
        all_results = {}
        
        for model_name in models_to_compare:
            print(f"\n🎯 Training model: {model_name}")
            
            # Create model-specific config
            model_config = self._create_model_config(model_name)

            # Ensure aliases for each model config as well
            self._ensure_training_aliases(model_config)
            
            # Setup wandb for this model
            wandb_logger = self._setup_wandb(f"comparison_{model_name}")
            
            # Create data module
            data_module = build_data_module(model_config.data)
            data_module.setup("fit", model_config.training)
            
            # Create model wrapper
            model_wrapper = LightningRlVAETrainer(
                model_config,
                data_module=data_module
            )
            
            # Setup trainer
            trainer = self._create_trainer(wandb_logger)
            
            # Train
            trainer.fit(model_wrapper, data_module)
            
            # Test
            test_results = trainer.test(model_wrapper, data_module)
            
            # Collect metrics
            if test_results:
                metrics = self._extract_comparison_metrics(test_results[0], comparison_metrics)
                metrics_collector.add_model_metrics(model_name, metrics)
            
            # Store results
            all_results[model_name] = {
                'test_results': test_results[0] if test_results else {},
                'model_summary': model_wrapper.model.get_model_summary()
            }
            
            # Finish this wandb run
            wandb.finish()
        
        # Create comparison analysis
        print(f"\n📊 Analyzing comparison results...")
        comparison_summary = metrics_collector.get_comparison_summary()
        
        # Start final wandb run for comparison
        wandb_logger = self._setup_wandb("comparison_analysis")
        metrics_collector.log_comparison_to_wandb()
        
        combined_results = dict(getattr(self, "results", {}))
        combined_results.update({
            'comparison_summary': comparison_summary,
            'individual_results': all_results,
            'comparison_metrics': comparison_metrics
        })
        self.results = combined_results
        
        print(f"✅ Comparison study completed!")
        self._save_results()
        self._print_comparison_summary(comparison_summary)
    
    def run_hyperparameter_sweep(self):
        """Run hyperparameter sweep (placeholder for Hydra sweep)."""
        print(f"🌊 Hyperparameter sweep mode")
        print(f"⚠️ This should be run with Hydra multirun (-m flag)")
        print(f"Example: python run_experiment.py experiment=hyperparameter_sweep -m")
        
        # For individual sweep runs, just run single experiment
        self.run_single_experiment()
    
    def run_pipeline_experiment(self):
        """
        Run a two-stage pipeline:
        1. Vanilla VAE training + diverse metric extraction (with wandb logging and results file)
        2. RLVAE training with pretrained components from stage 1
        """
        import torch
        import wandb
        from pathlib import Path
        from datetime import datetime
        from scripts.train_diverse_metric_vae import create_model, SpritesDataset, extract_diverse_metric, save_model_components
        from torch.utils.data import DataLoader, ConcatDataset
        import torch.nn.functional as F
        import torchvision.utils as vutils
        import numpy as np
        import yaml
        self._maybe_run_data_analysis()
        # --- Stage 1: Vanilla VAE + Diverse Metric ---
        if hasattr(self.config.experiment, 'skip_stage1') and self.config.experiment.skip_stage1:
            print("⏭️ Skipping Stage 1 (vanilla VAE) as requested by experiment.skip_stage1=true")
            # Expect user to provide pretrained paths in config.pretrained
            encoder_path = getattr(self.config.pretrained, 'encoder_path', None)
            decoder_path = getattr(self.config.pretrained, 'decoder_path', None)
            metric_path = getattr(self.config.pretrained, 'metric_path', None)
            if not all([encoder_path, decoder_path, metric_path]):
                raise FileNotFoundError("Missing pretrained components in config.pretrained for skipped Stage 1")
            component_paths = {'encoder': encoder_path, 'decoder': decoder_path, 'metric': metric_path}
        else:
            stage1_cfg = OmegaConf.merge(
                OmegaConf.create({
                    'architecture': self.config.settings.training.stage_overrides.stage_a.architecture,
                    'latent_dim': self.config.settings.training.stage_overrides.stage_a.latent_dim,
                    'epochs': self.config.settings.training.stage_overrides.stage_a.epochs,
                    'batch_size': self.config.settings.training.stage_overrides.stage_a.batch_size
                }),
                self.config.experiment.stage_a if hasattr(self.config.experiment, 'stage_a') else {}
            )
            print("\n=== [Pipeline] Stage 1: Vanilla VAE + Diverse Metric ===")
            architecture = getattr(stage1_cfg, 'architecture', 'mlp') if hasattr(stage1_cfg, 'architecture') else 'mlp'
            latent_dim = getattr(stage1_cfg, 'latent_dim', 16) if hasattr(stage1_cfg, 'latent_dim') else 16
            epochs = getattr(stage1_cfg, 'epochs', 50) if hasattr(stage1_cfg, 'epochs') else 50
            temperature = getattr(stage1_cfg, 'temperature', 0.5) if hasattr(stage1_cfg, 'temperature') else 0.5
            regularization = getattr(stage1_cfg, 'regularization', 0.01) if hasattr(stage1_cfg, 'regularization') else 0.01
            preset = getattr(stage1_cfg, 'preset', 'balanced') if hasattr(stage1_cfg, 'preset') else 'balanced'
            n_heatmaps = getattr(stage1_cfg, 'n_heatmaps', 6) if hasattr(stage1_cfg, 'n_heatmaps') else 6
            data_config_name = getattr(stage1_cfg, 'data', 'cyclic_sprites')
            data_presets = {
                'cyclic_sprites': {
                    'train_path': 'data/processed/Sprites_train_cyclic.pt',
                    'test_path': 'data/processed/Sprites_test_cyclic.pt'
                },
                'ellipse_sequences': {
                    'train_path': getattr(self.config.data, 'train_path', None),
                    'test_path': getattr(self.config.data, 'test_path', None)
                }
            }
            data_cfg = data_presets.get(data_config_name, data_presets['cyclic_sprites'])
            train_data_path = data_cfg.get('train_path', 'data/processed/Sprites_train_cyclic.pt')
            test_data_path = data_cfg.get('test_path', 'data/processed/Sprites_test_cyclic.pt')
            print(f"[Stage 1] Using train data: {train_data_path}")
            print(f"[Stage 1] Using test data: {test_data_path}")
            train_dataset = SpritesDataset(train_data_path, normalize=False)
            test_dataset = SpritesDataset(test_data_path, normalize=False)
            full_dataset = ConcatDataset([train_dataset, test_dataset])
            train_loader = DataLoader(full_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

            # --- MANUAL PATCH: Set model.sequence_length and n_flows from dataset (ALWAYS RUN) ---
            import torch
            raw_data = torch.load(self.config.data.train_path, map_location='cpu')
            if raw_data.ndim >= 2:
                detected_seq_len = raw_data.shape[1]
            else:
                detected_seq_len = 1
            self._set_config_value(self.config, "model.sequence_length", detected_seq_len)
            self._set_config_value(self.config, "model.n_flows", detected_seq_len - 1)
            self._set_config_value(stage_a_base, "model.sequence_length", detected_seq_len)
            self._set_config_value(stage_a_base, "model.n_flows", detected_seq_len - 1)
            print(f"[PATCH] Stage A auto sequence_length={detected_seq_len}, n_flows={detected_seq_len - 1}")
            # Ensure input_dim is set from data config for Stage 1 (robust automation)
            input_dim = [self.config.data.channels, self.config.data.image_size[0], self.config.data.image_size[1]]
            model = create_model(architecture, input_dim=tuple(input_dim), latent_dim=latent_dim).to(self.device)
            if architecture.lower() in ['cnn', 'resnet']:
                lr = 5e-5
                print(f"[Stage 1] Using lower learning rate {lr} for {architecture} architecture")
            else:
                lr = 1e-4
            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
            wandb_run = wandb.init(
                project=self.config.wandb.project, 
                name=f"pipeline_stage1_vanilla_vae_{architecture}_ld{latent_dim}",
                settings=wandb.Settings(init_timeout=300),
                config={
                "architecture": architecture,
                "latent_dim": latent_dim,
                "epochs": epochs,
                "temperature": temperature,
                "regularization": regularization,
                "preset": preset,
                "n_heatmaps": n_heatmaps,
                "stage": "pipeline_stage1_vanilla_vae"
            })
            print(f"[Stage 1] Training for {epochs} epochs...")
            batch_counter = 0
            for epoch in range(epochs):
                model.train()
                total_loss = 0
                total_recon_loss = 0
                total_kld_loss = 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    if architecture.lower() in ["mlp", "pythae"]:
                        output = model({"data": batch})
                        loss = output.loss
                        recon_loss = output.recon_loss.item()
                        kld_loss = output.reg_loss.item()
                        recon_batch = output.recon_x
                    else:
                        output = model(batch)
                        loss = output.loss
                        recon_loss = output.reconstruction_loss.item()
                        kld_loss = output.reg_loss.item()
                        recon_batch = output.recon_x
                    optimizer.zero_grad()
                    loss.backward()
                    if architecture.lower() in ['cnn', 'resnet']:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    total_loss += loss.item()
                    total_recon_loss += recon_loss
                    total_kld_loss += kld_loss
                    if batch_counter % 100 == 0:
                        with torch.no_grad():
                            recon_display = recon_batch.clamp(0, 1)
                            comparison = torch.cat([batch[:8], recon_display[:8]], dim=0)
                            grid = vutils.make_grid(comparison, nrow=8, normalize=False)
                            wandb.log({
                                "reconstructions": wandb.Image(grid),
                                "batch": batch_counter,
                                "epoch": epoch + 1
                            })
                    batch_counter += 1
                avg_loss = total_loss / len(train_loader)
                avg_recon_loss = total_recon_loss / len(train_loader)
                avg_kld_loss = total_kld_loss / len(train_loader)
                model.eval()
                val_loss = 0
                val_recon_loss = 0
                val_kld_loss = 0
                val_batches = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        if architecture.lower() in ["mlp", "pythae"]:
                            output = model({"data": batch})
                            recon_loss_val = output.recon_loss.item()
                            kld_loss_val = output.reg_loss.item()
                        else:
                            output = model(batch)
                            recon_loss_val = output.reconstruction_loss.item()
                            kld_loss_val = output.reg_loss.item()
                        if not (torch.isnan(output.loss) or torch.isinf(output.loss)):
                            val_loss += output.loss.item()
                            val_recon_loss += recon_loss_val
                            val_kld_loss += kld_loss_val
                            val_batches += 1
                if val_batches > 0:
                    avg_val_loss = val_loss / val_batches
                    avg_val_recon_loss = val_recon_loss / val_batches
                    avg_val_kld_loss = val_kld_loss / val_batches
                else:
                    avg_val_loss = avg_val_recon_loss = avg_val_kld_loss = 0
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": avg_loss,
                    "train/reconstruction_loss": avg_recon_loss,
                    "train/kld_loss": avg_kld_loss,
                    "val/loss": avg_val_loss,
                    "val/reconstruction_loss": avg_val_recon_loss,
                    "val/kld_loss": avg_val_kld_loss
                })
                print(f"[Stage 1] Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} (Recon: {avg_recon_loss:.4f} + KL: {avg_kld_loss:.4f}) | Val Loss: {avg_val_loss:.4f}")
            print("[Stage 1] Training complete. Running encoder sanity check...")
            try:
                sample_batch = next(iter(train_loader))
                if isinstance(sample_batch, (list, tuple)):
                    sample_batch = sample_batch[0]
                sample_batch = sample_batch.to(self.device)
                if architecture.lower() in ["mlp", "pythae"]:
                    enc_out = model.encoder({"data": sample_batch})
                else:
                    enc_out = model.encoder(sample_batch)
                mu_chk = enc_out.embedding
                logvar_chk = enc_out.log_covariance
                print(f"[Stage 1][ENC CHECK] mu: min={mu_chk.min().item():.4e}, max={mu_chk.max().item():.4e}, mean={mu_chk.mean().item():.4e}, std={mu_chk.std().item():.4e}")
                print(f"[Stage 1][ENC CHECK] log_var: min={logvar_chk.min().item():.4e}, max={logvar_chk.max().item():.4e}, mean={logvar_chk.mean().item():.4e}, std={logvar_chk.std().item():.4e}")
                if torch.isnan(mu_chk).any() or torch.isinf(mu_chk).any() or torch.isnan(logvar_chk).any() or torch.isinf(logvar_chk).any():
                    print("[Stage 1][ENC CHECK] ⚠️ Detected NaN/Inf in encoder outputs!")
                else:
                    print("[Stage 1][ENC CHECK] ✅ Encoder outputs are finite.")
            except Exception as enc_exc:
                print(f"[Stage 1][ENC CHECK] ⚠️ Encoder sanity check failed: {enc_exc}")
            print("[Stage 1] Training complete. Saving components...")
            component_paths = save_model_components(model, architecture, latent_dim)
            print("[Stage 1] Extracting diverse metric...")
            metric_path = extract_diverse_metric(model, architecture, latent_dim, temperature=temperature, regularization=regularization, input_dim=tuple(input_dim), data_path=train_data_path)
            
            # --- METRIC ANALYSIS & VISUALIZATION (from train_diverse_metric_vae.py) ---
            print("[Stage 1] Creating metric analysis and visualizations...")
            import matplotlib.pyplot as plt
            import seaborn as sns
            plt.style.use('default')
            sns.set_palette("husl")
            
            try:
                metric_data = torch.load(metric_path, map_location='cpu', weights_only=False)
                M_matrices = metric_data['M_matrices']
                centroids = metric_data['centroids']
                eigenvals = torch.linalg.eigvals(M_matrices).real
                min_eigenvals = eigenvals.min(dim=-1)[0]
                max_eigenvals = eigenvals.max(dim=-1)[0]
                mean_eigenvals = eigenvals.mean(dim=-1)
                cond_nums = max_eigenvals / (min_eigenvals + 1e-12)
                determinants = torch.linalg.det(M_matrices)
                eigenval_spread = max_eigenvals - min_eigenvals
                
                # 1. Eigenvalue distributions
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                fig.suptitle('Metric Eigenvalue & Condition Analysis', fontsize=16, fontweight='bold')
                axes[0,0].hist(min_eigenvals.numpy(), bins=40, color='red', edgecolor='black')
                axes[0,0].set_title('Min Eigenvalue')
                axes[0,1].hist(max_eigenvals.numpy(), bins=40, color='blue', edgecolor='black')
                axes[0,1].set_title('Max Eigenvalue')
                axes[0,2].hist(mean_eigenvals.numpy(), bins=40, color='green', edgecolor='black')
                axes[0,2].set_title('Mean Eigenvalue')
                axes[1,0].hist(cond_nums.numpy(), bins=40, color='orange', edgecolor='black')
                axes[1,0].set_title('Condition Number')
                log_dets = torch.log10(torch.abs(determinants) + 1e-50)
                axes[1,1].hist(log_dets.numpy(), bins=40, color='purple', edgecolor='black')
                axes[1,1].set_title('Log₁₀|Determinant|')
                axes[1,2].hist(eigenval_spread.numpy(), bins=40, color='cyan', edgecolor='black')
                axes[1,2].set_title('Eigenvalue Spread')
                plt.tight_layout()
                wandb.log({"metric_analysis/eigenvalue_distributions": wandb.Image(fig)})
                plt.close(fig)
                
                # 2. Heatmaps of metric matrices (configurable number, or fewer if less matrices available)
                n_heatmaps = min(n_heatmaps, len(M_matrices))
                fig, axes = plt.subplots(1, n_heatmaps, figsize=(4*n_heatmaps, 4))
                for i in range(n_heatmaps):
                    im = axes[i].imshow(M_matrices[i].numpy(), cmap='RdYlBu_r', aspect='auto')
                    axes[i].set_title(f'Matrix {i}')
                    plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                plt.tight_layout()
                wandb.log({"metric_analysis/metric_matrix_heatmaps": wandb.Image(fig)})
                plt.close(fig)
                
                # 3. Centroid statistics
                centroid_norms = torch.norm(centroids, dim=1)
                pairwise_dists = torch.cdist(centroids, centroids)
                triu_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=bool), diagonal=1)
                pairwise_vals = pairwise_dists[triu_mask]
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].hist(centroid_norms.numpy(), bins=30, color='red', edgecolor='black')
                axes[0].set_title('Centroid Norms')
                axes[1].hist(pairwise_vals.numpy(), bins=40, color='blue', edgecolor='black')
                axes[1].set_title('Pairwise Centroid Distances')
                plt.tight_layout()
                wandb.log({"metric_analysis/centroid_stats": wandb.Image(fig)})
                plt.close(fig)
                
                print(f"[Stage 1] ✅ Metric analysis complete: {n_heatmaps} matrix heatmaps created")
                
            except Exception as e:
                print(f"[Stage 1] ⚠️ Could not analyze metric for graphs: {e}")
            
            # Log metric file as wandb artifact
            if Path(metric_path).exists():
                artifact = wandb.Artifact(f"diverse_metric_{architecture}_ld{latent_dim}", type="metric")
                artifact.add_file(metric_path)
                wandb.log_artifact(artifact)
            
            # Save results YAML for Stage 1
            vanilla_results = {
                'architecture': architecture,
                'latent_dim': latent_dim,
                'epochs': epochs,
                'temperature': temperature,
                'regularization': regularization,
                'preset': preset,
                'encoder_path': component_paths['encoder'],
                'decoder_path': component_paths['decoder'],
                'metric_path': metric_path
            }
            results_path = Path(self.output_dir) / "vanilla_vae_results.yaml"
            with open(results_path, 'w') as f:
                yaml.dump(vanilla_results, f)
            print(f"[Stage 1] Results saved to: {results_path}")
            wandb.finish()
        # --- Stage 2: RLVAE Training ---
        print("\n=== [Pipeline] Stage 2: RLVAE Training ===")
        # Update config for pretrained paths
        self._set_config_value(self.config, "model.pretrained.encoder_path", component_paths['encoder'])
        self._set_config_value(self.config, "model.pretrained.decoder_path", component_paths['decoder'])
        self._set_config_value(self.config, "model.pretrained.metric_path", metric_path)
        # Ensure latent_dim is set from stage2 config if present
        # Resolve all interpolations (e.g., oc.decode expressions) before accessing Stage 2 values
        try:
            OmegaConf.resolve(self.config)
        except Exception as e:
            print(f"[PATCH] OmegaConf.resolve failed: {e}")
        stage2_cfg = self.config.experiment.stage2
        latent_dim2 = getattr(stage2_cfg, 'latent_dim', None)
        if latent_dim2 is not None:
            self._set_config_value(self.config, "model.latent_dim", latent_dim2)
        # --- PATCH: Ensure model.riemannian_beta is set from experiment.stage2 ---
        if hasattr(self.config, 'experiment') and hasattr(self.config.experiment, 'stage2') and hasattr(self.config.experiment.stage2, 'riemannian_beta'):
            self._set_config_value(self.config, "model.riemannian_beta", self.config.experiment.stage2.riemannian_beta)
            print(f"[PATCH] Set self.config.model.riemannian_beta = {self.config.model.riemannian_beta}")
        # --- PATCH: Ensure model.n_flows is set from experiment.stage2 ---
        if hasattr(self.config, 'experiment') and hasattr(self.config.experiment, 'stage2') and hasattr(self.config.experiment.stage2, 'n_flows'):
            n_flows_val = self.config.experiment.stage2.n_flows
            # Coerce string expressions like "8 - 1" to integers if present
            if isinstance(n_flows_val, str):
                import re
                m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", n_flows_val)
                if m:
                    try:
                        n_flows_val = int(m.group(1)) - int(m.group(2))
                        print(f"[PATCH] Coerced stage2.n_flows expression to int: {n_flows_val}")
                    except Exception:
                        pass
                else:
                    try:
                        n_flows_val = int(n_flows_val)
                        print(f"[PATCH] Coerced stage2.n_flows string to int: {n_flows_val}")
                    except Exception:
                        pass
            self._set_config_value(self.config, "model.n_flows", n_flows_val)
            print(f"[PATCH] Set self.config.model.n_flows = {self.config.model.n_flows}")
        # Ensure input_dim is set from data config for Stage 2 (robust automation)
        self._set_config_value(self.config, "model.input_dim", [
            self.config.data.channels,
            self.config.data.image_size[0],
            self.config.data.image_size[1]
        ])
        # Ensure visualization level respects CLI overrides; only apply stage2 default if no CLI override
        visualization_level2 = getattr(stage2_cfg, 'visualization', None)
        cli_overrides_visualization = False
        try:
            from hydra.core.hydra_config import HydraConfig
            task_overrides = getattr(HydraConfig.get(), 'overrides', None)
            if task_overrides and hasattr(task_overrides, 'task'):
                cli_overrides_visualization = any(
                    o.startswith('visualization=') or o.startswith('visualization.level=')
                    for o in task_overrides.task
                )
        except Exception:
            pass
        if visualization_level2 is not None and not cli_overrides_visualization:
            self._set_config_value(self.config, "visualization.level", visualization_level2)
        elif cli_overrides_visualization:
            print("[PATCH] Respecting CLI visualization override; not applying stage2.visualization")
        # === PATCH: Ensure evaluation config is present for Stage 2 ===
        from omegaconf import OmegaConf
        import yaml
        if not hasattr(self.config, "evaluation") or self.config.evaluation is None:
            with open("conf/evaluation/default.yaml", "r") as f:
                eval_cfg = OmegaConf.create(yaml.safe_load(f))
            self.config.evaluation = eval_cfg
        # === END PATCH ===
        # --- FINAL PATCH: Set model.sequence_length and n_flows right before Stage 2 model creation ---
        import torch
        raw_data = torch.load(self.config.data.train_path, map_location='cpu')
        if raw_data.ndim >= 2:
            detected_seq_len = raw_data.shape[1]
        else:
            detected_seq_len = 1
        self._set_config_value(self.config, "model.sequence_length", detected_seq_len)

        # CRITICAL FIX: Only override n_flows if not explicitly set in stage2 config
        if not (hasattr(self.config, 'experiment') and hasattr(self.config.experiment, 'stage2') and hasattr(self.config.experiment.stage2, 'n_flows')):
            self._set_config_value(self.config, "model.n_flows", detected_seq_len - 1)
            print(f"[FINAL PATCH] Auto-set self.config.model.n_flows = {detected_seq_len - 1}")
        else:
            print(f"[FINAL PATCH] Keeping user-specified self.config.model.n_flows = {self.config.model.n_flows}")
        print(f"[FINAL PATCH] Set self.config.model.sequence_length = {detected_seq_len}")
        
        # Use modular model as configured
        print(f"[FINAL PATCH] Using modular model as configured: {self.config.model._target_}")
        
        print("[FINAL PATCH] Model config before creation:")
        print(OmegaConf.to_yaml(self.config.model))
        # Ensure input_dim is set from data config for Stage 2 (robust automation)
        self._set_config_value(self.config, "model.input_dim", [
            self.config.data.channels,
            self.config.data.image_size[0],
            self.config.data.image_size[1]
        ])
        # Ensure visualization level respects CLI overrides; only apply stage2 default if no CLI override
        visualization_level2 = getattr(stage2_cfg, 'visualization', None)
        cli_overrides_visualization = False
        try:
            from hydra.core.hydra_config import HydraConfig
            task_overrides = getattr(HydraConfig.get(), 'overrides', None)
            if task_overrides and hasattr(task_overrides, 'task'):
                cli_overrides_visualization = any(
                    o.startswith('visualization=') or o.startswith('visualization.level=')
                    for o in task_overrides.task
                )
        except Exception:
            pass
        if visualization_level2 is not None and not cli_overrides_visualization:
            if hasattr(self.config, 'visualization') and hasattr(self.config.visualization, 'level'):
                self.config.visualization.level = visualization_level2
            elif hasattr(self.config, 'visualization') and isinstance(self.config.visualization, dict):
                self.config.visualization['level'] = visualization_level2
            else:
                self.config.visualization = {'level': visualization_level2}
        elif cli_overrides_visualization:
            print("[PATCH] Respecting CLI visualization override; not applying stage2.visualization")
        # CRITICAL FIX: Override training epochs for Stage 2 
        stage2_epochs = getattr(self.config.experiment.stage2, 'epochs', None)
        if stage2_epochs is not None:
            self.config.training.trainer.max_epochs = stage2_epochs
            print(f"[PATCH] Stage 2: Set training.trainer.max_epochs = {stage2_epochs}")
        
        n_flows = getattr(self.config.model, 'n_flows', None)
        print(f"[Stage 2] DEBUG: Training for {self.config.training.trainer.max_epochs} epochs")
        print(f"[Stage 2] DEBUG: Model n_flows = {n_flows} (config.model.n_flows)")
        # Run RLVAE training as in single experiment
        self.run_single_experiment()

        # === FINAL VISUALIZATIONS (after Stage 2) ===
        try:
            print("\n=== [Pipeline] Final Visualizations ===")
            from visualizations.manager import VisualizationManager
            # Assume model_wrapper is available from run_single_experiment (or reload model if needed)
            # For demonstration, we use the last trained model
            model = None
            if hasattr(self, 'evaluator') and hasattr(self.evaluator, 'model'):
                model = self.evaluator.model
            elif hasattr(self, 'generator') and hasattr(self.generator, 'model'):
                model = self.generator.model
            elif hasattr(self, 'inference_pipeline') and hasattr(self.inference_pipeline, 'model'):
                model = self.inference_pipeline.model
            if model is None:
                print("[Pipeline] Could not find trained model for visualization.")
                return
            viz_manager = VisualizationManager(model, self.device, self.config)
            basic_viz = viz_manager.modules['basic']
            # 1. Generation grid (prior samples)
            basic_viz.create_generation_grid(num_samples=16, epoch=self.config.training.trainer.max_epochs)
            # 2. Interpolation grid (random pairs)
            basic_viz.create_interpolation_grid(num_interpolations=5, steps=8, epoch=self.config.training.trainer.max_epochs)
            # 3. Comprehensive generation comparison (if multiple methods available)
            # Example: try to use generator to get images for several methods
            generation_methods = ["geodesic", "enhanced", "basic", "standard"]
            generation_results = {}
            fid_scores = {}
            for method in generation_methods:
                try:
                    if hasattr(model, 'generate_samples'):
                        result = model.generate_samples(num_samples=4, method=method)
                        images = result['images']
                        if images.dim() == 5:
                            images = images[:, 0]
                        generation_results[method] = images
                        # Optionally compute FID if available
                        if hasattr(model, 'compute_fid_score'):
                            fid = model.compute_fid_score(real_images=images, num_generated=min(16, len(images)), cache_key=f"pipeline_{method}")
                            if fid and 'fid_score' in fid:
                                fid_scores[method] = fid['fid_score']
                    else:
                        generation_results[method] = None
                except Exception as e:
                    print(f"[Pipeline] Generation for method {method} failed: {e}")
                    generation_results[method] = None
            # Only plot if at least one method succeeded
            if any(v is not None for v in generation_results.values()):
                basic_viz.create_comprehensive_generation_visualization(
                    generation_results=generation_results,
                    fid_scores=fid_scores,
                    num_samples_per_method=4,
                    epoch=self.config.training.trainer.max_epochs
                )
            # Log all final visualizations to wandb
            viz_manager.log_final_visualizations_to_wandb(epoch=self.config.training.trainer.max_epochs)
        except Exception as e:
            print(f"[Pipeline] ⚠️ Final visualizations failed: {e}")

    def run_three_stage_experiment(self):
        """
        Multi-stage pipeline:
          - Stage A: Warm VAE on full sequences (flattened)
          - Stage B: Metric learning at t=0 with selectable implementation (RHVAE-style or precision)
          - Optional RHMC sampling using the learned metric
          - Stage C: RLVAE with Riemannian prior/posterior, optionally updating metric
        """
        from torch.utils.data import DataLoader
        cfg = self.config
        # Fallback component paths (e.g., encoder/decoder) when Stage A is skipped
        comp_paths = {}

        # Lazy-import Stage A/B helpers only when those stages are requested.
        create_vanilla = None
        SpritesDataset = None
        extract_diverse_metric = None
        save_model_components = None
        if getattr(cfg.experiment, 'run_stage_a', True) or getattr(cfg.experiment, 'run_stage_b', True):
            from scripts.train_diverse_metric_vae import (
                create_model as create_vanilla,
                SpritesDataset,
                extract_diverse_metric,
                save_model_components,
            )
        from models.components.metric_tensor import MetricTensor
        from models.components.native_inverse_metric import NativeInverseMetricTensor, NativeInverseRHMC
        import torch
        import os
        import yaml
        from pathlib import Path
        import wandb
        out_dir = Path(cfg.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create a single WandB run for the whole three-stage pipeline
        if cfg.wandb.mode != "disabled" and wandb.run is None:
            try:
                from omegaconf import OmegaConf  # Ensure OmegaConf is available
                wandb.init(
                    project=cfg.wandb.project,
                    name=f"three_stage_pipeline_{cfg.experiment.name}",
                    mode=cfg.wandb.mode,
                    settings=wandb.Settings(init_timeout=300),
                    group=getattr(cfg.wandb, 'group', None),
                    config=OmegaConf.to_container(cfg, resolve=True)
                )
                print(f"✅ WandB initialized successfully for three-stage pipeline")
            except Exception as e:
                print(f"⚠️ WandB initialization failed: {e}")
                print("Continuing without WandB logging...")
                self._set_config_value(self.config, "wandb.mode", "disabled")
                cfg = self.config

        # Optional data sanity panel (ellipse-specific)
        try:
            self._maybe_run_data_analysis(wandb_run=wandb.run)
        except Exception as da_err:
            print(f"[Data Analysis] ⚠️ Skipping data overview: {da_err}")

        # ============================================================================
        # CENTRALIZED STAGE DATA LOOKUP - ELIMINATE ALL DUPLICATION
        # ============================================================================
        print("\n=== [CENTRALIZED] Stage Data Lookup ===")
        
        # Define architecture and latent_dim once
        arch = cfg.model.encoder.architecture if hasattr(cfg.model, 'encoder') and cfg.model.encoder is not None else 'mlp'
        latent_dim = cfg.model.latent_dim
        metric_impl = cfg.experiment.stage_b.implementation
        
        print(f"[CENTRALIZED] Architecture: {arch}, Latent dim: {latent_dim}")
        print(f"[CENTRALIZED] Metric implementation: {metric_impl}")
        
        # Find Stage A and B data ONCE at the beginning
        stage_a_data = None
        stage_b_data = None
        
        if getattr(cfg.experiment, 'run_stage_b', True) or getattr(cfg.experiment, 'run_stage_c', True):
            print(f"[CENTRALIZED] Looking up Stage A data...")
            stage_a_data = find_stage_a_data(cfg, arch, latent_dim)
            if stage_a_data:
                print(f"[CENTRALIZED] ✅ Found Stage A data: {stage_a_data['base_dir']}")
            else:
                print(f"[CENTRALIZED] ❌ No Stage A data found")
        
        if getattr(cfg.experiment, 'run_stage_c', True):
            print(f"[CENTRALIZED] Looking up Stage B data...")
            stage_b_data = find_stage_b_data(cfg, arch, latent_dim, metric_impl.upper())
            if stage_b_data:
                print(f"[CENTRALIZED] ✅ Found Stage B data: {stage_b_data['base_dir']}")
            else:
                print(f"[CENTRALIZED] ❌ No Stage B data found")
        
        print(f"[CENTRALIZED] Stage data lookup completed")
        print("=" * 50)

        # Stage A: Train base model
        if getattr(cfg.experiment, 'run_stage_a', True):
            stage_a_base = self.config
            self.config = OmegaConf.create(OmegaConf.to_container(stage_a_base, resolve=False))
            cfg = self.config
            model_choice = str(getattr(cfg.experiment.stage_a, 'model', 'vanilla_vae')).lower()
            # Set defaults if not specified
            arch = getattr(cfg.model.encoder, 'architecture', 'mlp')
            latent_dim = getattr(cfg.model, 'latent_dim', 16)
            print(f"[Stage A] Using architecture: {arch}, latent_dim: {latent_dim}")
            input_dim = (cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1])

            # Build data module generically (supports sprites, ellipses, etc.)
            data_module = None
            train_frames_tensor = None
            test_frames_tensor = None
            val_frames_tensor = None

            def _dataset_to_frame_tensor(dataset):
                if dataset is None:
                    return None
                frames = []
                for item in dataset:
                    sample = item[0] if isinstance(item, (tuple, list)) else item
                    if isinstance(sample, dict):
                        sample = sample.get('data', None)
                    if sample is None:
                        continue
                    if sample.dim() == 5:
                        # [B, T, C, H, W] -> flatten batch dimension first if any
                        b, t = sample.shape[:2]
                        sample = sample.reshape(b * t, *sample.shape[2:])
                    if sample.dim() == 4:
                        frames.append(sample)
                    elif sample.dim() == 3:
                        frames.append(sample.unsqueeze(0))
                    else:
                        raise ValueError(f"Unsupported sample rank {sample.dim()} for Stage A dataset flattening")
                if not frames:
                    return None
                return torch.cat(frames, dim=0)

            try:
                data_module = build_data_module(cfg.data)
                data_module.setup("fit", getattr(cfg, "training", None))
                train_frames_tensor = _dataset_to_frame_tensor(getattr(data_module, 'train_dataset', None))
                val_frames_tensor = _dataset_to_frame_tensor(getattr(data_module, 'val_dataset', None))
                test_frames_tensor = _dataset_to_frame_tensor(getattr(data_module, 'test_dataset', None))
            except Exception as data_module_err:
                print(f"[Stage A] ⚠️ Data module setup failed ({data_module_err}); falling back to paths where available.")
                data_module = None

            train_path = getattr(cfg.data, 'train_path', None)
            test_path = getattr(cfg.data, 'test_path', None)

            if model_choice == 'rhvae':
                print("\n=== [Stage A] RHVAE training ===")
                from models.rhvae_experiment import RHVAEExperiment as _RHVAEExp
                # Initialize RHVAEExperiment
                rh_exp = _RHVAEExp(
                    input_dim=list(input_dim),
                    latent_dim=int(getattr(cfg.model, 'latent_dim', latent_dim)),
                    # Core HMC / RHVAE params (fallbacks preserve previous behavior)
                    n_lf=int(getattr(cfg.model, 'n_lf', 6)),
                    eps_lf=float(getattr(cfg.model, 'eps_lf', 2e-5)),
                    beta_zero=float(getattr(cfg.model, 'beta_zero', 1.0)),
                    temperature=float(getattr(cfg.model, 'temperature', getattr(cfg.metric, 'temperature', 0.7))),
                    regularization=float(getattr(cfg.model, 'regularization', getattr(cfg.metric, 'regularization', 1e-3))),
                    # Enc/Dec (optional hydra-style dicts)
                    encoder=getattr(cfg.model, 'encoder', None),
                    decoder=getattr(cfg.model, 'decoder', None),
                    # Advanced metric/weighting controls to match Hydra script behavior
                    align_with_knn_cov=bool(getattr(cfg.model, 'align_with_knn_cov', False)),
                    knn_k=int(getattr(cfg.model, 'knn_k', 300)),
                    alpha_align=float(getattr(cfg.model, 'alpha_align', 0.5)),
                    metric_normalization=str(getattr(cfg.model, 'metric_normalization', 'none')),
                    weight_kernel=str(getattr(cfg.model, 'weight_kernel', 'isotropic')),
                    weight_metric_normalization=str(getattr(cfg.model, 'weight_metric_normalization', 'trace')),
                    normalize_weight_sum=bool(getattr(cfg.model, 'normalize_weight_sum', False)),
                    topk_weights=getattr(cfg.model, 'topk_weights', None),
                    reestimate_metric_from_decoder_jacobian=bool(getattr(cfg.model, 'reestimate_metric_from_decoder_jacobian', False)),
                    jacobian_alpha=float(getattr(cfg.model, 'jacobian_alpha', 0.5)),
                    jacobian_h=float(getattr(cfg.model, 'jacobian_h', 1e-3)),
                    jacobian_stride=int(getattr(cfg.model, 'jacobian_stride', 4)),
                    metric_scale=float(getattr(cfg.model, 'metric_scale', 1.0)),
                    realign_centroids=bool(getattr(cfg.model, 'realign_centroids', False)),
                    centroid_realign_method=str(getattr(cfg.model, 'centroid_realign_method', 'kmeans')),
                    max_centroids=getattr(cfg.model, 'max_centroids', None),
                    centroid_subsample_method=str(getattr(cfg.model, 'centroid_subsample_method', 'fps')),
                    # System
                    device=str(self.device),
                    seed=int(getattr(cfg, 'seed', 42)),
                )
                # Use configured batch size for RHVAE Stage A
                batch_size = int(cfg.experiment.stage_a.batch_size)
                if train_frames_tensor is not None and test_frames_tensor is not None:
                    rh_exp.train_data = train_frames_tensor.float()
                    rh_exp.test_data = test_frames_tensor.float()
                    rh_exp.batch_size = batch_size
                    print(f"[Stage A] RHVAE using in-memory tensors from data module (train={rh_exp.train_data.shape}, test={rh_exp.test_data.shape})")
                elif train_path is not None and test_path is not None:
                    rh_exp.load_data(train_path=train_path, test_path=test_path, batch_size=batch_size)
                else:
                    raise RuntimeError("Stage A RHVAE requires either data module tensors or explicit train/test paths")
                rh_exp.train(
                    epochs=int(cfg.experiment.stage_a.epochs),
                    learning_rate=1e-4,
                    weight_decay=1e-5,
                    log_every=50,
                    save_every=10,
                    output_dir=str(Path(cfg.checkpoint.stageA_dir)),
                    use_wandb=(cfg.wandb.mode != 'disabled'),
                    wandb_config={
                        "project": cfg.wandb.project,
                        "name": f"pipeline_stage1_rhvae_ld{latent_dim}",
                        "group": getattr(cfg.wandb, 'group', None),
                        "tags": ["pipeline", "stageA", "rhvae"],
                    },
                )
                # Stage A: Reconstruction vs Original (RHVAE path)
                try:
                    import torchvision.utils as vutils
                    # Load a small batch from test set
                    if test_path is not None:
                        raw_test = torch.load(test_path, map_location='cpu')
                    elif test_frames_tensor is not None:
                        raw_test = test_frames_tensor.cpu()
                    else:
                        raise RuntimeError("No Stage A test data available for RHVAE reconstruction logging")
                    if raw_test.ndim == 5:
                        # Use first timestep frames for clarity
                        imgs = raw_test[:8, 0].to(self.device)
                    elif raw_test.ndim == 4:
                        imgs = raw_test[:8].to(self.device)
                    else:
                        raise ValueError(f"Unsupported data shape for recon logging: {tuple(raw_test.shape)}")
                    rh_exp.model.eval()
                    with torch.no_grad():
                        enc = rh_exp.model.encoder(imgs)
                        z = enc.get('embedding', None) if isinstance(enc, dict) else enc
                        if z is None and isinstance(enc, dict):
                            z = enc.get('z', None)
                        dec = rh_exp.model.decoder(z)
                        recon = dec.get('reconstruction', None) if isinstance(dec, dict) else dec
                        if recon is None and isinstance(dec, dict):
                            recon = dec.get('recon_x', None)
                        recon = torch.sigmoid(recon) if recon.dtype.is_floating_point else recon
                        recon = recon.clamp(0, 1)
                        # Normalize for visualization only to avoid dim-looking outputs
                        grid = vutils.make_grid(
                            torch.cat([imgs, recon], dim=0),
                            nrow=8,
                            normalize=True,
                            value_range=(0.0, 1.0)
                        )
                        if wandb.run is not None:
                            wandb.log({"stageA/recon_vs_real": wandb.Image(grid)})
                except Exception as e:
                    print(f"[Stage A] ⚠️ RHVAE recon_vs_real logging failed: {e}")
                # Compute t=0 latents with the trained RHVAE encoder (to subset centroids later)
                try:
                    # Load raw training tensor and extract timestep 0 frames
                    from torch.utils.data import DataLoader as TorchLoader, TensorDataset
                    raw = torch.load(train_path, map_location='cpu')
                    if raw.ndim == 5:  # [N, S, C, H, W]
                        imgs0 = raw[:, 0]
                    elif raw.ndim == 4:  # [N, C, H, W]
                        imgs0 = raw
                    else:
                        raise ValueError(f"Unsupported data shape for t=0 extraction: {tuple(raw.shape)}")
                    ds0 = TensorDataset(imgs0)
                    dl0 = TorchLoader(ds0, batch_size=256, shuffle=False)
                    mu_list = []
                    rh_exp.model.eval()
                    with torch.no_grad():
                        for (xb,) in dl0:
                            xb = xb.to(self.device)
                            enc = rh_exp.model.encoder(xb)
                            if isinstance(enc, dict) and 'embedding' in enc:
                                mu_list.append(enc['embedding'].detach().cpu())
                            elif isinstance(enc, dict) and 'z' in enc:
                                mu_list.append(enc['z'].detach().cpu())
                            elif isinstance(enc, torch.Tensor):
                                mu_list.append(enc.detach().cpu())
                    t0_latents = torch.cat(mu_list, dim=0) if mu_list else None
                except Exception as e:
                    print(f"[Stage A] ⚠️ Could not compute t=0 latents: {e}")
                    t0_latents = None

                # Export metric to organized Stage A folder
                stageA_paths = get_stage_paths(cfg, 'A', 'RHVAE', arch, latent_dim)
                try:
                    centroids = rh_exp.metric_adapter.centroids_tens.detach().cpu()
                    M = rh_exp.metric_adapter.M_tens.detach().cpu()
                    # Save encoder/decoder and full model for reuse
                    enc_path = stageA_paths['encoder_path']
                    dec_path = stageA_paths['decoder_path']
                    model_path = stageA_paths['model_path']
                    try:
                        torch.save(rh_exp.model.encoder.state_dict(), enc_path)
                    except Exception:
                        enc_path = None
                    try:
                        torch.save(rh_exp.model.decoder.state_dict(), dec_path)
                    except Exception:
                        dec_path = None
                    try:
                        # Save a portable checkpoint via experiment helper
                        if hasattr(rh_exp, 'save_model'):
                            rh_exp.save_model(str(model_path))
                        else:
                            torch.save({'model_state_dict': rh_exp.model.state_dict()}, model_path)
                    except Exception:
                        model_path = None
                    payload = {
                        'centroids': centroids,
                        'metric_matrices': M,
                        'temperature': float(rh_exp.temperature),
                        'regularization': float(rh_exp.regularization),
                        'latent_dim': int(latent_dim),
                        't0_latents': t0_latents,
                        'encoder_path': str(enc_path) if enc_path is not None else None,
                        'decoder_path': str(dec_path) if dec_path is not None else None,
                        'model_path': str(model_path) if model_path is not None else None,
                    }
                    torch.save(payload, stageA_paths['metric_path'])
                    print(f"[Stage A] ✅ Saved RHVAE metric to {stageA_paths['metric_path']}")
                    
                    # Save Stage A configuration
                    stageA_config = {
                        'stage': 'A',
                        'model_type': 'RHVAE',
                        'architecture': arch,
                        'latent_dim': latent_dim,
                        'epochs': cfg.experiment.stage_a.epochs,
                        'temperature': float(rh_exp.temperature),
                        'regularization': float(rh_exp.regularization),
                        'timestamp': datetime.now().isoformat()
                    }
                    with open(stageA_paths['config_path'], 'w') as f:
                        yaml.dump(stageA_config, f)
                    print(f"[Stage A] ✅ Saved Stage A config to {stageA_paths['config_path']}")
                except Exception as e:
                    print(f"[Stage A] ⚠️ Could not export RHVAE metric: {e}")
                # Propagate component paths for Stage B/C
                comp_paths = {}
                if enc_path is not None:
                    comp_paths['encoder'] = str(enc_path)
                if dec_path is not None:
                    comp_paths['decoder'] = str(dec_path)
                # Also expose the live model for Stage B recon logging
                stageB_model = getattr(rh_exp, 'model', None)
                # keep single run open
            else:
                print("\n=== [Stage A] Warm VAE training ===")
                print(f"[Stage A] Configuration:")
                print(f"  - Architecture: {arch}")
                print(f"  - Latent dim: {latent_dim}")
                print(f"  - Epochs: {cfg.experiment.stage_a.epochs}")
                print(f"  - Batch size: {cfg.experiment.stage_a.batch_size}")
                print(f"  - Learning rate: {cfg.experiment.stage_a.lr}")
                
                # Build dataset: flatten sequences (support in-memory tensors or legacy paths)
                from torch.utils.data import DataLoader as TorchDataLoader
                from torch.utils.data import ConcatDataset as TorchConcat
                if train_frames_tensor is not None and train_frames_tensor.numel() > 0:
                    ds_train = torch.utils.data.TensorDataset(train_frames_tensor.float())
                else:
                    ds_train = SpritesDataset(train_path, normalize=False, timestep_only=None)

                if test_frames_tensor is not None and test_frames_tensor.numel() > 0:
                    ds_test = torch.utils.data.TensorDataset(test_frames_tensor.float())
                elif test_path is not None:
                    ds_test = SpritesDataset(test_path, normalize=False, timestep_only=None)
                elif val_frames_tensor is not None and val_frames_tensor.numel() > 0:
                    ds_test = torch.utils.data.TensorDataset(val_frames_tensor.float())
                else:
                    ds_test = None

                datasets_for_training = [ds_train] if ds_train is not None else []
                if ds_test is not None:
                    datasets_for_training.append(ds_test)
                if len(datasets_for_training) == 0:
                    raise RuntimeError("Stage A could not build a training dataset")
                if len(datasets_for_training) == 1:
                    full_training_dataset = datasets_for_training[0]
                else:
                    full_training_dataset = TorchConcat(datasets_for_training)

                loader = TorchDataLoader(
                    full_training_dataset,
                    batch_size=cfg.experiment.stage_a.batch_size,
                    shuffle=True
                )

                arch_lower = arch.lower()
                effective_arch = arch_lower.replace('_gray', '') if '_gray' in arch_lower else arch_lower
                factory_arch = effective_arch
                if factory_arch not in ["cnn", "resnet", "mlp", "pythae"]:
                    factory_arch = arch_lower

                vanilla = create_vanilla(factory_arch, input_dim=input_dim, latent_dim=latent_dim).to(self.device)
                optim = torch.optim.Adam(vanilla.parameters(), lr=cfg.experiment.stage_a.lr, weight_decay=1e-5)
                # use single run; no extra wandb.init here
                for epoch in range(cfg.experiment.stage_a.epochs):
                    vanilla.train()
                    total = 0.0
                    for batch in loader:
                        batch = batch[0] if isinstance(batch, (tuple, list)) else batch
                        batch = batch.to(self.device)
                        if batch.dim() == 5:
                            b, t = batch.shape[:2]
                            batch = batch.reshape(b * t, *batch.shape[2:])
                        if effective_arch in ["mlp", "pythae"]:
                            out = vanilla({"data": batch}); loss = out.loss
                        else:
                            out = vanilla(batch); loss = out.loss
                        optim.zero_grad(); loss.backward(); optim.step(); total += loss.item()
                    if wandb.run is not None:
                        wandb.log({"stageA/train_loss": total/len(loader), "stageA/epoch": epoch+1})

                    # Every 10 epochs, log PCA of latent means at t=0 to monitor clustering
                    try:
                        if (epoch + 1) % 10 == 0 or (epoch + 1) == int(cfg.experiment.stage_a.epochs):
                            import matplotlib.pyplot as plt
                            from torch.utils.data import DataLoader as _DL
                            dl0 = None
                            if train_path is not None:
                                from scripts.train_diverse_metric_vae import SpritesDataset as _SD
                                ds0 = _SD(train_path, normalize=False, timestep_only=0)
                                dl0 = _DL(ds0, batch_size=256, shuffle=False)
                            elif data_module is not None and getattr(data_module, 'train_dataset', None) is not None:
                                first_frames = []
                                for sample in data_module.train_dataset:
                                    seq = sample[0] if isinstance(sample, (tuple, list)) else sample
                                    if seq is None:
                                        continue
                                    if seq.dim() == 4:
                                        frame0 = seq[0]
                                    elif seq.dim() == 3:
                                        frame0 = seq
                                    else:
                                        continue
                                    first_frames.append(frame0.unsqueeze(0))
                                    if len(first_frames) >= 6000:
                                        break
                                if first_frames:
                                    frames_tensor = torch.cat(first_frames, dim=0).float()
                                    dl0 = _DL(torch.utils.data.TensorDataset(frames_tensor), batch_size=256, shuffle=False)
                            if dl0 is None:
                                raise RuntimeError("No dataset available for Stage A PCA logging")
                            vanilla.eval()
                            mus = []
                            with torch.no_grad():
                                for xb in dl0:
                                    xb = xb[0] if isinstance(xb, (tuple, list)) else xb
                                    xb = xb.to(self.device)
                                    if effective_arch in ["mlp", "pythae"]:
                                        enc = vanilla.encoder(xb); mu = enc.embedding
                                    else:
                                        mu, _ = vanilla.encode(xb)
                                    mus.append(mu.detach().cpu())
                                    if sum(t.shape[0] for t in mus) >= 6000:
                                        break
                            if mus:
                                Z = torch.cat(mus, dim=0)
                                Zc = Z - Z.mean(dim=0, keepdim=True)
                                U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
                                comp = Vh[:2].T
                                proj = (Zc @ comp).cpu().numpy()
                                plt.figure(figsize=(6, 5))
                                plt.scatter(proj[:, 0], proj[:, 1], s=4, alpha=0.35, c='tab:blue')
                                plt.title(f'Stage A: Latent μ PCA(2) — t=0 (epoch {epoch+1})')
                                plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout()
                                if wandb.run is not None:
                                    wandb.log({"stageA/latent_pca_t0": wandb.Image(plt.gcf())})
                                plt.close()
                            # Also log a PCA over all timesteps on the same cadence
                            try:
                                raw_all = None
                                if train_path is not None and Path(train_path).exists():
                                    raw_all = torch.load(train_path, map_location='cpu')
                                elif data_module is not None and getattr(data_module, 'train_dataset', None) is not None:
                                    seq_samples = []
                                    for sample in data_module.train_dataset:
                                        seq = sample[0] if isinstance(sample, (tuple, list)) else sample
                                        if isinstance(seq, torch.Tensor) and seq.dim() == 4:
                                            seq_samples.append(seq)
                                        if len(seq_samples) >= 512:
                                            break
                                    if seq_samples:
                                        raw_all = torch.stack(seq_samples, dim=0)
                                if isinstance(raw_all, torch.Tensor):
                                    if raw_all.ndim == 5:
                                        B, T = raw_all.shape[:2]
                                        frames = raw_all.view(B * T, *raw_all.shape[2:])
                                    elif raw_all.ndim == 4:
                                        frames = raw_all
                                    else:
                                        frames = None
                                else:
                                    frames = None
                                if isinstance(frames, torch.Tensor):
                                    with torch.no_grad():
                                        xb = frames.to(self.device)
                                        if effective_arch in ["mlp", "pythae"]:
                                            enc = vanilla.encoder(xb); Z_all = enc.embedding
                                        else:
                                            Z_all, _ = vanilla.encode(xb)
                                    Z_all = Z_all.detach().cpu()
                                    Zc_all = Z_all - Z_all.mean(dim=0, keepdim=True)
                                    U_all, S_all, Vh_all = torch.linalg.svd(Zc_all, full_matrices=False)
                                    comp_all = Vh_all[:2].T
                                    proj_all = (Zc_all @ comp_all).cpu().numpy()
                                    plt.figure(figsize=(6, 5))
                                    plt.scatter(proj_all[:, 0], proj_all[:, 1], s=3, alpha=0.3, c='tab:blue')
                                    plt.title(f'Stage A: Latent μ PCA(2) — all timesteps (epoch {epoch+1})')
                                    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout()
                                    if wandb.run is not None:
                                        wandb.log({"stageA/latent_pca_all_t": wandb.Image(plt.gcf())})
                                    plt.close()
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"[Stage A] ⚠️ Periodic PCA failed (epoch {epoch+1}): {e}")

                # Small helper to robustly unwrap batches that may be dict/tuple/list
                def _extract_tensor(batch):
                    # Recursively grab the first tensor we can find.
                    if isinstance(batch, torch.Tensor):
                        return batch
                    if isinstance(batch, dict):
                        if "data" in batch:
                            try:
                                return _extract_tensor(batch["data"])
                            except Exception:
                                pass
                        for v in batch.values():
                            try:
                                return _extract_tensor(v)
                            except Exception:
                                continue
                        raise ValueError("No tensor found in dict batch")
                    if isinstance(batch, (tuple, list)):
                        for elem in batch:
                            try:
                                return _extract_tensor(elem)
                            except Exception:
                                continue
                        raise ValueError("No tensor found in tuple/list batch")
                    raise ValueError(f"Unsupported batch type: {type(batch)}")

                # Encoder sanity check (post-training) before saving artifacts
                try:
                    vanilla.eval()
                    sample_batch = next(iter(loader))
                    sample_batch = _extract_tensor(sample_batch)
                    sample_batch = sample_batch.to(self.device)
                    if effective_arch in ["mlp", "pythae"]:
                        enc_out = vanilla.encoder({"data": sample_batch})
                    else:
                        enc_out = vanilla.encoder(sample_batch)
                    mu_chk = enc_out.embedding
                    logvar_chk = enc_out.log_covariance
                    print(f"[Stage A][ENC CHECK] mu: min={mu_chk.min().item():.4e}, max={mu_chk.max().item():.4e}, mean={mu_chk.mean().item():.4e}, std={mu_chk.std().item():.4e}")
                    print(f"[Stage A][ENC CHECK] log_var: min={logvar_chk.min().item():.4e}, max={logvar_chk.max().item():.4e}, mean={logvar_chk.mean().item():.4e}, std={logvar_chk.std().item():.4e}")
                    if torch.isnan(mu_chk).any() or torch.isinf(mu_chk).any() or torch.isnan(logvar_chk).any() or torch.isinf(logvar_chk).any():
                        print("[Stage A][ENC CHECK] ⚠️ Detected NaN/Inf in encoder outputs after training!")
                    else:
                        print("[Stage A][ENC CHECK] ✅ Encoder outputs are finite.")
                except Exception as enc_exc:
                    print(f"[Stage A][ENC CHECK] ⚠️ Encoder sanity check failed: {enc_exc}")
            # Reconstructions vs real for Stage A (vanilla path)
            try:
                import torchvision.utils as vutils
                vanilla.eval()
                # Sample random, independent frames (not a temporal sequence)
                sample_loader = TorchDataLoader(
                    full_training_dataset,
                    batch_size=8,
                    shuffle=True
                )
                batch = next(iter(sample_loader))
                batch = _extract_tensor(batch)
                batch = batch.to(self.device)
                if batch.dim() == 5:
                    b, t = batch.shape[:2]
                    batch = batch.reshape(b * t, *batch.shape[2:])
                with torch.no_grad():
                    if effective_arch in ["mlp", "pythae"]:
                        out = vanilla({"data": batch}); recon = out.recon_x.clamp(0, 1)
                    else:
                        out = vanilla(batch); recon = out.recon_x.clamp(0, 1)
                grid = vutils.make_grid(
                    torch.cat([batch, recon], dim=0),
                    nrow=8,
                    normalize=False
                )
                if wandb.run is not None:
                    wandb.log({"stageA/recon_vs_real": wandb.Image(grid)})
            except Exception as e:
                print(f"[Stage A] ⚠️ Recon vs real logging failed: {e}")

            # Stage A latent space PCA visualization (μ embeddings)
            try:
                import matplotlib.pyplot as plt
                import plotly.graph_objects as go
                from torch.utils.data import DataLoader as TorchLoader
                from torch.utils.data import ConcatDataset as TorchConcat
                vanilla.eval()
                combined_dataset = full_training_dataset
                full_loader = TorchLoader(combined_dataset, batch_size=256, shuffle=False)
                mus = []
                with torch.no_grad():
                    for xb in full_loader:
                        xb = xb[0] if isinstance(xb, (tuple, list)) else xb
                        xb = xb.to(self.device)
                        if xb.dim() == 5:
                            b, t = xb.shape[:2]
                            xb = xb.reshape(b * t, *xb.shape[2:])
                        if effective_arch in ["mlp", "pythae"]:
                            enc = vanilla.encoder(xb)
                            mu = enc.embedding
                        else:
                            mu, _ = vanilla.encode(xb)
                        mus.append(mu.detach().cpu())
                        if sum(t.shape[0] for t in mus) >= 5000:
                            break
                if mus:
                    Z = torch.cat(mus, dim=0)
                    Zc = Z - Z.mean(dim=0, keepdim=True)
                    # Torch PCA via SVD
                    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
                    comp = Vh[:2].T  # [D,2]
                    proj = (Zc @ comp).numpy()
                    plt.figure(figsize=(6,5))
                    plt.scatter(proj[:,0], proj[:,1], s=4, alpha=0.4, c='tab:blue')
                    plt.title('Stage A: Latent μ PCA(2)')
                    plt.xlabel('PC1'); plt.ylabel('PC2'); plt.tight_layout()
                    if wandb.run is not None:
                        wandb.log({"stageA/latent_pca": wandb.Image(plt.gcf())})
                    plt.close()
            except Exception as e:
                print(f"[Stage A] ⚠️ Latent PCA visualization failed: {e}")

            # Save vanilla VAE to organized Stage A folder
            stageA_paths = get_stage_paths(cfg, 'A', 'VANILLA', arch, latent_dim)
            comp_paths = save_model_components(vanilla, arch, latent_dim, save_dir=str(stageA_paths['base_dir']))
            # Also persist canonical encoder/decoder filenames for downstream stages
            try:
                torch.save(vanilla.encoder.state_dict(), stageA_paths['encoder_path'])
                torch.save(vanilla.decoder.state_dict(), stageA_paths['decoder_path'])
            except Exception as save_exc:
                print(f"[Stage A] ⚠️ Failed to save canonical encoder/decoder: {save_exc}")
            torch.save(vanilla.state_dict(), stageA_paths['model_path'])
            # Mirror fresh Stage A weights to generic outputs/stageA for downstream fallback
            try:
                stageA_generic = Path(cfg.output_dir) / "stageA"
                stageA_generic.mkdir(parents=True, exist_ok=True)
                torch.save(vanilla.encoder.state_dict(), stageA_generic / "encoder.pt")
                torch.save(vanilla.decoder.state_dict(), stageA_generic / "decoder.pt")
                torch.save(vanilla.state_dict(), stageA_generic / "model.pt")
            except Exception as save_exc:
                print(f"[Stage A] ⚠️ Failed to update generic stageA checkpoints: {save_exc}")
            
            # Save Stage A configuration
            beta_value = getattr(cfg.experiment.stage_a, 'beta', None)
            if beta_value is None:
                beta_value = getattr(self.config.settings.model.losses, 'beta', None)
            stageA_config = {
                'stage': 'A',
                'model_type': 'VANILLA',
                'architecture': arch,
                'latent_dim': latent_dim,
                'epochs': cfg.experiment.stage_a.epochs,
                'lr': cfg.experiment.stage_a.lr,
                'beta': beta_value,
                'timestamp': datetime.now().isoformat()
            }
            with open(stageA_paths['config_path'], 'w') as f:
                yaml.dump(stageA_config, f)
            print(f"[Stage A] ✅ Saved Stage A config to {stageA_paths['config_path']}")
            # Log Stage A artifacts (encoder/decoder/model) to WandB for pipeline chaining
            artifacts_enabled = self._wandb_artifacts_enabled(cfg)
            if cfg.wandb.mode != "disabled" and wandb.run is not None and artifacts_enabled:
                try:
                    art = wandb.Artifact(
                        name=f"stageA_vae_{arch}_ld{latent_dim}",
                        type="model",
                        metadata={"stage": "A", "architecture": arch, "latent_dim": latent_dim}
                    )
                    if 'encoder' in comp_paths: art.add_file(comp_paths['encoder'])
                    if 'decoder' in comp_paths: art.add_file(comp_paths['decoder'])
                    if 'model' in comp_paths:   art.add_file(comp_paths['model'])
                    alias = self._resolve_wandb_alias(cfg, 'stage_a_latest', 'stageA_latest')
                    aliases = [alias]
                    wandb.log_artifact(art, aliases=aliases)
                except Exception as e:
                    print(f"[Stage A] ⚠️ Artifact logging failed: {e}")
                    # keep single run open
            self.config = stage_a_base
            cfg = self.config
        else:
            # Stage A was skipped (e.g., running Stage C only). Reuse centralized lookup
            # results to populate the pretrained paths so downstream code can load the
            # correct encoder/decoder automatically.
            if stage_a_data is not None:
                enc_path = str(stage_a_data['encoder_path'])
                dec_path = str(stage_a_data['decoder_path'])
                # Populate compatibility mirrors only when they have not been overridden already.
                ExperimentRunner._set_config_value_if_missing(self.config, "pretrained.encoder_path", enc_path)
                ExperimentRunner._set_config_value_if_missing(self.config, "pretrained.decoder_path", dec_path)
                ExperimentRunner._set_config_value_if_missing(self.config, "settings.model.pretrained.encoder_path", enc_path)
                ExperimentRunner._set_config_value_if_missing(self.config, "settings.model.pretrained.decoder_path", dec_path)
                ExperimentRunner._set_config_value_if_missing(self.config, "model.pretrained.encoder_path", enc_path)
                ExperimentRunner._set_config_value_if_missing(self.config, "model.pretrained.decoder_path", dec_path)

            pretrained_section = getattr(cfg, 'pretrained', None)
            encoder_path = getattr(pretrained_section, 'encoder_path', None) if pretrained_section is not None else None
            decoder_path = getattr(pretrained_section, 'decoder_path', None) if pretrained_section is not None else None

            if encoder_path is None or decoder_path is None:
                missing = []
                if encoder_path is None:
                    missing.append("encoder_path")
                if decoder_path is None:
                    missing.append("decoder_path")
                msg = ", ".join(missing)
                raise RuntimeError(
                    f"Stage A artifacts are required when skipping Stage A, but missing: {msg}. "
                    "Either run Stage A first or provide --overrides pretrained.encoder_path=... "
                    "and pretrained.decoder_path=...."
                )

            comp_paths = {
                'encoder': encoder_path,
                'decoder': decoder_path
            }

        def _log_stage_b_wandb_visuals(metric_state, stageB_model, sample_tensor, data_train_path, canonical_arch, latent_dim, metric_save_path=None, extended_visuals=False):
            if wandb.run is None:
                return
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                import torchvision.utils as vutils
                from pathlib import Path
                centroids = metric_state.get('centroids')
                matrices = metric_state.get('M_matrices', None)
                if matrices is None:
                    matrices = metric_state.get('metric_matrices', None)
                if centroids is None or matrices is None or centroids.numel() == 0:
                    return
                temperature = float(metric_state.get('temperature', cfg.experiment.stage_b.temperature))
                regularization = float(metric_state.get('regularization', cfg.experiment.stage_b.regularization))
                metric_tensor = MetricTensor(
                    latent_dim=latent_dim,
                    temperature=temperature,
                    regularization=regularization,
                    device=self.device
                )
                metric_tensor.load_pretrained(centroids.to(self.device), matrices.to(self.device), temperature, regularization)
                with torch.no_grad():
                    G_inv_centroids = metric_tensor.compute_inverse_metric(centroids.to(self.device))
                det_centroids = torch.linalg.det(G_inv_centroids).clamp(min=1e-12).cpu()
                log_det = det_centroids.log10()
                eigvals = torch.linalg.eigvalsh(G_inv_centroids.cpu())
                min_eigs = eigvals[:, 0].clamp(min=1e-12)
                max_eigs = eigvals[:, -1].clamp(min=1e-12)
                cond = (max_eigs / min_eigs).cpu()
                log_payload = {}
                visuals_scale_mode = str(getattr(cfg.experiment.stage_b, 'visuals_scale_mode', 'percentile') or 'percentile').lower()
                if visuals_scale_mode not in {'percentile', 'global'}:
                    visuals_scale_mode = 'percentile'
                visuals_jitter_setting = getattr(cfg.experiment.stage_b, 'visuals_jitter_centroids', False)
                visuals_jitter_auto = isinstance(visuals_jitter_setting, str) and visuals_jitter_setting.lower() == 'auto'
                visuals_jitter = bool(visuals_jitter_setting) if not visuals_jitter_auto else False
                visuals_filter_t0 = bool(getattr(cfg.experiment.stage_b, 'visuals_filter_centroids_to_t0', False))
                data_diag_payload = {}
                dataset_tensor = None
                if isinstance(sample_tensor, torch.Tensor) and sample_tensor.numel() > 0:
                    dataset_tensor = sample_tensor.detach().cpu()
                elif data_train_path is not None:
                    try:
                        path_obj = Path(data_train_path)
                        if path_obj.exists():
                            raw_ds = torch.load(path_obj, map_location='cpu', weights_only=False)
                            if isinstance(raw_ds, dict):
                                raw_ds = raw_ds.get('data', raw_ds)
                            if isinstance(raw_ds, torch.Tensor):
                                dataset_tensor = raw_ds
                    except Exception as dataset_exc:
                        print(f"[Stage B] ⚠️ Dataset diagnostics skipped: {dataset_exc}")
                if isinstance(dataset_tensor, torch.Tensor) and dataset_tensor.ndim >= 4:
                    ds = dataset_tensor.float()
                    n_sequences = int(ds.shape[0])
                    seq_len = int(ds.shape[1]) if ds.ndim >= 5 else 1
                    frame_shape = tuple(int(x) for x in ds.shape[-3:])
                    print(f"[Stage B] Dataset diagnostics: n_sequences={n_sequences}, seq_len={seq_len}, frame_shape={frame_shape}")
                    try:
                        t0 = ds[:, 0].reshape(n_sequences, -1)
                        quant = torch.round(t0 * 255).to(torch.int16)
                        unique_t0 = int(torch.unique(quant, dim=0).shape[0])
                        print(f"[Stage B] Dataset diagnostics: unique_t0_frames≈{unique_t0}")
                        data_diag_payload['stageB/data/unique_t0_frames'] = unique_t0
                    except Exception as diag_exc:
                        print(f"[Stage B] ⚠️ Could not compute unique t0 frames: {diag_exc}")
                    data_diag_payload.update({
                        'stageB/data/n_sequences': n_sequences,
                        'stageB/data/seq_len': seq_len,
                    })
                    if wandb.run is not None and data_diag_payload:
                        wandb.log(data_diag_payload)

                def _compute_percentile_bounds(log_img):
                    finite = np.isfinite(log_img)
                    if not finite.any():
                        return -12.0, 3.0
                    lo = float(np.nanpercentile(log_img[finite], 5))
                    hi = float(np.nanpercentile(log_img[finite], 95))
                    vmin = max(lo, -12.0)
                    vmax = min(hi, 3.0)
                    if not np.isfinite(vmin):
                        vmin = -12.0
                    if not np.isfinite(vmax):
                        vmax = 3.0
                    if vmax <= vmin:
                        vmax = vmin + 1e-3
                    return vmin, vmax

                if stageB_model is not None:
                    real_batch = None
                    if sample_tensor is not None:
                        if sample_tensor.dim() == 5:
                            real_batch = sample_tensor[:8, 0]
                        elif sample_tensor.dim() == 4:
                            real_batch = sample_tensor[:8]
                    elif data_train_path is not None:
                        path_obj = Path(data_train_path)
                        if path_obj.exists():
                            raw = torch.load(path_obj, map_location='cpu', weights_only=False)
                            if isinstance(raw, dict):
                                raw = raw.get('data', raw)
                            if isinstance(raw, torch.Tensor):
                                if raw.ndim == 5:
                                    real_batch = raw[:8, 0]
                                elif raw.ndim == 4:
                                    real_batch = raw[:8]
                    if real_batch is not None and isinstance(real_batch, torch.Tensor):
                        real_batch = real_batch.float().to(self.device)
                        stageB_model = stageB_model.to(self.device).eval()
                        with torch.no_grad():
                            recon = None
                            if canonical_arch in ['mlp', 'pythae']:
                                out = stageB_model({'data': real_batch})
                                if isinstance(out, dict):
                                    recon = out.get('recon_x')
                                    if recon is None:
                                        recon = out.get('reconstruction')
                                else:
                                    recon = getattr(out, 'recon_x', None)
                                    if recon is None:
                                        recon = getattr(out, 'reconstruction', None)
                            else:
                                out = stageB_model(real_batch)
                                if isinstance(out, dict):
                                    recon = out.get('recon_x')
                                    if recon is None:
                                        recon = out.get('reconstruction')
                                else:
                                    recon = getattr(out, 'recon_x', None)
                                    if recon is None:
                                        recon = getattr(out, 'reconstruction', None)
                            if recon is not None:
                                recon = recon.to(real_batch.device).clamp(0, 1)
                                grid = vutils.make_grid(
                                    torch.cat([real_batch.cpu(), recon.cpu()], dim=0),
                                    nrow=real_batch.shape[0],
                                    normalize=True,
                                    value_range=(0.0, 1.0)
                                )
                                log_payload['stageB/recon_grid'] = wandb.Image(grid, caption='Stage B real (top) vs recon (bottom)')
                centroids_viz = centroids.clone()
                log_det_viz = log_det.clone()
                if visuals_filter_t0:
                    try:
                        mu0 = None
                        for key in ['t0_latents', 'mu0', 'latents_t0', 'stageA_latents_t0']:
                            cand = metric_state.get(key, None)
                            if isinstance(cand, torch.Tensor) and cand.numel() > 0:
                                mu0 = cand
                                break
                        if mu0 is not None:
                            mu0 = mu0.to(self.device)
                            distances = torch.cdist(mu0, centroids.to(self.device))
                            winners = torch.argmin(distances, dim=1)
                            visible_idx = torch.unique(winners).to(centroids.device).long()
                            centroids_viz = centroids.index_select(0, visible_idx)
                            log_det_viz = log_det.index_select(0, visible_idx.cpu())
                            print(f"[Stage B] Visuals filtered to {centroids_viz.shape[0]} centroid(s) (config enabled).")
                        else:
                            print("[Stage B] No t=0 latents available; using all centroids for visuals.")
                    except Exception as filt_exc:
                        print(f"[Stage B] ⚠️ t=0 centroid filtering skipped during visuals: {filt_exc}")
                        centroids_viz = centroids.clone()
                        log_det_viz = log_det.clone()
                total_centroids = int(centroids.shape[0])
                try:
                    unique_centroids_raw = int(torch.unique(centroids_viz, dim=0).shape[0])
                except Exception:
                    unique_centroids_raw = total_centroids
                jitter_sigma = float(getattr(cfg.experiment.stage_b, 'visuals_jitter_sigma', 0.005) or 0.005)
                if not np.isfinite(jitter_sigma) or jitter_sigma <= 0.0:
                    jitter_sigma = 0.005
                visuals_jitter_effective = bool(visuals_jitter)
                if unique_centroids_raw < total_centroids:
                    if visuals_jitter_auto:
                        visuals_jitter_effective = True
                        print(f"[Stage B] Auto-enabling centroid jitter (unique={unique_centroids_raw}/{total_centroids}).")
                    elif not visuals_jitter_effective:
                        print(f"[Stage B] Centroid overlap detected: {unique_centroids_raw}/{total_centroids} unique centroids. Enable experiment.stage_b.visuals_jitter_centroids to separate overlapping markers.")
                log_payload['stageB/centroids/n_unique_raw'] = unique_centroids_raw
                log_det_viz_np = log_det_viz.detach().cpu().numpy()

                # Build plotting grid in PCA(2) coordinates unconditionally for stable alignment
                # Prefer PCA fitted on a dense t=0 latent batch if available in metric_state
                ref_latents = None
                try:
                    # Common keys used to store t=0 latents in the metric payload
                    for k in ['t0_latents', 'z_sample', 'mu0', 'latents_t0']:
                        if isinstance(metric_state.get(k, None), torch.Tensor):
                            ref_latents = metric_state[k]
                            break
                except Exception:
                    ref_latents = None

                if ref_latents is not None and isinstance(ref_latents, torch.Tensor) and ref_latents.numel() > 0:
                    # Subsample to a manageable size for PCA fit
                    ref = ref_latents.detach().to(self.device)
                    if ref.dim() != 2 or ref.size(1) != latent_dim:
                        # Fallback if shape is unexpected
                        ref = centroids.to(self.device)
                else:
                    ref = centroids.to(self.device)

                # If too many points, take up to 2000 for robust PCA fit
                if ref.size(0) > 2000:
                    idx = torch.randperm(ref.size(0), device=ref.device)[:2000]
                    ref = ref[idx]

                ref_mean = ref.mean(0, keepdim=True)
                centered = ref - ref_mean
                # Compute top-2 principal directions using torch PCA
                _, _, V = torch.pca_lowrank(centered, q=2)
                basis = V[:, :2]  # [D, 2]

                # Project centroids into PCA(2) for overlay
                proj_centroids = ((centroids_viz.to(self.device) - ref_mean) @ basis).detach().cpu().numpy()
                proj_centroids_display = proj_centroids.copy()
                if visuals_jitter_effective and proj_centroids_display.size:
                    jitter = np.random.normal(loc=0.0, scale=jitter_sigma, size=proj_centroids_display.shape)
                    proj_centroids_display = proj_centroids_display + jitter
                    print(f'[Stage B] Applied centroid jitter (sigma={jitter_sigma:.4f}) for visuals.')
                # Diagnostics: count unique projected centroids (to detect overlaps)
                try:
                    pc_round = np.round(proj_centroids, 4)
                    n_visible = int(pc_round.shape[0])
                    n_unique = int(np.unique(pc_round, axis=0).shape[0])
                    print(f"[Stage B] Centroid visibility diagnostics: total={total_centroids}, visible={n_visible}, unique_proj={n_unique}")
                    if wandb.run is not None:
                        wandb.log({
                            'stageB/centroids/n_total': total_centroids,
                            'stageB/centroids/n_unique_proj': n_unique,
                            'stageB/centroids/n_unique_raw': unique_centroids_raw,
                            'stageB/centroids/n_visible_after_filter': n_visible
                        })
                except Exception:
                    pass

                # Set extents from reference points AND projected centroids, so all centroids are visible
                ref_proj = (centered @ basis).detach().cpu().numpy()  # already centered by ref_mean
                x_data = np.concatenate([ref_proj[:, 0], proj_centroids[:, 0]]) if ref_proj.size else proj_centroids[:, 0]
                y_data = np.concatenate([ref_proj[:, 1], proj_centroids[:, 1]]) if ref_proj.size else proj_centroids[:, 1]
                # Use strict min/max to GUARANTEE all centroids appear, with gentle padding
                x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
                y_min, y_max = float(np.min(y_data)), float(np.max(y_data))
                pad_x = 0.05 * (x_max - x_min + 1e-6)
                pad_y = 0.05 * (y_max - y_min + 1e-6)
                xs = np.linspace(x_min - pad_x, x_max + pad_x, 300)
                ys = np.linspace(y_min - pad_y, y_max + pad_y, 300)

                # Build grid in PCA space, then map back to latent space: z = mean + U2 @ grid
                # Build grid with PC1 as x and PC2 as y to match the projection of centroids
                gx, gy = np.meshgrid(xs, ys, indexing='xy')  # gx: PC1, gy: PC2
                grid_proj = torch.from_numpy(
                    np.stack([gx, gy], axis=-1).reshape(-1, 2)
                ).float().to(self.device)
                mean_latent = ref_mean
                latent_grid = grid_proj @ basis.T + mean_latent
                # Optional: project random t=0 latent points for consistent overlays
                # (Log-scale figures are generated after T selection; skip redundant baseline plots.)

                # Temperature auto-calibration (median 5-NN distance) and T sweep overlays
                try:
                    # Compute 5-NN distances on reference latents used for PCA fit
                    ref_for_t = ref.detach()
                    if ref_for_t.size(0) > 4000:
                        idx = torch.randperm(ref_for_t.size(0), device=ref_for_t.device)[:4000]
                        ref_for_t = ref_for_t[idx]
                    with torch.no_grad():
                        # Pairwise distances and k=5 neighbor (skip self at idx 0)
                        dmat = torch.cdist(ref_for_t, ref_for_t)
                        dsort, _ = torch.sort(dmat, dim=1)
                        d5 = dsort[:, 5].clamp_min(1e-8) if dsort.size(1) > 5 else dsort[:, -1].clamp_min(1e-8)
                        d5_med = float(d5.median().item())
                        d5_mean = float(d5.mean().item())
                        d5_p10 = float(torch.quantile(d5, 0.10).item())
                        d5_p90 = float(torch.quantile(d5, 0.90).item())
                    T_auto = max(0.05, min(2.5, d5_med))
                    if wandb.run is not None:
                        wandb.log({
                            'stageB/auto_temperature': T_auto,
                            'stageB/nn_stats/d5_median': d5_med,
                            'stageB/nn_stats/d5_mean': d5_mean,
                            'stageB/nn_stats/d5_p10': d5_p10,
                            'stageB/nn_stats/d5_p90': d5_p90,
                        })

                    # Precompute T-invariant density proxy on t=0 latents in PCA(2)
                    ref_eval = ref_for_t  # use possibly downsampled ref set for cost control
                    ref_eval_proj = ((ref_eval - ref_mean) @ basis).detach().cpu().numpy()
                    # 5-NN density proxy: rho5 = 1 / d5^2 (computed in PCA space)
                    try:
                        from scipy.spatial import cKDTree
                        tree = cKDTree(ref_eval_proj)
                        dists, _ = tree.query(ref_eval_proj, k=min(6, max(2, ref_eval_proj.shape[0]-1)))
                        # dists[:,0] is self-distance ~0; use the 6th or last as 5-NN
                        d5_proxy = dists[:, -1]
                    except Exception:
                        # Fallback: brute force
                        R = ref_eval_proj
                        d2 = ((R[:,None,:] - R[None,:,:])**2).sum(-1)
                        np.fill_diagonal(d2, np.inf)
                        d5_proxy = np.partition(d2, 4, axis=1)[:, 4] ** 0.5
                    rho5 = 1.0 / (d5_proxy**2 + 1e-12)
                    log_rho5 = np.log(rho5 + 1e-24)

                    # Far-set radius in PCA space for contrast computation (T-invariant threshold)
                    r_far = max(0.75 * d5_p90, d5_med)

                    # Evaluate and log det heatmaps for a small T sweep including T_auto
                    def _canon_temp(val):
                        return float(np.round(float(val), 6))
                    base_candidates = [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0]
                    sweep_candidates = {_canon_temp(v) for v in base_candidates}
                    try:
                        sweep_candidates.add(_canon_temp(temperature))
                    except Exception:
                        pass
                    if not np.isnan(T_auto):
                        sweep_candidates.add(_canon_temp(T_auto))
                    extra_temps = getattr(cfg.experiment.stage_b, 'extra_temps', None)
                    if isinstance(extra_temps, (list, tuple)):
                        for val in extra_temps:
                            try:
                                sweep_candidates.add(_canon_temp(val))
                            except (TypeError, ValueError):
                                print(f"[Stage B] ⚠️ Could not parse extra temperature value: {val}")
                    sweep_candidates = {t for t in sweep_candidates if np.isfinite(t) and t > 0.0}
                    sweep_T = sorted(sweep_candidates)
                    # Storage for T-selection diagnostics (revised criterion) and slider grids
                    t_sel_rows = []
                    slider_data = []
                    for Tval in sweep_T:
                        try:
                            mt = MetricTensor(
                                latent_dim=latent_dim,
                                temperature=float(Tval),
                                regularization=regularization,
                                device=self.device
                            )
                            mt.load_pretrained(centroids.to(self.device), matrices.to(self.device), float(Tval), regularization)
                            with torch.no_grad():
                                det_grid_T = torch.linalg.det(mt.compute_inverse_metric(latent_grid)).clamp(min=1e-12).cpu().numpy().reshape(len(ys), len(xs))
                                # Also evaluate det at reference evaluation points (latent space)
                                ref_eval_lat = ref_eval
                                if ref_eval_lat.size(0) > 2000:
                                    idx_eval = torch.randperm(ref_eval_lat.size(0), device=ref_eval_lat.device)[:2000]
                                    ref_eval_lat = ref_eval_lat[idx_eval]
                                det_ref = torch.linalg.det(mt.compute_inverse_metric(ref_eval_lat)).clamp(min=1e-12).detach().cpu().numpy()

                            # Correlation with fixed density proxy
                            try:
                                from scipy.stats import spearmanr
                                # Align length if we subsampled ref_eval_lat
                                if ref_eval_lat.size(0) != ref_eval_proj.shape[0]:
                                    # Take same number from ref_eval_proj deterministically
                                    ref_proj_sub = ref_eval_proj[:ref_eval_lat.size(0)]
                                    # recompute rho5 subset via nearest neighbor to keep order simple
                                    # approximate by slicing first N entries
                                    log_rho5_sub = log_rho5[:ref_eval_lat.size(0)]
                                    corr, _ = spearmanr(np.log10(det_ref), log_rho5_sub)
                                else:
                                    corr, _ = spearmanr(np.log10(det_ref), log_rho5)
                                corr = float(corr) if corr == corr else 0.0
                            except Exception:
                                # Fallback to Pearson
                                a = np.log10(det_ref); b = log_rho5[:a.shape[0]]
                                am = a - a.mean(); bm = b - b.mean()
                                corr = float((am * bm).mean() / (am.std() * bm.std() + 1e-12))

                            # Smoothness (lower is better): mean |gradient| of the log-det map
                            det_grid_T_log = np.log10(np.asarray(det_grid_T) + 1e-16)
                            local_vmin, local_vmax = _compute_percentile_bounds(det_grid_T_log)
                            slider_data.append({'T': float(Tval), 'grid_log': det_grid_T_log, 'local_vmin': local_vmin, 'local_vmax': local_vmax})
                            try:
                                gy, gx = np.gradient(det_grid_T_log)
                                smooth = float(np.mean(np.sqrt(gx*gx + gy*gy)))
                            except Exception:
                                smooth = float(np.mean(np.abs(det_grid_T_log)))

                            # Contrast near vs far: median(logdet near) - median(logdet far)
                            # Near: det at ref_eval_lat; Far: det at grid points far from any ref point
                            try:
                                # Build far-mask using KD-tree in PCA space
                                try:
                                    from scipy.spatial import cKDTree
                                    tree = cKDTree(ref_eval_proj)
                                    gx_full, gy_full = np.meshgrid(xs, ys, indexing='xy')
                                    grid_pts = np.stack([gx_full.ravel(), gy_full.ravel()], axis=1)
                                    dmin, _ = tree.query(grid_pts, k=1)
                                except Exception:
                                    # brute-force fallback in chunks
                                    GX, GY = np.meshgrid(xs, ys, indexing='xy')
                                    grid_pts = np.stack([GX.ravel(), GY.ravel()], axis=1)
                                    R = ref_eval_proj
                                    dmin = np.empty(grid_pts.shape[0])
                                    bs = 2000
                                    for s in range(0, grid_pts.shape[0], bs):
                                        gp = grid_pts[s:s+bs]
                                        d2 = ((gp[:,None,:] - R[None,:,:])**2).sum(-1)
                                        dmin[s:s+bs] = np.sqrt(d2.min(axis=1))
                                far_mask = dmin > r_far
                                far_idx = np.flatnonzero(far_mask)
                                if far_idx.size > 0:
                                    # Subsample far indices for cost control
                                    sel = far_idx[::max(1, far_idx.size // 2000)]
                                    latent_far = latent_grid[sel]
                                    with torch.no_grad():
                                        det_far = torch.linalg.det(mt.compute_inverse_metric(latent_far)).clamp(min=1e-12).detach().cpu().numpy()
                                    logdet_far_med = float(np.median(np.log10(det_far)))
                                else:
                                    logdet_far_med = float(np.median(det_grid_T_log))
                                logdet_near_med = float(np.median(np.log10(det_ref)))
                                contrast = logdet_near_med - logdet_far_med
                            except Exception:
                                contrast = 0.0

                            t_sel_rows.append({'T': float(Tval), 'corr_fixed': corr, 'contrast': contrast, 'smoothness': smooth})

                            # Skip per-T logging; data captured for slider and diagnostics
                        except Exception as _e:
                            print(f"[Stage B] ⚠️ T-sweep visualization failed for T={Tval:.3f}: {_e}")
                    if slider_data:
                        if visuals_scale_mode == 'global':
                            global_vmin, global_vmax = -12.0, 3.0
                        else:
                            global_vmin = global_vmax = None
                        for entry in slider_data:
                            if visuals_scale_mode == 'percentile':
                                entry['vmin'] = entry['local_vmin']
                                entry['vmax'] = entry['local_vmax']
                            else:
                                entry['vmin'] = global_vmin
                                entry['vmax'] = global_vmax

                except Exception as e:
                    print(f"[Stage B] ⚠️ Auto-T and sweep logging failed: {e}")
                else:
                    # Choose best T among tested using fixed-density correlation + contrast - small smoothness
                    try:
                        if t_sel_rows:
                            smeds = np.median([r['smoothness'] for r in t_sel_rows]) + 1e-12
                            for r in t_sel_rows:
                                r['score'] = 0.6 * r['corr_fixed'] + 0.4 * r['contrast'] - 0.05 * (r['smoothness'] / smeds)
                            best = max(t_sel_rows, key=lambda r: r['score'])
                            best_T = float(best['T'])
                            best_entry = None
                            for entry in slider_data:
                                if abs(entry['T'] - best_T) < 1e-6:
                                    best_entry = entry
                                    break
                            if wandb.run is not None:
                                wandb_payload = {}
                                table = wandb.Table(columns=['T', 'corr_fixed', 'contrast', 'smoothness', 'score'])
                                for r in sorted(t_sel_rows, key=lambda x: x['T']):
                                    table.add_data(r['T'], r['corr_fixed'], r['contrast'], r['smoothness'], r.get('score', 0.0))
                                wandb_payload['stageB/T_selection_fixed'] = table
                                wandb_payload['stageB/selected_temperature_fixed'] = best_T
                                if extended_visuals:
                                    try:
                                        import plotly.graph_objects as go
                                        slider_sorted = sorted(slider_data, key=lambda d: d['T'])
                                        if slider_sorted:
                                            active_idx = 0
                                            for idx, entry in enumerate(slider_sorted):
                                                if abs(entry['T'] - best_T) < 1e-6:
                                                    active_idx = idx
                                                    break
                                            initial = slider_sorted[active_idx] if active_idx < len(slider_sorted) else slider_sorted[0]
                                            fig_slider = go.Figure()
                                            heatmap_kwargs = dict(
                                                z=initial['grid_log'], x=xs, y=ys,
                                                colorscale='Viridis', showscale=True,
                                                colorbar=dict(title='log10 det(G^{-1})')
                                            )
                                            if initial.get('vmin') is not None and initial.get('vmax') is not None:
                                                heatmap_kwargs['zmin'] = initial['vmin']
                                                heatmap_kwargs['zmax'] = initial['vmax']
                                            fig_slider.add_trace(go.Heatmap(**heatmap_kwargs))
                                            if proj_centroids_display.size:
                                                fig_slider.add_trace(go.Scattergl(
                                                    x=proj_centroids_display[:, 0], y=proj_centroids_display[:, 1], mode='markers',
                                                    marker=dict(
                                                        size=6,
                                                        color=log_det_viz_np,
                                                        colorscale='Magma',
                                                        showscale=False
                                                    ),
                                                    name='centroids',
                                                    hovertemplate='centroid<br>log10 det: %{marker.color:.2f}<extra></extra>'
                                                ))
                                            if isinstance(ref_eval_proj, np.ndarray) and ref_eval_proj.size:
                                                fig_slider.add_trace(go.Scattergl(
                                                    x=ref_eval_proj[:, 0], y=ref_eval_proj[:, 1], mode='markers',
                                                    marker=dict(color='rgba(255,255,255,0.35)', size=2),
                                                    name='t=0 samples',
                                                    hoverinfo='skip'
                                                ))
                                            try:
                                                stride = max(1, len(xs)//20)
                                                gx_s = xs[::stride]; gy_s = ys[::stride]
                                                GXs, GYs = np.meshgrid(gx_s, gy_s)
                                                fig_slider.add_trace(go.Scattergl(
                                                    x=GXs.ravel(), y=GYs.ravel(), mode='markers',
                                                    marker=dict(color='black', size=2, opacity=0.2), name='grid samples'
                                                ))
                                            except Exception:
                                                pass
                                            frames = []
                                            for entry in slider_sorted:
                                                frame_name = f"T={entry['T']:.2f}"
                                                frames.append(go.Frame(
                                                    data=[go.Heatmap(z=entry['grid_log'], zmin=entry.get('vmin'), zmax=entry.get('vmax'))],
                                                    name=frame_name,
                                                    traces=[0]
                                                ))
                                            fig_slider.frames = frames
                                            steps = []
                                            for idx, entry in enumerate(slider_sorted):
                                                label = f"T={entry['T']:.2f}"
                                                steps.append({
                                                    'method': 'animate',
                                                    'label': label,
                                                    'args': [[label], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}, 'transition': {'duration': 0}}]
                                                })
                                            fig_slider.update_layout(
                                                title=f'Stage B: log10 det(G^{{-1}}) (interactive)',
                                                xaxis_title='z1', yaxis_title='z2', width=720, height=560,
                                                sliders=[{'active': active_idx, 'steps': steps, 'x': 0.1, 'y': -0.08, 'len': 0.8, 'currentvalue': {'prefix': 'T=', 'visible': True}}],
                                                updatemenus=[{'type': 'buttons', 'showactive': False, 'x': 1.0, 'y': 1.05, 'xanchor': 'right', 'yanchor': 'top',
                                                              'buttons': [{'label': 'Play', 'method': 'animate', 'args': [None, {'fromcurrent': True}]}]}]
                                            )
                                            wandb_payload['stageB/visuals/n_slider_frames'] = len(slider_sorted)
                                            wandb_payload['stageB/plotly_logdet_heatmap_slider'] = fig_slider
                                    except Exception as slider_exc:
                                        print(f"[Stage B] ⚠️ Could not build slider visualization: {slider_exc}")
                                if wandb_payload:
                                    wandb.log(wandb_payload)
                            # Persist selected temperature into metric artifact
                            try:
                                if metric_save_path is not None:
                                    state = torch.load(metric_save_path, map_location='cpu', weights_only=False)
                                    state['selected_temperature'] = best_T
                                    state['temperature'] = best_T
                                    torch.save(state, metric_save_path)
                                    print(f"[Stage B] ✅ Persisted selected T={best_T:.3f} into metric: {metric_save_path}")
                                    try:
                                        metric_state['selected_temperature'] = best_T
                                        metric_state['temperature'] = best_T
                                    except Exception:
                                        pass
                            except Exception as persist_exc:
                                print(f"[Stage B] ⚠️ Could not persist selected T: {persist_exc}")

                            print(f"[Stage B] Selected temperature (fixed) T*={best_T:.3f} (corr={best['corr_fixed']:.3f}, contrast={best['contrast']:.3f}, smooth={best['smoothness']:.4f}, score={best['score']:.3f})")
                    except Exception as e:
                        print(f"[Stage B] ⚠️ Could not compute best-T selection: {e}")
                # Note: linear-scale interactive plots removed in favour of consolidated log-scale slider.
                wandb.log(log_payload)
            except Exception as exc:
                print(f"[Stage B] Enhanced WandB logging failed: {exc}")

        # Stage B: Metric learning at t=0
        if getattr(cfg.experiment, 'run_stage_b', True):
            stage_b_base = self.config
            self.config = OmegaConf.create(OmegaConf.to_container(stage_b_base, resolve=False))
            cfg = self.config
            print("\n=== [Stage B] Metric learning at t=0 ===")
            arch = cfg.model.encoder.architecture
            latent_dim = cfg.model.latent_dim
            metric_impl = cfg.experiment.stage_b.implementation

            def _canonical_architecture(name: str) -> str:
                if not isinstance(name, str):
                    return name
                lowered = name.lower()
                return lowered.replace('_gray', '')

            canonical_arch = _canonical_architecture(arch)

            data_train_path = getattr(cfg.data, 'train_path', None)
            data_test_path = getattr(cfg.data, 'test_path', None)
            stage_b_train_tensor = None

            def _sequence_tensor(dataset):
                if dataset is None:
                    return None
                sequences = []
                for item in dataset:
                    seq = item[0] if isinstance(item, (tuple, list)) else item
                    if seq is None:
                        continue
                    if isinstance(seq, dict):
                        seq = seq.get('data', None)
                    if seq is None:
                        continue
                    if seq.dim() == 5:
                        sequences.append(seq)
                    elif seq.dim() == 4:
                        sequences.append(seq.unsqueeze(0))
                    elif seq.dim() == 3:
                        sequences.append(seq.unsqueeze(0))
                    else:
                        raise ValueError(f"Unsupported sequence rank {seq.dim()} for Stage B dataset export")
                if not sequences:
                    return None
                return torch.cat(sequences, dim=0)

            # Get organized Stage B paths
            stageB_paths = get_stage_paths(cfg, 'B', metric_impl.upper(), arch, latent_dim)

            generated_dataset_path = None
            if data_train_path is None or (isinstance(data_train_path, str) and not Path(data_train_path).exists()):
                try:
                    stage_b_datamodule = build_data_module(cfg.data)
                    stage_b_datamodule.setup("fit", getattr(cfg, "training", None))
                    stage_b_train_tensor = _sequence_tensor(getattr(stage_b_datamodule, 'train_dataset', None))
                    if stage_b_train_tensor is not None:
                        generated_dataset_path = stageB_paths['base_dir'] / 'stageB_train_sequences.pt'
                        generated_dataset_path.parent.mkdir(parents=True, exist_ok=True)
                        torch.save(stage_b_train_tensor, generated_dataset_path)
                        data_train_path = str(generated_dataset_path)
                        print(f"[Stage B] Generated train dataset tensor at {data_train_path}")
                except Exception as dm_err:
                    print(f"[Stage B] ⚠️ Could not generate in-memory dataset for Stage B metric extraction: {dm_err}")

            if (metric_impl in ('rhvae', 'precision')) and data_train_path is None:
                raise RuntimeError("Stage B requires a train dataset (data.train_path) but none was provided and automatic generation failed")
            
            # Use centralized Stage A data (already looked up)
            # stage_a_data is already available from centralized lookup
            
            # Fix the metric_file reference for WandB logging
            metric_file = stageB_paths['metric_path']
            reused_stageA_metric = False
            
            print(f"\n=== [Stage B] COMPONENT LOADING DEBUG ===")
            print(f"[Stage B] Architecture: {arch}, Latent dim: {latent_dim}")
            print(f"[Stage B] Metric implementation: {metric_impl}")
            print(f"[Stage B] Stage A data found: {stage_a_data is not None}")
            
            if stage_a_data:
                print(f"[Stage B] Stage A paths:")
                print(f"  - Base dir: {stage_a_data['base_dir']}")
                print(f"  - Encoder: {stage_a_data['encoder_path']}")
                print(f"  - Decoder: {stage_a_data['decoder_path']}")
                print(f"  - Config: {stage_a_data['config_path']}")
                if stage_a_data.get('metric_path'):
                    print(f"  - Metric: {stage_a_data['metric_path']}")
                
                # Verify architecture and latent_dim consistency with Stage A
                print(f"[Stage B] Architecture/Latent Dim Verification:")
                try:
                    with open(stage_a_data['config_path'], 'r') as f:
                        stage_a_config = yaml.safe_load(f)
                    stage_a_arch = stage_a_config.get('architecture', 'unknown')
                    stage_a_ld = stage_a_config.get('latent_dim', 'unknown')
                    print(f"  - Stage A: arch={stage_a_arch}, latent_dim={stage_a_ld}")
                    print(f"  - Stage B: arch={arch}, latent_dim={latent_dim}")
                    if stage_a_arch != arch or stage_a_ld != latent_dim:
                        print(f"  - ⚠️ MISMATCH DETECTED! Stage A and Stage B have different architecture/latent_dim!")
                        print(f"  - Forcing Stage B to use Stage A parameters...")
                        # Force Stage B to use Stage A's architecture and latent_dim
                        arch = stage_a_arch
                        latent_dim = stage_a_ld
                        canonical_arch = _canonical_architecture(arch)
                        print(f"  - ✅ Updated Stage B to use arch={arch}, latent_dim={latent_dim}")
                    else:
                        print(f"  - ✅ Architecture and latent_dim match between Stage A and Stage B")
                except Exception as e:
                    print(f"  - ⚠️ Could not verify Stage A config: {e}")
            print(f"=== [Stage B] END COMPONENT DEBUG ===\n")
            
            if stage_a_data is not None and metric_impl == 'rhvae':
                try:
                    _st = torch.load(stage_a_data['metric_path'], map_location='cpu', weights_only=False)
                    if _st is not None and 'centroids' in _st and ('metric_matrices' in _st or 'M_matrices' in _st or 'inverse_metrics' in _st):
                        print(f"[Stage B] Reusing RHVAE metric from Stage A: {stage_a_data['metric_path']}")
                        metric_path = str(stage_a_data['metric_path'])
                        reused_stageA_metric = True
                    else:
                        print("[Stage B] Existing metric file is incompatible; re-extracting.")
                except Exception as _e:
                    print(f"[Stage B] Could not reuse Stage A metric ({_e}); re-extracting.")
            if metric_impl == 'rhvae' and not reused_stageA_metric:
                # Load model from Stage A if available
                if stage_a_data is not None:
                    print(f"[Stage B] ✅ Loading model from Stage A:")
                    print(f"  - Encoder path: {stage_a_data['encoder_path']}")
                    print(f"  - Decoder path: {stage_a_data['decoder_path']}")
                    stageB_model = create_vanilla(canonical_arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim).to(self.device)
                    enc_state = torch.load(stage_a_data['encoder_path'], map_location=self.device, weights_only=False)
                    # Strip possible "encoder." prefix from keys when loading raw state dicts
                    if isinstance(enc_state, dict):
                        enc_state = {k.replace("encoder.", "", 1): v for k, v in enc_state.items()}
                    stageB_model.encoder.load_state_dict(enc_state)
                    dec_state = torch.load(stage_a_data['decoder_path'], map_location=self.device, weights_only=False)
                    if isinstance(dec_state, dict):
                        dec_state = {k.replace("decoder.", "", 1): v for k, v in dec_state.items()}
                    stageB_model.decoder.load_state_dict(dec_state)
                    print(f"[Stage B] ✅ Successfully loaded Stage A encoder/decoder")
                else:
                    print(f"[Stage B] ⚠️ No Stage A data found, using fallback model")
                    # Fallback to existing model
                    stageB_model = (
                        rh_exp.model if 'rh_exp' in locals() and hasattr(rh_exp, 'model') else
                        (vanilla if 'vanilla' in locals() else create_vanilla(canonical_arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim).to(self.device))
                    )
                
                # Force 150 centroids for Stage B extraction (RHVAE implementation)
                n_centroids_local = 100
                print(f"[Stage B] Overriding number of centroids to {n_centroids_local}")
                metric_path = extract_diverse_metric(
                    model=stageB_model,
                    architecture=canonical_arch,
                    latent_dim=latent_dim,
                    temperature=cfg.experiment.stage_b.temperature,
                    regularization=cfg.experiment.stage_b.regularization,
                    num_centroids=n_centroids_local,
                    save_dir=str(stageB_paths['base_dir']),
                    input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]),
                    data_path=data_train_path,
                    timestep_only=cfg.experiment.stage_b.use_timestep,
                    standardize_latents=cfg.experiment.stage_b.standardize_latents,
                    centroid_method=cfg.experiment.stage_b.centroid_method,
                    neighbor_mode=cfg.experiment.stage_b.neighbor_mode,
                    knn_k=cfg.experiment.stage_b.knn_k,
                )
            elif metric_impl == 'rhvae' and reused_stageA_metric:
                # When reusing Stage A metric, we need to copy it to Stage B location
                metric_path = str(stage_a_data['metric_path'])
                print(f"[Stage B] ✅ Reusing Stage A metric: {metric_path}")
            elif metric_impl == 'precision':
                # Load model from Stage A if available
                if stage_a_data is not None:
                    print(f"[Stage B] Loading model from Stage A for precision metric: {stage_a_data['encoder_path']}")
                    stageB_model = create_vanilla(canonical_arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim).to(self.device)
                    stageB_model.encoder.load_state_dict(torch.load(stage_a_data['encoder_path'], map_location=self.device, weights_only=False))
                    stageB_model.decoder.load_state_dict(torch.load(stage_a_data['decoder_path'], map_location=self.device, weights_only=False))
                else:
                    stageB_model = vanilla if 'vanilla' in locals() else create_vanilla(canonical_arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim).to(self.device)
                
                # Precision metric from posterior: reuse extraction with local KNN covariance -> invert
                # Force 150 centroids for Stage B extraction (per request)
                n_centroids_local = 100
                print(f"[Stage B] Overriding number of centroids to {n_centroids_local}")
                metric_path = extract_diverse_metric(
                    model=stageB_model,
                    architecture=canonical_arch,
                    latent_dim=latent_dim,
                    temperature=cfg.experiment.stage_b.temperature,
                    regularization=cfg.experiment.stage_b.regularization,
                    num_centroids=n_centroids_local,
                    save_dir=str(stageB_paths['base_dir']),
                    input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]),
                    data_path=data_train_path,
                    timestep_only=cfg.experiment.stage_b.use_timestep,
                    standardize_latents=cfg.experiment.stage_b.standardize_latents,
                    centroid_method='kmeans',
                    neighbor_mode='knn',
                    knn_k=max(100, cfg.experiment.stage_b.knn_k),
                    normalize_M='none',
                )
                # Convert saved covariance matrices to precision (G^{-1}) for Stage C compatibility
                try:
                    state = torch.load(metric_path, map_location=self.device, weights_only=False)
                    if 'M_matrices' in state and 'inverse_metrics' not in state:
                        cov = state['M_matrices']
                        precision = torch.linalg.inv(cov)
                        state['M_matrices'] = precision
                        state['extraction_method'] = str(state.get('extraction_method', '')) + '_precision'
                        torch.save(state, metric_path)
                        print("[Stage B] Converted covariance matrices to precision (G^{-1}) for precision metric.")
                except Exception as e:
                    print(f"[Stage B] ⚠️ Precision conversion failed: {e}")
            else:
                raise ValueError(f"Unknown metric implementation: {metric_impl}")
            # Copy metric to organized Stage B location
            if Path(metric_path) != stageB_paths['metric_path']:
                try:
                    import shutil
                    # Load the metric data to ensure we preserve all fields including t0_latents
                    metric_data = torch.load(metric_path, map_location='cpu', weights_only=False)
                    # Save to Stage B location with all data preserved
                    torch.save(metric_data, stageB_paths['metric_path'])
                    metric_path = str(stageB_paths['metric_path'])
                    print(f"[Stage B] ✅ Copied metric with all data to: {metric_path}")
                except Exception as e:
                    print(f"[Stage B] ⚠️ Could not copy metric to organized location: {e}")
                    # Fallback to direct file copy
                    try:
                        shutil.copyfile(metric_path, stageB_paths['metric_path'])
                        metric_path = str(stageB_paths['metric_path'])
                    except Exception as e2:
                        print(f"[Stage B] ⚠️ Direct file copy also failed: {e2}")
                except Exception as e:
                    print(f"[Stage B] ⚠️ Could not copy metric to organized location: {e}")
            
            try:
                metric_state_for_logging = locals().get('metric_data', None)
                if metric_state_for_logging is None:
                    metric_state_for_logging = torch.load(metric_path, map_location='cpu', weights_only=False)
                # Extended visuals only when running the full pipeline (A + B + C)
                stage_abc = (
                    bool(getattr(cfg.experiment, 'run_stage_a', True)) and
                    bool(getattr(cfg.experiment, 'run_stage_b', True)) and
                    bool(getattr(cfg.experiment, 'run_stage_c', True))
                )
                _log_stage_b_wandb_visuals(
                    metric_state_for_logging,
                    stageB_model if 'stageB_model' in locals() else None,
                    stage_b_train_tensor,
                    data_train_path,
                    canonical_arch,
                    int(latent_dim),
                    metric_save_path=stageB_paths['metric_path'],
                    extended_visuals=stage_abc
                )
            except Exception:
                pass

            # Save Stage B configuration
            metric_temp_cfg = cfg.experiment.stage_b.temperature
            metric_temp_selected = metric_temp_cfg
            try:
                persisted_state = torch.load(stageB_paths['metric_path'], map_location='cpu', weights_only=False)
                metric_temp_cfg = float(persisted_state.get('temperature', metric_temp_cfg))
                metric_temp_selected = float(persisted_state.get('selected_temperature', metric_temp_cfg))
            except Exception:
                pass

            stageB_config = {
                'stage': 'B',
                'model_type': metric_impl.upper(),
                'architecture': arch,
                'latent_dim': latent_dim,
                'temperature': metric_temp_cfg,
                'selected_temperature': metric_temp_selected,
                'regularization': cfg.experiment.stage_b.regularization,
                'n_centroids': 100,
                'centroid_method': cfg.experiment.stage_b.centroid_method,
                'neighbor_mode': cfg.experiment.stage_b.neighbor_mode,
                'knn_k': cfg.experiment.stage_b.knn_k,
                'use_timestep': cfg.experiment.stage_b.use_timestep,
                'standardize_latents': cfg.experiment.stage_b.standardize_latents,
                'stage_a_source': str(stage_a_data['base_dir']) if stage_a_data else None,
                'timestamp': datetime.now().isoformat()
            }
            with open(stageB_paths['config_path'], 'w') as f:
                yaml.dump(stageB_config, f)
            print(f"[Stage B] ✅ Saved Stage B config to {stageB_paths['config_path']}")
            print(f"[Stage B] ✅ Saved metric checkpoint: {stageB_paths['metric_path']}")
            # Stage B basic visuals (eigenvalue/condition/heatmaps) — only when running Stage B standalone
            stage_b_standalone = (
                bool(getattr(cfg.experiment, 'run_stage_b', True)) and
                not bool(getattr(cfg.experiment, 'run_stage_a', True)) and
                not bool(getattr(cfg.experiment, 'run_stage_c', True))
            )
            if cfg.wandb.mode != "disabled" and stage_b_standalone:
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    plt.style.use('default')
                    sns.set_palette("husl")
                    state = torch.load(metric_path, map_location='cpu', weights_only=False)
                    M_matrices = state.get('M_matrices')
                    if M_matrices is None:
                        M_matrices = state.get('metric_matrices')
                    centroids = state.get('centroids')
                    if M_matrices is not None and centroids is not None:
                        eigenvals = torch.linalg.eigvals(M_matrices).real
                        min_eigenvals = eigenvals.min(dim=-1)[0]
                        max_eigenvals = eigenvals.max(dim=-1)[0]
                        mean_eigenvals = eigenvals.mean(dim=-1)
                        cond_nums = max_eigenvals / (min_eigenvals + 1e-12)
                        determinants = torch.linalg.det(M_matrices)
                        # 1) Eigenvalue distributions
                        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                        fig.suptitle('Stage B Metric — Eigenvalue & Condition Analysis', fontsize=16, fontweight='bold')
                        axes[0,0].hist(min_eigenvals.numpy(), bins=40, color='red', edgecolor='black'); axes[0,0].set_title('Min Eigenvalue')
                        axes[0,1].hist(max_eigenvals.numpy(), bins=40, color='blue', edgecolor='black'); axes[0,1].set_title('Max Eigenvalue')
                        axes[0,2].hist(mean_eigenvals.numpy(), bins=40, color='green', edgecolor='black'); axes[0,2].set_title('Mean Eigenvalue')
                        axes[1,0].hist(cond_nums.numpy(), bins=40, color='orange', edgecolor='black'); axes[1,0].set_title('Condition Number')
                        log_dets = torch.log10(torch.abs(determinants) + 1e-50)
                        axes[1,1].hist(log_dets.numpy(), bins=40, color='purple', edgecolor='black'); axes[1,1].set_title('Log₁₀|Determinant|')
                        eig_spread = (max_eigenvals - min_eigenvals).numpy()
                        axes[1,2].hist(eig_spread, bins=40, color='cyan', edgecolor='black'); axes[1,2].set_title('Eigenvalue Spread')
                        plt.tight_layout()
                        wandb.log({"stageB/metric_eigen_diagnostics": wandb.Image(fig)})
                        plt.close(fig)
                        # 2) Heatmaps
                        n_heatmaps = min(6, len(M_matrices))
                        fig, axes = plt.subplots(1, n_heatmaps, figsize=(4*n_heatmaps, 4))
                        for i in range(n_heatmaps):
                            im = axes[i].imshow(M_matrices[i].numpy(), cmap='RdYlBu_r', aspect='auto')
                            axes[i].set_title(f'Matrix {i}')
                            plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
                        plt.tight_layout()
                        wandb.log({"stageB/metric_heatmaps": wandb.Image(fig)})
                        plt.close(fig)
                        # 3) Centroid stats
                        centroid_norms = torch.norm(centroids, dim=1)
                        pairwise_dists = torch.cdist(centroids, centroids)
                        triu_mask = torch.triu(torch.ones_like(pairwise_dists, dtype=bool), diagonal=1)
                        pairwise_vals = pairwise_dists[triu_mask]
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        axes[0].hist(centroid_norms.numpy(), bins=30, color='red', edgecolor='black'); axes[0].set_title('Centroid Norms')
                        axes[1].hist(pairwise_vals.numpy(), bins=40, color='blue', edgecolor='black'); axes[1].set_title('Pairwise Distances')
                        plt.tight_layout()
                        wandb.log({"stageB/centroid_stats": wandb.Image(fig)})
                        plt.close(fig)
                except Exception as e:
                    print(f"[Stage B] ⚠️ Visual diagnostics failed: {e}")

            # Stage B PCA(2) heatmap of det(G^{-1}) in latent subspace with overlays
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                from rlvae.models.components.metric_loader import MetricLoader
                loader = MetricLoader(device=self.device)
                blob = loader.load_from_file(metric_path, None, None)
                C = blob['centroids'].to(self.device)
                M = blob['metric_matrices'].to(self.device)
                # Use cached t0 latents if available
                state = torch.load(metric_path, map_location='cpu', weights_only=False)
                Z0 = state.get('t0_latents', None)
                if Z0 is None:
                    print("[Stage B] No t0_latents cached; sampling 4000 points around centroids for PCA basis")
                    Z0 = C.detach().cpu()
                else:
                    Z0 = Z0.detach().cpu()
                # PCA basis
                Zc = Z0 - Z0.mean(dim=0, keepdim=True)
                U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
                comp = Vh[:2].T  # [D,2]
                mean = Z0.mean(dim=0)
                # Grid in PCA space
                Proj = (Zc @ comp).numpy()
                x_min, x_max = np.percentile(Proj[:,0], [1, 99])
                y_min, y_max = np.percentile(Proj[:,1], [1, 99])
                xs = np.linspace(x_min, x_max, 100)
                ys = np.linspace(y_min, y_max, 100)
                XX, YY = np.meshgrid(xs, ys)
                pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
                # Back to latent space: z = mean + comp @ p
                comp_t = comp
                Zgrid = mean.unsqueeze(0) + torch.from_numpy(pts).to(mean.dtype) @ comp_t.T
                Zgrid = Zgrid.to(self.device)
                # Compute det(G^{-1}) using MetricTensor rebuilt from loaded data
                from rlvae.models.components.metric_tensor import MetricTensor as _MetricTensor
                mt = _MetricTensor(latent_dim=Zgrid.shape[1], device=self.device)
                mt.load_pretrained(C, M, blob['temperature'], blob['regularization'])
                Ginvg = mt.compute_inverse_metric(Zgrid)
                half_logdet = half_logdet_volume(Ginvg, 'ginv')
                det_vals = torch.exp(2.0 * half_logdet).detach().cpu().numpy().reshape(XX.shape)
                # RHMC samples overlay if available
                rhmc_pts = None
                try:
                    from src.models.samplers.hmc_sampler import RHVAEVolumeElementHMCSampler
                    sampler = RHVAEVolumeElementHMCSampler(mt, device=self.device)
                    rhmc = sampler.sample(num_samples=1000)
                    rhmc_centered = (rhmc.detach().cpu() - mean)
                    rhmc_proj = (rhmc_centered @ comp).numpy()
                    rhmc_pts = rhmc_proj
                except Exception:
                    pass
                # Build figure
                plt.figure(figsize=(7,6))
                plt.imshow(det_vals, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis', aspect='auto')
                # overlay t0 latents and centroids
                P0 = ( (Z0 - mean) @ comp ).numpy()
                plt.scatter(P0[:,0], P0[:,1], s=3, c='white', alpha=0.15, label='t=0 latents')
                Pc = ( (C.detach().cpu() - mean) @ comp ).numpy()
                plt.scatter(Pc[:,0], Pc[:,1], s=40, c='red', marker='*', label='centroids')
                if rhmc_pts is not None:
                    plt.scatter(rhmc_pts[:,0], rhmc_pts[:,1], s=3, c='deepskyblue', alpha=0.3, label='RHMC samples')
                plt.colorbar(label='det(G^{-1}) (PCA subspace)')
                plt.title('Stage B: det(G^{-1}) in PCA(2) subspace (t=0)')
                plt.legend(loc='upper right', fontsize=8)
                plt.tight_layout()
                if wandb.run is not None:
                    wandb.log({"stageB/pca_det_heatmap": wandb.Image(plt.gcf())})
                plt.close()
            except Exception as e:
                print(f"[Stage B] ⚠️ PCA det(G^-1) heatmap failed: {e}")

            # Stage B WandB logging (artifact)
            if cfg.wandb.mode != "disabled":
                try:
                    state = torch.load(metric_path, map_location='cpu', weights_only=False)
                    centroids = state.get('centroids')
                    # Log metric artifact for reuse in Stage C or future runs
                    try:
                        art = wandb.Artifact(
                            name=f"stageB_metric_{arch}_ld{latent_dim}",
                            type="metric",
                            metadata={"stage": "B", "architecture": arch, "latent_dim": latent_dim, "implementation": metric_impl}
                        )
                        art.add_file(str(metric_path))
                        alias = self._resolve_wandb_alias(cfg, 'stage_b_latest', 'stageB_latest')
                        aliases = [alias]
                        wandb.log_artifact(art, aliases=aliases)
                    except Exception as eart:
                        print(f"[Stage B] ⚠️ Artifact logging failed: {eart}")
                    # keep run open
                except Exception as e:
                    print(f"[Stage B] ⚠️ WandB logging failed: {e}")
        else:
            # Only check metric_file if it's defined (i.e., if we're in Stage B context)
            if 'metric_file' in locals():
                assert metric_file.exists(), f"Metric file not found: {metric_file}"

        # RHMC sampling at t=0 (use volume-element sampler)
        if getattr(cfg.experiment, 'run_sampling', True) and cfg.sampling.enabled:
            print("\n=== RHMC sampling at t=0 ===")
            # Define arch and latent_dim for sampling if not already defined
            if 'arch' not in locals():
                arch = getattr(cfg.model.encoder, 'architecture', 'mlp')
            if 'latent_dim' not in locals():
                latent_dim = getattr(cfg.model, 'latent_dim', 16)
            if 'metric_impl' not in locals():
                metric_impl = getattr(cfg.experiment.stage_b, 'implementation', 'rhvae')
            
            # Reuse the unified sampler from scripts tester for reliability
            from src.models.components.metric_loader import MetricLoader
            from src.models.samplers.hmc_sampler import RHVAEVolumeElementHMCSampler
            loader = MetricLoader(device=self.device)
            # Use Stage B metric if available, otherwise fallback
            # Use centralized Stage B data (already looked up)
            stage_b_data_for_sampling = stage_b_data
            # Define metric_file for sampling if not already defined
            if 'metric_file' not in locals():
                metric_file = stage_b_data_for_sampling['metric_path'] if stage_b_data_for_sampling else None
            metric_path_for_sampling = stage_b_data_for_sampling['metric_path'] if stage_b_data_for_sampling else metric_file
            blob = loader.load_from_file(str(metric_path_for_sampling), None, None)
            # Optionally filter centroids to those used at timestep 0 (RHVAE-style t=0 winners)
            visuals_filter_t0 = bool(getattr(cfg.experiment.stage_b, 'visuals_filter_centroids_to_t0', False))
            C_all = blob['centroids'].to(self.device)
            M_all = blob['metric_matrices'].to(self.device)
            C_use, M_use = C_all, M_all
            if visuals_filter_t0:
                # Prefer precomputed t=0 latents from Stage A payload for deterministic filtering
                try:
                    state = torch.load(metric_path_for_sampling, map_location='cpu', weights_only=False)
                    mu0 = state.get('t0_latents', None)
                    if mu0 is not None:
                        mu0 = mu0.to(self.device)
                        d2 = torch.cdist(mu0, C_all)  # [N0, K]
                        winners = torch.argmin(d2, dim=1)
                        used_ids = torch.unique(winners)
                        C_use = C_all[used_ids]
                        M_use = M_all[used_ids]
                        print(f"[Stage B] Using {C_use.shape[0]} t0-relevant centroids (from {C_all.shape[0]} total) for visuals.")
                    else:
                        print("[Stage B] No t=0 latents in metric payload; using all centroids for visuals.")
                except Exception as e:
                    print(f"[Stage B] ⚠️ t=0 centroid filtering skipped: {e}")
                    
            class _MetricStub:
                def __init__(
                    self,
                    centroids,
                    M,
                    T,
                    lbd,
                    device,
                    normalize_weight_sum=False,
                    weight_kernel: str = 'isotropic',
                    weight_metric_normalization: str = 'trace',
                    topk_weights: int | None = None,
                    metric_scale: float = 1.0,
                    regularization_mode: str = 'precision',
                ):
                    self.device = device
                    self.centroids_tens = centroids.to(device)
                    self.M_raw = M.to(device)
                    base_dtype = self.M_raw.dtype
                    self.temperature = torch.as_tensor(T, device=device, dtype=base_dtype)
                    self.regularization = torch.as_tensor(lbd, device=device, dtype=base_dtype)
                    self.normalize_weight_sum = bool(normalize_weight_sum)
                    self.weight_kernel = (weight_kernel or 'isotropic').lower()
                    self.weight_metric_normalization = (weight_metric_normalization or 'trace').lower()
                    self.topk_weights = int(topk_weights) if topk_weights is not None else None
                    if self.topk_weights is not None and self.topk_weights <= 0:
                        self.topk_weights = None
                    self.metric_scale = float(metric_scale)
                    self.regularization_mode = str(regularization_mode).lower()
                    if self.regularization_mode not in {'precision', 'metric'}:
                        raise ValueError("Metric stub: regularization_mode must be 'precision' or 'metric'")

                    self.M_normed = normalize_metric_atoms(self.M_raw, mode=self.weight_metric_normalization)
                    if self.weight_kernel == 'mahalanobis_normed':
                        self.weight_atoms = self.M_normed
                        self.mix_atoms = self.metric_scale * self.M_normed
                    else:
                        self.weight_atoms = self.M_raw
                        self.mix_atoms = self.metric_scale * self.M_raw
                        
                    # Allow samplers expecting Stage-B buffers and background params
                    self.M_tens = self.mix_atoms
                    self.bg_strength = 1e-2
                    self.bg_radius = None
                    self.use_background_identity = False

                def parameters(self):
                    return iter([torch.empty(0, device=self.device)])

                def _compute_weights(self, z: torch.Tensor) -> torch.Tensor:
                    dtype = self.centroids_tens.dtype if not z.dtype.is_floating_point else z.dtype
                    z_local = z.to(self.device, dtype=dtype)
                    centroids = self.centroids_tens.to(dtype=dtype)
                    atoms = self.weight_atoms.to(dtype=dtype)
                    temp = self.temperature.to(dtype=dtype)
                    return compute_metric_weights(
                        z_local,
                        centroids,
                        atoms,
                        temp,
                        kernel=self.weight_kernel,
                        normalize=self.normalize_weight_sum,
                        topk=self.topk_weights,
                    )

                def G_inv(self, z):
                    dtype = self.centroids_tens.dtype if not z.dtype.is_floating_point else z.dtype
                    z_local = z.to(self.device, dtype=dtype)
                    weights = self._compute_weights(z_local)
                    if not self.normalize_weight_sum:
                        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)
                    atoms = self.mix_atoms.to(dtype=weights.dtype)
                    G_inv_mix = torch.einsum('bk,kij->bij', weights, atoms)
                    G_inv_mix = 0.5 * (G_inv_mix + G_inv_mix.transpose(-1, -2))

                    I = torch.eye(G_inv_mix.shape[-1], device=self.device, dtype=weights.dtype).unsqueeze(0)
                    lam = self.regularization.to(dtype=weights.dtype).clamp_min(0.0)
                    if lam.item() > 0:
                        G_inv_mix = G_inv_mix + lam * I

                    centroids = self.centroids_tens.to(dtype=dtype)
                    diff = z_local.unsqueeze(1) - centroids.unsqueeze(0)
                    if self.weight_kernel == 'isotropic':
                        d2 = torch.sum(diff * diff, dim=-1)
                    else:
                        tmp = torch.einsum('bkd,kde->bke', diff, self.weight_atoms.to(dtype=dtype))
                        d2 = torch.sum(tmp * diff, dim=-1)
                    d2min = d2.min(dim=1).values
                    temp_scalar = float(self.temperature)
                    thresh = (4.0 * temp_scalar) ** 2
                    far = torch.sigmoid((d2min - thresh) / (0.25 * thresh + 1e-12))
                    if self.use_background_identity:
                        G_inv_mix = G_inv_mix + self.bg_strength * far.view(-1, 1, 1) * I

                    if self.regularization_mode == 'precision':
                        return G_inv_mix
                    G = torch.linalg.inv(G_inv_mix.float()).to(G_inv_mix.dtype)
                    if lam.item() > 0:
                        G = G + lam * I
                    return torch.linalg.inv(G.float()).to(G.dtype)

                def G(self, z):
                    G_inv = self.G_inv(z)
                    G = torch.linalg.inv(G_inv.float()).to(G_inv.dtype)
                    return G
            # Match Stage A behavior if available (fallback False)
            model_cfg = getattr(cfg, 'model', None)
            normalize_w = bool(getattr(model_cfg, 'normalize_weight_sum', True)) if model_cfg else True
            weight_kernel = str(getattr(model_cfg, 'weight_kernel', 'isotropic') or 'isotropic') if model_cfg else 'isotropic'
            weight_metric_norm = str(getattr(model_cfg, 'weight_metric_normalization', 'trace')) if model_cfg else 'trace'
            topk_weights = getattr(model_cfg, 'topk_weights', 8) if model_cfg else 8
            metric_scale = float(getattr(model_cfg, 'metric_scale', 1.0)) if model_cfg else 1.0
            metric_model = _MetricStub(
                C_use,
                M_use,
                blob['temperature'],
                blob['regularization'],
                self.device,
                normalize_weight_sum=normalize_w,
                weight_kernel=weight_kernel,
                weight_metric_normalization=weight_metric_norm,
                topk_weights=topk_weights,
                metric_scale=metric_scale,
                regularization_mode=str(getattr(model_cfg, 'regularization_mode', 'precision')).lower() if model_cfg else 'precision',
            )
            if model_cfg is not None:
                if hasattr(model_cfg, 'bg_strength') and model_cfg.bg_strength is not None:
                    metric_model.bg_strength = float(model_cfg.bg_strength)
                if hasattr(model_cfg, 'bg_radius') and model_cfg.bg_radius is not None:
                    metric_model.bg_radius = float(model_cfg.bg_radius)
                if hasattr(model_cfg, 'use_background_identity') and model_cfg.use_background_identity is not None:
                    metric_model.use_background_identity = bool(model_cfg.use_background_identity)
            # Tune RHMC for better acceptance on Stage B diagnostic overlay
            step_size = float(getattr(cfg.sampling, 'step_size', 0.001)) if hasattr(cfg, 'sampling') else 0.001
            tuned_eps = min(0.005, max(1e-5, step_size))
            rhmc = RHVAEVolumeElementHMCSampler(
                metric_model,
                mcmc_steps_nbr=max(100, cfg.sampling.n_steps),
                n_lf=max(20, cfg.sampling.n_leapfrog),
                eps_lf=tuned_eps,
                beta_zero=1.0
            )
            samples = rhmc.sample(n_samples=1024)
            metric_dir = getattr(cfg.checkpoint, "metric_dir", None)
            if not metric_dir:
                metric_dir = getattr(cfg.checkpoint, "stageB_dir", None)
            if not metric_dir:
                metric_dir = getattr(cfg.checkpoint, "dir", None)
            if not metric_dir:
                metric_dir = getattr(cfg.settings.pipeline.outputs, "stage_b_dir", None)
            if metric_dir is None:
                raise RuntimeError("Unable to determine target directory for RHMC samples (checkpoint.metric_dir/stageB_dir/dir missing)")
            metric_dir_path = Path(metric_dir)
            metric_dir_path.mkdir(parents=True, exist_ok=True)
            torch.save({'samples': samples.detach().cpu()}, metric_dir_path / cfg.checkpoint.rhmc_samples)
            print(f"[Sampling] Saved RHMC samples to {metric_dir_path / cfg.checkpoint.rhmc_samples}")
            # (Disabled raw z overlay by request; PCA overlays logged below)

            # PCA(2) visualization on manifold t=0 with RHMC overlay and reconstructions
            try:
                import numpy as np
                import matplotlib.pyplot as plt
                from sklearn.decomposition import PCA
                import torchvision.utils as vutils

                # Load latent references from metric file if available
                state = torch.load(metric_file, map_location='cpu', weights_only=False)
                Z_ref = state.get('t0_latents', None)
                pca_fit_basis = 'mu0' if Z_ref is not None else None
                if Z_ref is None:
                    Z_ref = state.get('z_sample', None)
                    pca_fit_basis = 'z_sample' if Z_ref is not None else None
                if Z_ref is None:
                    # Fallback to centroids for PCA fit
                    Z_ref = C_use.detach().cpu()
                    pca_fit_basis = 'centroids'
                else:
                    Z_ref = Z_ref.detach().cpu()
                Z_ref_np = Z_ref.numpy()
                # Fit PCA(2) on t=0 latent references
                pca = PCA(n_components=2, random_state=42)
                Z_ref_pca = pca.fit_transform(Z_ref_np)
                U = pca.components_  # [2, D]
                mean_np = pca.mean_   # [D]
                U_d2 = torch.tensor(U.T, device=self.device, dtype=blob['centroids'].dtype)
                mean_t = torch.tensor(mean_np, device=self.device, dtype=blob['centroids'].dtype)

                # Build grid in PCA coordinates
                xmin, xmax = np.percentile(Z_ref_pca[:,0], [1, 99])
                ymin, ymax = np.percentile(Z_ref_pca[:,1], [1, 99])
                pad_x = 0.1 * (xmax - xmin + 1e-6); pad_y = 0.1 * (ymax - ymin + 1e-6)
                xmin -= pad_x; xmax += pad_x; ymin -= pad_y; ymax += pad_y
                gx, gy = np.meshgrid(np.linspace(xmin, xmax, 180), np.linspace(ymin, ymax, 180))
                grid2 = np.stack([gx.ravel(), gy.ravel()], axis=1)
                grid2_t = torch.tensor(grid2, device=self.device, dtype=blob['centroids'].dtype)
                # Compute det(G^{-1}) strictly in the PCA(2) subspace, matching Stage A logic
                # Project centroids and per-centroid matrices onto PCA(2)
                C2 = torch.matmul((C_use - mean_t), U_d2)                               # [K, 2]
                MU = torch.einsum('kde,ej->kdj', M_use, U_d2)                           # [K, D, 2]
                M2 = torch.einsum('id,kdj->kij', U_d2.T, MU)                            # [K, 2, 2]

                # Select S2 for Mahalanobis weights in 2D (trace/det normalization) — used only for distance, not aggregation
                if weight_kernel == 'mahalanobis_normed':
                    if weight_metric_norm == 'trace':
                        traces2 = torch.einsum('kii->k', M2).clamp_min(1e-12)
                        S2 = M2 / (traces2.view(-1, 1, 1) / 2.0)
                    elif weight_metric_norm == 'det':
                        dets2 = torch.linalg.det(M2).abs().clamp_min(1e-24)
                        scales2 = dets2.pow(1.0 / 2.0).clamp_min(1e-12)
                        S2 = M2 / scales2.view(-1, 1, 1)
                    else:
                        S2 = M2
                elif weight_kernel == 'mahalanobis':
                    S2 = M2
                else:  # isotropic
                    S2 = None

                # Evaluate det(G^{-1}_subspace) on the PCA grid using RAW (non-normalized) weights
                z2_all = grid2_t  # [B, 2] coordinates in PCA plane
                det_map_sub = torch.empty(z2_all.shape[0], device=self.device, dtype=blob['centroids'].dtype)
                chunk = 2048
                eye2 = torch.eye(2, device=self.device, dtype=blob['centroids'].dtype).unsqueeze(0)
                T2 = (float(blob['temperature']) ** 2) + 1e-12
                atoms_for_weights = S2 if S2 is not None else M2
                temperature_tensor = torch.as_tensor(float(blob['temperature']), device=self.device, dtype=blob['centroids'].dtype)
                lambda_tensor = torch.as_tensor(float(blob['regularization']), device=self.device, dtype=blob['centroids'].dtype)
                for s in range(0, z2_all.shape[0], chunk):
                    z2 = z2_all[s:s+chunk]                                                    # [b, 2]
                    with torch.no_grad():
                        weights = compute_metric_weights(
                            z2,
                            C2,
                            atoms_for_weights,
                            temperature_tensor,
                            kernel=weight_kernel,
                            normalize=normalize_w,
                            topk=topk_weights,
                            stabilize=False
                        )
                        atoms_mix = M2.to(dtype=weights.dtype, device=self.device)
                        Ginv2 = torch.einsum('bk,kij->bij', weights, atoms_mix) + lambda_tensor * eye2  # [b, 2, 2]
                        det2 = torch.linalg.det(Ginv2).clamp_min(1e-20)
                        det_map_sub[s:s+z2.shape[0]] = det2
                det_img_sub = det_map_sub.detach().cpu().numpy().reshape(gx.shape)

                # Project centroids and samples to PCA(2)
                C_np = C_use.detach().cpu().numpy()
                C_pca = (C_np - mean_np) @ U.T
                S_np = samples.detach().cpu().numpy()
                S_pca = (S_np - mean_np) @ U.T

                # Plot heatmap + overlays (subspace determinant to match Stage A visuals)
                fig, ax = plt.subplots(1,1,figsize=(7,6))
                # Improve contrast using percentile clipping
                lo, hi = np.nanpercentile(det_img_sub, [5, 95])
                im = ax.imshow(det_img_sub, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto', vmin=lo, vmax=hi)
                ax.scatter(Z_ref_pca[:,0][::max(1, len(Z_ref_pca)//3000)], Z_ref_pca[:,1][::max(1, len(Z_ref_pca)//3000)], s=4, c='white', alpha=0.15, label='t=0 latents')
                ax.scatter(C_pca[:,0], C_pca[:,1], c='red', marker='*', s=80, label='centroids')
                ax.scatter(S_pca[:,0], S_pca[:,1], s=6, c='xkcd:blue', alpha=0.35, label='RHMC samples')
                ax.set_title('Stage B: det(G^{-1}) in PCA(2) subspace (t=0) + RHMC')
                ax.legend(frameon=False, loc='upper right')
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='det(G^{-1}) (PCA subspace)')
                # Single run already open
                if wandb.run is not None:
                    wandb.log({"stageB/pca/rhmc_overlay_t0": wandb.Image(fig)})
                plt.close(fig)

                # Also log standalone heatmap and centroids-only scatter
                fig_h, ax_h = plt.subplots(1,1,figsize=(6,5))
                ax_h.set_title('Stage B: det(G^{-1}) heatmap (PCA subspace, t=0)')
                imh = ax_h.imshow(det_img_sub, origin='lower', extent=[xmin, xmax, ymin, ymax], cmap='viridis', aspect='auto', vmin=lo, vmax=hi)
                plt.colorbar(imh, ax=ax_h, fraction=0.046, pad=0.04)
                if wandb.run is not None:
                    wandb.log({"stageB/pca/det_heatmap_t0": wandb.Image(fig_h)})
                plt.close(fig_h)

                fig_c, ax_c = plt.subplots(1,1,figsize=(6,5))
                ax_c.scatter(C_pca[:,0], C_pca[:,1], c='red', marker='*', s=80, label='centroids')
                ax_c.set_title('Stage B: Centroids on PCA(2) (t=0)')
                ax_c.legend(frameon=False)
                if wandb.run is not None:
                    wandb.log({"stageB/pca/centroids_t0": wandb.Image(fig_c)})
                plt.close(fig_c)

                # Concise Stage B summary as a table (instead of extra plots)
                if wandb.run is not None:
                    try:
                        acceptance = getattr(rhmc, 'last_acceptance_rate', None)
                        summary_cols = [
                            'timestep', 'centroids_source', 'pca_fit_basis', 'n_centroids_total', 'n_centroids_t0',
                            'latent_dim', 'temperature', 'regularization', 'weight_kernel', 'weight_metric_normalization',
                            'normalize_weight_sum', 'topk_weights', 'metric_scale', 'n_steps', 'n_leapfrog', 'step_size', 'acceptance_rate'
                        ]
                        summary_row = [
                            0,
                            't0-filtered' if state.get('t0_latents', None) is not None else 'all',
                            pca_fit_basis,
                            int(C_all.shape[0]),
                            int(C_use.shape[0]),
                            int(C_all.shape[1]) if C_all.ndim == 2 else int(M_all.shape[-1]),
                            float(blob['temperature']),
                            float(blob['regularization']),
                            weight_kernel,
                            weight_metric_norm,
                            bool(normalize_w),
                            int(topk_weights) if topk_weights is not None else None,
                            float(metric_scale),
                            int(getattr(cfg.sampling, 'n_steps', 100)),
                            int(getattr(cfg.sampling, 'n_leapfrog', 50)),
                            float(getattr(cfg.sampling, 'step_size', 0.001)),
                            float(acceptance) if acceptance is not None else None,
                        ]
                        table = wandb.Table(data=[summary_row], columns=summary_cols)
                        wandb.log({"stageB/summary_table": table})
                    except Exception as e:
                        print(f"[Stage B] ⚠️ Summary table logging failed: {e}")

                # Stage B: Reconstruction vs Original using available encoder/decoder (RHVAE or Vanilla)
                try:
                    import torchvision.utils as vutils
                    model_enc = None
                    model_dec = None
                    if 'rh_exp' in locals() and hasattr(rh_exp, 'model'):
                        model_enc = getattr(rh_exp.model, 'encoder', None)
                        model_dec = getattr(rh_exp.model, 'decoder', None)
                    elif 'vanilla' in locals():
                        model_enc = getattr(vanilla, 'encoder', None)
                        model_dec = getattr(vanilla, 'decoder', None)
                    elif 'stageB_model' in locals():
                        model_enc = getattr(stageB_model, 'encoder', None)
                        model_dec = getattr(stageB_model, 'decoder', None)
                    if model_enc is not None and model_dec is not None:
                        model_enc = model_enc.to(self.device).eval()
                        model_dec = model_dec.to(self.device).eval()
                        if data_train_path is not None and Path(data_train_path).exists():
                            raw_train = torch.load(data_train_path, map_location='cpu')
                        elif stage_b_train_tensor is not None:
                            raw_train = stage_b_train_tensor.cpu()
                        else:
                            raw_train = None
                        imgs0 = None
                        if raw_train is not None:
                            if raw_train.ndim == 5:
                                imgs0 = raw_train[:8, 0].to(self.device)
                            elif raw_train.ndim == 4:
                                imgs0 = raw_train[:8].to(self.device)
                        if imgs0 is not None:
                            with torch.no_grad():
                                enc_out = model_enc(imgs0)
                                z = enc_out.get('embedding', None) if isinstance(enc_out, dict) else enc_out
                                if z is None and isinstance(enc_out, dict):
                                    z = enc_out.get('z', None)
                                dec_out = model_dec(z)
                                recon = dec_out.get('reconstruction', None) if isinstance(dec_out, dict) else dec_out
                                if recon is None and isinstance(dec_out, dict):
                                    recon = dec_out.get('recon_x', None)
                                recon = torch.sigmoid(recon) if recon.dtype.is_floating_point else recon
                                recon = recon.clamp(0, 1)
                            grid = vutils.make_grid(
                                torch.cat([imgs0, recon], dim=0),
                                nrow=8,
                                normalize=True,
                                value_range=(0.0, 1.0)
                            )
                            if wandb.run is not None:
                                wandb.log({"stageB/recon_vs_real": wandb.Image(grid)})
                    else:
                        print("[Stage B] Note: No encoder/decoder available for recon_vs_real.")
                except Exception as e:
                    print(f"[Stage B] ⚠️ stageB/recon_vs_real logging failed: {e}")
            except Exception as e:
                print(f"[Stage B] ⚠️ PCA RHMC overlay failed: {e}")

            self.config = stage_b_base
            cfg = self.config

        # Stage C: RLVAE
        if getattr(cfg.experiment, 'run_stage_c', True):
            stage_c_base = self.config
            self.config = OmegaConf.create(OmegaConf.to_container(stage_c_base, resolve=False))
            cfg = self.config
            print("\n=== [Stage C] RLVAE training ===")
            stagec_debugger.log_event(
                "stagec_start",
                {
                    "arch": getattr(cfg.model.encoder, 'architecture', None) if hasattr(cfg.model, 'encoder') and cfg.model.encoder is not None else 'mlp',
                    "latent_dim": getattr(cfg.model, 'latent_dim', None),
                    "metric_impl": getattr(cfg.experiment.stage_b, 'implementation', None),
                },
            )
            
            # Define architecture and latent_dim for Stage C
            arch = cfg.model.encoder.architecture if hasattr(cfg.model, 'encoder') and cfg.model.encoder is not None else 'mlp'
            latent_dim = cfg.model.latent_dim
            metric_impl = cfg.experiment.stage_b.implementation
            
            # Get organized Stage C paths
            stageC_paths = get_stage_paths(cfg, 'C', 'RLVAE', arch, latent_dim)
            
            # Use centralized Stage A and B data (already looked up)
            # stage_a_data and stage_b_data are already available from centralized lookup
        
        # Use enhanced component loader to reduce init phases (SINGLE LOADING)
        if hasattr(cfg.experiment, 'initialization') and cfg.experiment.initialization.get('track_phases', False):
            print("[ENHANCED] Using enhanced component loader to reduce init phases")
            component_loader = EnhancedComponentLoader(cfg)
            all_components = component_loader.load_all_components(stage_a_data, stage_b_data)
            
            # Update config with loaded components (NO DUPLICATE LOADING)
            if 'encoder' in all_components and stage_a_data:
                self._set_config_value(cfg, "model.pretrained.encoder_path", str(stage_a_data['encoder_path']))
            if 'decoder' in all_components and stage_a_data:
                self._set_config_value(cfg, "model.pretrained.decoder_path", str(stage_a_data['decoder_path']))
            if 'metric' in all_components and stage_b_data:
                self._set_config_value(cfg, "model.pretrained.metric_path", str(stage_b_data['metric_path']))
            
            print(f"[ENHANCED] Loaded {len(all_components)} components in single phase")
            
            # Continue with Stage C training after enhanced loading
            print(f"\n=== [Stage C] CONTINUING WITH RLVAE TRAINING ===")
            print(f"[Stage C] Enhanced component loading completed, proceeding with model training...")
            
            # Create and train the RLVAE model with loaded components
            try:
                # Create data module
                data_module = build_data_module(cfg.data)
                
                # Create model wrapper with pretrained components
                model_wrapper = LightningRlVAETrainer(
                    cfg,
                    data_module=data_module
                )
                
                # Create trainer
                wandb_logger = self._setup_wandb("stage_c_enhanced")
                trainer = self._create_trainer(wandb_logger)
                
                # Train the model
                print(f"[Stage C] 🚀 Starting RLVAE training with enhanced components...")
                trainer.fit(model_wrapper, data_module)

                print(f"[Stage C] ✅ RLVAE training completed successfully!")
                stagec_debugger.log_event("stagec_training_complete", {"status": "success"})
                # Reuse the trained wrapper later if needed; also avoid
                # re-instantiation that causes metric to be loaded twice.
                self._stage_c_model_wrapper = model_wrapper

            except Exception as e:
                print(f"[Stage C] ❌ RLVAE training failed: {e}")
                stagec_debugger.log_fallback("stagec_training_complete", reason="exception", payload={"error": str(e)})
                raise
            # Enhanced path completes here; avoid legacy path below that
            # would recreate the model and reload the metric a second time.
            stagec_debugger.flush()
            return
        else:
            # Fallback: Load components individually (legacy method)
            print("[LEGACY] Using individual component loading")
            
            # Define metric_file for fallback cases
            metric_file = stage_b_data['metric_path'] if stage_b_data else None
            
            print(f"\n=== [Stage C] COMPONENT LOADING DEBUG ===")
            print(f"[Stage C] Architecture: {arch}, Latent dim: {latent_dim}")
            print(f"[Stage C] Stage A data found: {stage_a_data is not None}")
            print(f"[Stage C] Stage B data found: {stage_b_data is not None}")
            
            if stage_a_data:
                print(f"[Stage C] Stage A paths:")
                print(f"  - Base dir: {stage_a_data['base_dir']}")
                print(f"  - Encoder: {stage_a_data['encoder_path']}")
                print(f"  - Decoder: {stage_a_data['decoder_path']}")
                print(f"  - Config: {stage_a_data['config_path']}")
                if stage_a_data.get('metric_path'):
                    print(f"  - Metric: {stage_a_data['metric_path']}")
            
            if stage_b_data:
                print(f"[Stage C] Stage B paths:")
                print(f"  - Base dir: {stage_b_data['base_dir']}")
                print(f"  - Metric: {stage_b_data['metric_path']}")
                print(f"  - Config: {stage_b_data['config_path']}")
            
            # Wire metric path and pretrained components into model config (LEGACY METHOD ONLY)
            if not (hasattr(cfg.experiment, 'initialization') and cfg.experiment.initialization.get('track_phases', False)):
                if stage_b_data is not None:
                    # Always set the pretrained metric path (works across model variants)
                    try:
                        self._set_config_value_if_missing(self.config, "model.pretrained.metric_path", str(stage_b_data['metric_path']))
                    except Exception as e:
                        print(f"[Stage C] ⚠️ Could not set pretrained.metric_path: {e}")
                    # Try to set fixed metric path on metric block when schema supports it
                    try:
                        if hasattr(self.config.model, 'metric') and self.config.model.metric is not None:
                            # Only set if key exists to avoid struct errors
                            try:
                                if 'fixed_metric_path' in self.config.model.metric:
                                    self._set_config_value_if_missing(self.config, "model.metric.fixed_metric_path", str(stage_b_data['metric_path']))
                            except Exception:
                                pass
                            try:
                                if 'temperature_override' in self.config.model.metric:
                                    self._set_config_value_if_missing(self.config, "model.metric.temperature_override", None)
                            except Exception:
                                pass
                            try:
                                # For Phase‑2 adaptability: initialize trainable net from fixed metric
                                if 'init_from_fixed' in self.config.model.metric:
                                    self._set_config_value_if_missing(self.config, "model.metric.init_from_fixed", True)
                                if 'trainable' in self.config.model.metric:
                                    self._set_config_value_if_missing(self.config, "model.metric.trainable", True)
                                if 'architecture' in self.config.model.metric and not self.config.model.metric.architecture:
                                    self._set_config_value_if_missing(self.config, "model.metric.architecture", 'mlp')
                            except Exception:
                                pass
                    except Exception as e:
                        # Non-fatal; different model schemas may not expose these keys
                        print(f"[Stage C] ℹ️ Skipping metric.fixed_metric_path/init_from_fixed wiring: {e}")
                    print(f"[Stage C] ✅ Using Stage B metric: {stage_b_data['metric_path']}")
                else:
                    # Fallback to old method
                    if metric_file is not None:
                        self._set_config_value_if_missing(self.config, "model.pretrained.metric_path", str(metric_file))
                        self._set_config_value_if_missing(self.config, "model.metric.fixed_metric_path", str(metric_file))
                        # Allow metric to update during training for better temporal dynamics
                        self._set_config_value_if_missing(self.config, "model.metric.init_from_fixed", False)
                        print(f"[Stage C] ⚠️ Using fallback metric: {metric_file}")
                    else:
                        print(f"[Stage C] ❌ No metric file found!")
                        raise ValueError("No metric file available for Stage C")
                
                if stage_a_data is not None:
                    self._set_config_value_if_missing(self.config, "model.pretrained.encoder_path", str(stage_a_data['encoder_path']))
                    self._set_config_value_if_missing(self.config, "model.pretrained.decoder_path", str(stage_a_data['decoder_path']))
                    print(f"[Stage C] ✅ Using Stage A encoder/decoder:")
                    print(f"  - Encoder: {stage_a_data['encoder_path']}")
                    print(f"  - Decoder: {stage_a_data['decoder_path']}")
                elif 'encoder' in comp_paths:
                    self._set_config_value_if_missing(self.config, "model.pretrained.encoder_path", comp_paths['encoder'])
                    self._set_config_value_if_missing(self.config, "model.pretrained.decoder_path", comp_paths['decoder'])
                    print(f"[Stage C] ⚠️ Using fallback encoder/decoder: {comp_paths['encoder']}")
                else:
                    print(f"[Stage C] ❌ No encoder/decoder found!")
            
            # Final path summary (tolerate missing keys)
            enc_path = getattr(getattr(self.config.model, 'pretrained', {}), 'encoder_path', None)
            dec_path = getattr(getattr(self.config.model, 'pretrained', {}), 'decoder_path', None)
            met_path = getattr(getattr(self.config.model, 'pretrained', {}), 'metric_path', None)
            print(f"[Stage C] Final config paths:")
            print(f"  - Encoder: {enc_path}")
            print(f"  - Decoder: {dec_path}")
            print(f"  - Metric: {met_path}")
            print(f"[Stage C] Final metric config:")
            try:
                metric_cfg = getattr(self.config.model, 'metric', None)
                if metric_cfg is not None:
                    fixed_path_val = None
                    init_from_fixed_val = None
                    try:
                        if 'fixed_metric_path' in metric_cfg:
                            fixed_path_val = metric_cfg.fixed_metric_path
                    except Exception:
                        pass
                    try:
                        if 'init_from_fixed' in metric_cfg:
                            init_from_fixed_val = metric_cfg.init_from_fixed
                    except Exception:
                        pass
                    print(f"  - fixed_metric_path: {fixed_path_val if fixed_path_val is not None else 'n/a'}")
                    print(f"  - init_from_fixed: {init_from_fixed_val if init_from_fixed_val is not None else 'n/a'}")
                else:
                    print("  - metric config: n/a")
            except Exception as e:
                print(f"  - metric config unavailable ({e})")
            
            # Verify architecture and latent_dim consistency with Stage A
            if stage_a_data:
                print(f"[Stage C] Architecture/Latent Dim Verification:")
                print(f"  - Stage A config: {stage_a_data['config_path']}")
                try:
                    with open(stage_a_data['config_path'], 'r') as f:
                        stage_a_config = yaml.safe_load(f)
                    stage_a_arch = stage_a_config.get('architecture', 'unknown')
                    stage_a_ld = stage_a_config.get('latent_dim', 'unknown')
                    print(f"  - Stage A: arch={stage_a_arch}, latent_dim={stage_a_ld}")
                    print(f"  - Stage C: arch={arch}, latent_dim={latent_dim}")
                    if stage_a_arch != arch or stage_a_ld != latent_dim:
                        print(f"  - ⚠️ MISMATCH DETECTED! Stage A and Stage C have different architecture/latent_dim!")
                        print(f"  - Forcing Stage C to use Stage A parameters...")
                        # Force Stage C to use Stage A's architecture and latent_dim
                        self._set_config_value_if_missing(self.config, "model.encoder.architecture", stage_a_arch)
                        self._set_config_value_if_missing(self.config, "model.decoder.architecture", stage_a_arch)
                        self._set_config_value_if_missing(self.config, "model.latent_dim", stage_a_ld)
                        print(f"  - ✅ Updated Stage C config to match Stage A")
                    else:
                        print(f"  - ✅ Architecture and latent_dim match between Stage A and Stage C")
                except Exception as e:
                    print(f"  - ⚠️ Could not verify Stage A config: {e}")
            
            print(f"=== [Stage C] END COMPONENT DEBUG ===\n")

            # Keep monolith Stage‑C config in sync with latest artifacts
            try:
                self._update_monolith_stagec_config(
                    encoder_path=self.config.model.pretrained.encoder_path,
                    decoder_path=self.config.model.pretrained.decoder_path,
                    metric_path=self.config.model.pretrained.metric_path,
                    latent_dim=int(self.config.model.latent_dim),
                    input_dim=list(self.config.model.input_dim),
                )
            except Exception as e:
                print(f"[Stage C] ⚠️ Monolith config sync failed: {e}")
            # Ensure latent_dim matches the metric from Stage B
            try:
                metric_file_to_check = stage_b_data['metric_path'] if stage_b_data else (metric_file if metric_file is not None else None)
                state_metric = torch.load(metric_file_to_check, map_location='cpu', weights_only=False)
                ld = state_metric.get('latent_dim', None)
                if ld is None:
                    C_ = state_metric.get('centroids', None)
                    M_ = state_metric.get('metric_matrices', None)
                    if C_ is not None and C_.ndim == 2:
                        ld = int(C_.shape[1])
                    elif M_ is not None and M_.ndim == 3:
                        ld = int(M_.shape[-1])
                if ld is not None:
                    self._set_config_value_if_missing(self.config, "model.latent_dim", int(ld))
                    print(f"[Stage C] Set model.latent_dim = {self.config.model.latent_dim} from Stage A metric")
                # Fallback: read encoder/decoder paths from metric payload if not already set
                try:
                    if not getattr(self.config.model.pretrained, 'encoder_path', None):
                        enc_p = state_metric.get('encoder_path', None)
                        if enc_p:
                            self._set_config_value_if_missing(self.config, "model.pretrained.encoder_path", enc_p)
                    if not getattr(self.config.model.pretrained, 'decoder_path', None):
                        dec_p = state_metric.get('decoder_path', None)
                        if dec_p:
                            self._set_config_value_if_missing(self.config, "model.pretrained.decoder_path", dec_p)
                except Exception:
                    pass
            except Exception as e:
                print(f"[Stage C] ⚠️ Could not set latent_dim from metric: {e}")
            # Epoch override
            self._set_config_value_if_missing(self.config, "training.trainer.max_epochs", cfg.experiment.stage_c.epochs)

            # Apply Stage C model overrides (posterior/KL parameters etc.)
            try:
                sc = cfg.experiment.stage_c
                # Propagate riemannian_beta if provided at stage level
                if hasattr(sc, 'riemannian_beta') and sc.riemannian_beta is not None:
                    self._set_config_value_if_missing(self.config, "model.riemannian_beta", float(sc.riemannian_beta))
                # Propagate sampling method if present
                if hasattr(sc, 'sampling') and sc.sampling is not None and hasattr(sc.sampling, 'method'):
                    self._set_config_value_if_missing(self.config, "model.sampling.method", sc.sampling.method)
                # Generic overrides map (applied to model fields)
                if hasattr(sc, 'overrides') and sc.overrides is not None:
                    for k, v in sc.overrides.items():
                        try:
                            self._set_config_value_if_missing(self.config, f"model.{k}", v)
                        except Exception:
                            # Non‑fatal for structured configs
                            pass
                # Force n_flows if provided at stage C level
                try:
                    if hasattr(sc, 'n_flows') and sc.n_flows is not None:
                        self._set_config_value_if_missing(self.config, "model.n_flows", int(sc.n_flows))
                except Exception:
                    pass
                # Merge any training overrides declared inside experiment.stage_c (e.g., metric alternation)
                try:
                    if hasattr(sc, 'training') and sc.training is not None:
                        # Shallow merge metric_alternation
                        if hasattr(sc.training, 'metric_alternation'):
                            for k, v in sc.training.metric_alternation.items():
                                self._set_config_value_if_missing(self.config, f"training.metric_alternation.{k}", v)
                except Exception as e:
                    print(f"[Stage C] ⚠️ Failed to merge training overrides: {e}")
            except Exception as e:
                print(f"[Stage C] ⚠️ Failed to apply Stage C overrides: {e}")

            # Use fully modular Stage C (ModRLVAE) - respect existing target if set
            try:
                if not hasattr(self.config.model, '_target_') or not self.config.model._target_:
                    self._set_config_value_if_missing(self.config, "model._target_", 'rlvae.models.modular_rlvae.ModularRiemannianFlowVAE')
                else:
                    print(f"[Stage C] Using existing model target: {self.config.model._target_}")
            except Exception:
                pass

            # Resolve input_dim interpolations before model creation
            try:
                if (hasattr(self.config, 'data') and 
                    hasattr(self.config.data, 'channels') and 
                    hasattr(self.config.data, 'image_size')):
                    self._set_config_value_if_missing(self.config, "model.input_dim", [
                        self.config.data.channels,
                        self.config.data.image_size[0],
                        self.config.data.image_size[1]
                    ])
                    print(f"[Stage C] ✅ Resolved input_dim = {self.config.model.input_dim}")
            except Exception as e:
                print(f"[Stage C] ⚠️ Failed to resolve input_dim: {e}")

            # Set specific model parameters for Stage C (guarded for struct configs)
            try:
                # Preserve posterior type from config - DON'T override it!
                # If posterior.type is not set, default to 'riemannian_metric'
                if hasattr(self.config.model, 'posterior') and self.config.model.posterior is not None:
                    if not getattr(self.config.model.posterior, 'type', None):
                        self._set_config_value_if_missing(self.config, "model.posterior.type", 'riemannian_metric')
                else:
                    self._set_config_value_if_missing(self.config, "model.posterior.type", 'riemannian_metric')
                self._set_config_value_if_missing(self.config, "model.posterior_type", getattr(self.config.model.posterior, 'type', 'riemannian_metric'))
                # Sequence length and flows for modular temporal dynamics
                try:
                    seq_len = int(getattr(self.config.data, 'sequence_length', 0))
                    if seq_len > 0:
                        # Prefer sequence_length; ModRLVAE auto-derives n_flows = seq_len - 1 when n_flows missing
                        self._set_config_value_if_missing(self.config, "model.sequence_length", seq_len)
                        self._set_config_value_if_missing(self.config, "model.n_flows", max(0, seq_len - 1))
                except Exception:
                    pass
                try:
                    if 'riemannian_kl_mode' in self.config.model and (self.config.model.riemannian_kl_mode is None or self.config.model.riemannian_kl_mode == ''):
                        self._set_config_value_if_missing(self.config, "model.riemannian_kl_mode", 'sample_logq_logp')
                except Exception:
                    pass
                try:
                    if 'riemannian_beta' in self.config.model and (self.config.model.riemannian_beta is None):
                        self._set_config_value_if_missing(self.config, "model.riemannian_beta", 1.0)
                except Exception:
                    pass
                # Ensure temporal reconstruction is enabled for RLVAE
                try:
                    if 'reconstruction_mode' in self.config.model:
                        self._set_config_value_if_missing(self.config, "model.reconstruction_mode", 'all')
                except Exception:
                    pass
                # Ensure temporal dynamics are properly configured
                try:
                    if 'encode_all_timesteps_if_no_flows' in self.config.model:
                        self._set_config_value_if_missing(self.config, "model.encode_all_timesteps_if_no_flows", True)
                except Exception:
                    pass
                try:
                    if 'kl_over_all_timesteps_if_flows' in self.config.model:
                        self._set_config_value_if_missing(self.config, "model.kl_over_all_timesteps_if_flows", False)
                except Exception:
                    pass
            except Exception:
                pass
            
            # ============================================================================
            # CRITICAL: Force-sync posterior type from experiment config to all locations
            # This prevents training.model.* defaults from overriding experiment.model.*
            # ============================================================================
            try:
                # Get the intended posterior type from experiment config (highest priority)
                intended_posterior_type = None
                
                # Priority 1: already-resolved model.posterior.type (includes mega config overrides)
                if hasattr(self.config.model, 'posterior'):
                    intended_posterior_type = getattr(self.config.model.posterior, 'type', None)
                
                # Priority 2: explicit experiment.model overrides (e.g., stage configs)
                if intended_posterior_type is None and hasattr(cfg.experiment, 'model') and hasattr(cfg.experiment.model, 'posterior'):
                    intended_posterior_type = getattr(cfg.experiment.model.posterior, 'type', None)
                
                # Priority 3: experiment.model.posterior_type (alternative location)
                if intended_posterior_type is None and hasattr(cfg.experiment, 'model'):
                    intended_posterior_type = getattr(cfg.experiment.model, 'posterior_type', None)
                
                # If we found an intended type, sync it EVERYWHERE to prevent overrides
                if intended_posterior_type is not None and intended_posterior_type != '':
                    print(f"[Stage C] 🔒 Forcing posterior type sync: '{intended_posterior_type}'")
                    for path in [
                        "model.posterior.type",
                        "model.posterior_type",
                        "training.model.posterior.type",
                        "training.model.posterior_type",
                    ]:
                        self._set_config_value(self.config, path, intended_posterior_type)
                    print(f"[Stage C] ✅ Posterior type synced to all config locations")
                else:
                    print(f"[Stage C] ⚠️ No posterior type found in experiment config, using defaults")
                    
            except Exception as e:
                print(f"[Stage C] ⚠️ Error during posterior type sync: {e}")

            # ============================================================================
            # CRITICAL: Force-sync RHMC params and KL toggles across config locations
            # ============================================================================
            try:
                posterior_cfg = getattr(self.config.model, 'posterior', None)
                def _posterior_get(attr, fallback=None):
                    if posterior_cfg is None:
                        return fallback
                    try:
                        # OmegaConf DictConfig exposes .get(); fall back to getattr for plain objects
                        if hasattr(posterior_cfg, 'get'):
                            val = posterior_cfg.get(attr, fallback)
                        else:
                            val = getattr(posterior_cfg, attr)
                        return val
                    except Exception:
                        pass
                    return fallback

                def _positive_or(default_value, fallback_value):
                    try:
                        if default_value is None:
                            raise ValueError
                        value = float(default_value)
                        if not math.isfinite(value) or value <= 0.0:
                            raise ValueError
                        return value
                    except Exception:
                        return fallback_value

                rh_steps = _posterior_get('rhmc_steps', None)
                if rh_steps is None:
                    try:
                        rh_steps = self.config.settings.model.posterior.rhmc_steps
                    except Exception:
                        rh_steps = None
                if rh_steps is None:
                    print("[RHMC ENFORCE] ⚠️ rhmc_steps missing; falling back to 8")
                    rh_steps = 8
                try:
                    rh_steps = int(rh_steps)
                except Exception as exc:
                    raise ValueError(f"Invalid rhmc_steps value '{rh_steps}': {exc}")

                rh_eps = _posterior_get('rhmc_step_size', None)
                if rh_eps is None:
                    try:
                        rh_eps = self.config.settings.model.posterior.rhmc_step_size
                    except Exception:
                        rh_eps = None
                if rh_eps is None:
                    print("[RHMC ENFORCE] ⚠️ rhmc_step_size missing; falling back to 0.03")
                    rh_eps = 0.03
                import os
                if os.environ.get("RLVAE_DEBUG", "0") == "1":
                    print(f"[RUN_EXP DEBUG] rhmc_step_size from config: {rh_eps}")
                try:
                    rh_eps = float(rh_eps)
                    if os.environ.get("RLVAE_DEBUG", "0") == "1":
                        print(f"[RUN_EXP DEBUG] rhmc_step_size after float(): {rh_eps}")
                except Exception as e:
                    raise ValueError(f"Invalid rhmc_step_size value '{rh_eps}': {e}")
                rh_alpha = _posterior_get('rhmc_alpha', getattr(self.config.model, 'rhmc_alpha', None))
                if rh_alpha is None:
                    raise ValueError("rhmc_alpha must be set in the main config (model.posterior.rhmc_alpha)")
                rh_alpha = _positive_or(rh_alpha, None)
                if rh_alpha is None:
                    raise ValueError("rhmc_alpha must be a positive finite value in the main config")
                # Hard override via environment takes precedence for end-to-end consistency
                try:
                    env_alpha = os.environ.get('RLVAE_ALPHA', None)
                    if env_alpha is not None:
                        rh_alpha = float(env_alpha)
                        if os.environ.get("RLVAE_DEBUG", "0") == "1":
                            print(f"[RUN_EXP DEBUG] RLVAE_ALPHA override detected → rh_alpha={rh_alpha}")
                except Exception:
                    pass
                # Persist resolved alpha into config and model defaults to avoid divergence
                try:
                    self._set_config_value(self.config, 'model.rhmc_alpha', float(rh_alpha))
                    self._set_config_value(self.config, 'model.posterior_local_alpha', float(rh_alpha))
                except Exception:
                    pass
                rh_eps_reg = _positive_or(
                    _posterior_get('rhmc_eps_reg', getattr(self.config.model, 'rhmc_eps_reg', None)),
                    1e-4
                )
                safeties = {}
                for key, default in (
                    ('max_momentum_norm', 3.0),
                    ('max_velocity_norm', 2.0),
                    ('max_position_step', 1.0),
                    ('max_position_norm', 0.0),
                ):
                    val = _posterior_get(key, default)
                    try:
                        safeties[key] = float(val)
                    except Exception:
                        safeties[key] = float(default)
                stagec_debugger.log_event(
                    "stagec_rhmc_params_resolved",
                    {
                        "rhmc_steps": rh_steps,
                        "rhmc_step_size": rh_eps,
                        "alpha": rh_alpha,
                        "eps_reg": rh_eps_reg,
                        "safeties": safeties,
                        "kl_metric_eval_point": getattr(self.config.model, 'kl_metric_eval_point', 'z'),
                        "env_alpha": os.environ.get("RLVAE_ALPHA"),
                    },
                )
                kl_eval = getattr(self.config.model, 'kl_metric_eval_point', 'z')
                kl_norm = getattr(self.config.model, 'kl_use_metric_normalization', False)
                kl_norm_mode = getattr(self.config.model, 'kl_metric_norm_mode', 'none')
                try:
                    mu_l2_w = float(getattr(self.config.model, 'mu_l2_weight', 0.0))
                except Exception:
                    mu_l2_w = 2.0

                # Mirror into posterior config mirrors
                self._set_config_value_if_missing(self.config, "model.posterior.rhmc_steps", rh_steps)
                self._set_config_value_if_missing(self.config, "model.posterior.rhmc_step_size", rh_eps)
                self._set_config_value_if_missing(self.config, "model.posterior.rhmc_alpha", rh_alpha)
                self._set_config_value_if_missing(self.config, "model.posterior.rhmc_eps_reg", rh_eps_reg)
                for k, v in safeties.items():
                    self._set_config_value_if_missing(self.config, f"model.posterior.{k}", v)

                # Top-level mirrors expected by downstream code
                self._set_config_value_if_missing(self.config, "model.rhmc_steps", rh_steps)
                self._set_config_value_if_missing(self.config, "model.rhmc_step_size", rh_eps)
                self._set_config_value_if_missing(self.config, "model.rhmc_alpha", rh_alpha)
                self._set_config_value_if_missing(self.config, "model.rhmc_eps_reg", rh_eps_reg)
                self._set_config_value_if_missing(self.config, "model.kl_metric_eval_point", kl_eval)
                self._set_config_value_if_missing(self.config, "model.kl_use_metric_normalization", kl_norm)
                self._set_config_value_if_missing(self.config, "model.kl_metric_norm_mode", kl_norm_mode)
                self._set_config_value_if_missing(self.config, "model.mu_l2_weight", mu_l2_w)

                # Training mirrors (Lightning Trainer expects them)
                self._set_config_value_if_missing(self.config, "training.model.posterior.rhmc_steps", rh_steps)
                self._set_config_value_if_missing(self.config, "training.model.posterior.rhmc_step_size", rh_eps)
                self._set_config_value_if_missing(self.config, "training.model.posterior.rhmc_alpha", rh_alpha)
                self._set_config_value_if_missing(self.config, "training.model.posterior.rhmc_eps_reg", rh_eps_reg)
                self._set_config_value_if_missing(self.config, "training.model.kl_metric_eval_point", kl_eval)
                self._set_config_value_if_missing(self.config, "training.model.kl_use_metric_normalization", kl_norm)
                self._set_config_value_if_missing(self.config, "training.model.kl_metric_norm_mode", kl_norm_mode)
                self._set_config_value_if_missing(self.config, "training.model.mu_l2_weight", mu_l2_w)

                print(f"[Stage C] 🔒 Synced RHMC params (steps={rh_steps}, step_size={rh_eps}, alpha={rh_alpha}) and KL settings (eval={kl_eval}, norm={kl_norm}, mode={kl_norm_mode}, mu_l2={mu_l2_w})")
            except Exception as e:
                print(f"[Stage C] ⚠️ Error during RHMC/KL enforcement: {e}")

            # ============================================================================
            # ============================================================================
            # CRITICAL: Enforce correct flows count (sequence_length - 1)
            # ============================================================================
            try:
                seq_len = int(getattr(self.config.data, 'sequence_length', 8))
                correct_n_flows = max(0, seq_len - 1)
                
                print(f"[Stage C] 🔧 Enforcing flows count: sequence_length={seq_len} → n_flows={correct_n_flows}")
                
                self._set_config_value_if_missing(self.config, "model.sequence_length", seq_len)
                self._set_config_value_if_missing(self.config, "model.n_flows", correct_n_flows)
                self._set_config_value_if_missing(self.config, "training.model.n_flows", correct_n_flows)
                print(f"[Stage C] ✅ Also synced training.model.n_flows = {correct_n_flows}")
                    
            except Exception as e:
                print(f"[Stage C] ⚠️ Error during flows count enforcement: {e}")
            # ============================================================================
            
            print(f"[Stage C] Set model parameters:")
            try:
                post_type = getattr(self.config.model.posterior, 'type', 'n/a') if hasattr(self.config.model, 'posterior') else 'n/a'
                post_type_toplevel = getattr(self.config.model, 'posterior_type', 'n/a')
                training_post_type = getattr(self.config.training.model.posterior, 'type', 'n/a') if hasattr(self.config.training, 'model') and hasattr(self.config.training.model, 'posterior') else 'n/a'
            except Exception:
                post_type = 'n/a'
                post_type_toplevel = 'n/a'
                training_post_type = 'n/a'
            try:
                n_flows_val = self.config.model.n_flows if hasattr(self.config.model, 'n_flows') else 'n/a'
                seq_len_val = self.config.model.sequence_length if hasattr(self.config.model, 'sequence_length') else 'n/a'
            except Exception:
                n_flows_val = 'n/a'
                seq_len_val = 'n/a'
            try:
                rkm_val = self.config.model.riemannian_kl_mode if ('riemannian_kl_mode' in self.config.model) else 'n/a'
            except Exception:
                rkm_val = 'n/a'
            try:
                rbeta_val = self.config.model.riemannian_beta if ('riemannian_beta' in self.config.model) else 'n/a'
            except Exception:
                rbeta_val = 'n/a'
            print(f"  - model.posterior.type: {post_type}")
            print(f"  - model.posterior_type: {post_type_toplevel}")
            print(f"  - training.model.posterior.type: {training_post_type}")
            print(f"  - sequence_length: {seq_len_val}")
            print(f"  - n_flows: {n_flows_val}")
            print(f"  - riemannian_kl_mode: {rkm_val}")
            print(f"  - riemannian_beta: {rbeta_val}")
            print(f"  - reconstruction_mode: all (temporal reconstruction enabled)")
            print(f"  - encode_all_timesteps_if_no_flows: True")
            print(f"  - kl_over_all_timesteps_if_flows: False (KL only on z₀)")
            # Respect CLI overrides: only set defaults if not provided
            # Posterior type default (do not override if already set)
            current_post = getattr(self.config.model.posterior, 'type', None) if hasattr(self.config.model, 'posterior') else None
            if current_post in (None, ""):
                self._set_config_value_if_missing(self.config, "model.posterior.type", 'riem_hmc')

            try:
                level = str(getattr(self.config.visualization, 'level', 'standard')).lower()
            except Exception:
                level = 'standard'
            if level == 'none':
                print("[Stage C] Visualization level 'none' detected; skipping default overrides")
            else:
                self._set_config_value_if_missing(self.config, "visualization.level", level or 'standard')
                self._set_config_value_if_missing(self.config, "visualization.enable_basic", True)
                self._set_config_value_if_missing(self.config, "visualization.enable_manifold", True)
                self._set_config_value_if_missing(self.config, "visualization.basic_frequency", max(1, int(getattr(self.config.visualization, 'basic_frequency', 5))))
                self._set_config_value_if_missing(self.config, "visualization.manifold_frequency", max(1, int(getattr(self.config.visualization, 'manifold_frequency', 10))))
            # Disable heavy evaluation during quick Stage C to keep pipeline fast and clean
            self._set_config_value_if_missing(self.config, "evaluation.enabled", False)
            self._set_config_value_if_missing(self.config, "evaluation.generation.enabled", False)
            self._set_config_value_if_missing(self.config, "evaluation.fid.enabled", False)
            self._set_config_value_if_missing(self.config, "evaluation.inference.enabled", False)
            self._set_config_value_if_missing(self.config, "evaluation.reconstruction.enabled", False)
            print("[Stage C] Evaluation disabled for modular Stage C (clean, fast run)")

            # Link Stage B metric as an input artifact for provenance
            if cfg.wandb.mode != "disabled" and getattr(cfg.wandb, 'artifacts', {}).get('enabled', False):
                try:
                    # Use previously logged stageB artifact if present
                    alias = self._resolve_wandb_alias(cfg, 'stage_b_latest', 'stageB_latest')
                    art_name = f"stageB_metric_{arch}_ld{latent_dim}:{alias}"
                    wandb.use_artifact(art_name)
                except Exception as e:
                    print(f"[Stage C] ⚠️ Could not reference Stage B artifact: {e}")
            self.run_single_experiment()

            # Stage C latent PCA + det(G^{-1}) heatmap with centroids overlay
            try:
                import matplotlib.pyplot as plt
                import numpy as np
                # Load Stage B metric for centroids overlay
                if stage_b_data is not None:
                    stateB = torch.load(stage_b_data['metric_path'], map_location='cpu', weights_only=False)
                    Cb = stateB.get('centroids')
                else:
                    Cb = None
                # Build small dataloader for val set
                dm = build_data_module(cfg.data)
                dm.setup('fit', cfg.training)
                vl = dm.val_dataloader()
                # Reuse trained wrapper if available to avoid reloading metric
                model_wrapper = getattr(self, '_stage_c_model_wrapper', None)
                if model_wrapper is None:
                    model_wrapper = LightningRlVAETrainer(cfg, data_module=dm)
                model = model_wrapper.model.to(self.device)
                model.eval()
                mus = []
                with torch.no_grad():
                    for xb in vl:
                        xb = xb.to(self.device)
                        x0 = xb[:,0]
                        enc = model.encoder(x0)
                        mu = enc.embedding
                        mus.append(mu.detach().cpu())
                        if sum(t.shape[0] for t in mus) >= 4000:
                            break
                if mus:
                    Z = torch.cat(mus, dim=0)
                    Zc = Z - Z.mean(dim=0, keepdim=True)
                    U, S, Vh = torch.linalg.svd(Zc, full_matrices=False)
                    comp = Vh[:2].T
                    mean = Z.mean(dim=0)
                    Proj = (Zc @ comp).numpy()
                    x_min, x_max = np.percentile(Proj[:,0], [1,99])
                    y_min, y_max = np.percentile(Proj[:,1], [1,99])
                    xs = np.linspace(x_min, x_max, 100)
                    ys = np.linspace(y_min, y_max, 100)
                    XX, YY = np.meshgrid(xs, ys)
                    pts = np.stack([XX.reshape(-1), YY.reshape(-1)], axis=1)
                    Zgrid = mean.unsqueeze(0) + torch.from_numpy(pts).to(mean.dtype) @ comp.T
                    Zgrid = Zgrid.to(self.device)
                    Ginv = model.modular_metric.compute_inverse_metric(Zgrid)
                    half_logdet = half_logdet_volume(Ginv, 'ginv')
                    det_vals = torch.exp(2.0 * half_logdet).detach().cpu().numpy().reshape(XX.shape)
                    plt.figure(figsize=(7,6))
                    plt.imshow(det_vals, origin='lower', extent=[x_min, x_max, y_min, y_max], cmap='viridis', aspect='auto')
                    P = ((Z - mean) @ comp).numpy()
                    plt.scatter(P[:,0], P[:,1], s=3, c='white', alpha=0.15, label='μ (val t=0)')
                    if Cb is not None:
                        Pc = ((Cb - mean).detach().cpu() @ comp).numpy()
                        plt.scatter(Pc[:,0], Pc[:,1], s=40, c='red', marker='*', label='centroids (B)')
                    # Prior samples from Stage C
                    try:
                        zs = model.sample_prior(1000, method='geodesic').detach().cpu()
                        Ps = ((zs - mean) @ comp).numpy()
                        plt.scatter(Ps[:,0], Ps[:,1], s=3, c='deepskyblue', alpha=0.3, label='prior samples')
                    except Exception:
                        pass
                    plt.colorbar(label='det(G^{-1}) (PCA subspace)')
                    plt.title('Stage C: det(G^{-1}) in PCA(2) subspace')
                    plt.legend(loc='upper right', fontsize=8)
                    plt.tight_layout()
                    if wandb.run is not None:
                        wandb.log({"stageC/pca_det_heatmap": wandb.Image(plt.gcf())})
                    plt.close()
            except Exception as e:
                print(f"[Stage C] ⚠️ PCA det(G^-1) heatmap failed: {e}")
            
            # Save Stage C configuration
            stageC_config = {
                'stage': 'C',
                'model_type': 'RLVAE',
                'architecture': arch,
                'latent_dim': latent_dim,
                'epochs': cfg.experiment.stage_c.epochs,
                'riemannian_beta': cfg.experiment.stage_c.riemannian_beta,
                'stage_a_source': str(stage_a_data['base_dir']) if stage_a_data else None,
                'stage_b_source': str(stage_b_data['base_dir']) if stage_b_data else None,
                'timestamp': datetime.now().isoformat()
            }
            with open(stageC_paths['config_path'], 'w') as f:
                yaml.dump(stageC_config, f)
            print(f"[Stage C] ✅ Saved Stage C config to {stageC_paths['config_path']}")
            # After Stage C, snapshot and log the trained metric (if trainable) as an artifact
            artifacts_enabled_c = self._wandb_artifacts_enabled(cfg)
            if cfg.wandb.mode != "disabled" and artifacts_enabled_c:
                try:
                    # Try to locate last saved snapshot
                    last_snap = None
                    snap_dir = Path("metric_snapshots")
                    if snap_dir.exists():
                        snaps = sorted(snap_dir.glob("metric_epoch_*.pt"))
                        if snaps:
                            last_snap = snaps[-1]
                    if last_snap is not None and last_snap.exists():
                        art = wandb.Artifact(
                            name=f"stageC_metric_{arch}_ld{latent_dim}",
                            type="metric",
                            metadata={"stage": "C", "architecture": arch, "latent_dim": latent_dim}
                        )
                        art.add_file(str(last_snap))
                        alias = self._resolve_wandb_alias(cfg, 'stage_c_latest', 'stageC_latest')
                        aliases = [alias]
                        wandb.log_artifact(art, aliases=aliases)
                except Exception as e:
                    print(f"[Stage C] ⚠️ Artifact logging failed: {e}")
            # Optional summary run aggregating key graphs
            if cfg.wandb.mode != "disabled":
                try:
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    plt.style.use('default'); sns.set_palette("husl")
                    state = torch.load(metric_file, map_location='cpu', weights_only=False)
                    M_matrices = state.get('M_matrices')
                    if M_matrices is None:
                        M_matrices = state.get('metric_matrices')
                    if M_matrices is None:
                        M_matrices = state.get('inverse_metrics')
                    centroids = state.get('centroids')
                    if M_matrices is not None and centroids is not None:
                        eigenvals = torch.linalg.eigvals(M_matrices).real
                        min_eigenvals = eigenvals.min(dim=-1)[0]
                        max_eigenvals = eigenvals.max(dim=-1)[0]
                        determinants = torch.linalg.det(M_matrices)
                        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
                        axes[0].hist(min_eigenvals.numpy(), bins=40, color='red', edgecolor='black'); axes[0].set_title('Min Eig')
                        axes[1].hist(max_eigenvals.numpy(), bins=40, color='blue', edgecolor='black'); axes[1].set_title('Max Eig')
                        axes[2].hist((torch.log10(torch.abs(determinants)+1e-50)).numpy(), bins=40, color='purple', edgecolor='black'); axes[2].set_title('Log10|Det|')
                        plt.tight_layout(); wandb.log({"summary/stageB/metric_overview": wandb.Image(fig)}); plt.close(fig)
                        art = wandb.Artifact(f"stageB_metric_{arch}_ld{latent_dim}", type="metric"); art.add_file(str(metric_file)); wandb.log_artifact(art)
                    # Stage A quick recon grid (if Stage A model was saved)
                    try:
                        data_train_path = getattr(cfg.data, 'train_path', None)
                        if 'model' in comp_paths and data_train_path is not None:
                            from scripts.train_diverse_metric_vae import create_model as create_stage1
                            import torchvision.utils as vutils
                            stage1_arch = arch.lower().replace('_gray', '') if '_gray' in arch.lower() else arch
                            stage1 = create_stage1(stage1_arch, input_dim=(cfg.data.channels, cfg.data.image_size[0], cfg.data.image_size[1]), latent_dim=latent_dim)
                            stage1.load_state_dict(torch.load(comp_paths['model'], map_location='cpu', weights_only=False))
                            stage1.eval()
                            raw = torch.load(data_train_path, map_location='cpu')
                            if raw.ndim == 5:
                                # [B, S, C, H, W] -> flatten
                                b, s = raw.shape[:2]
                                raw = raw.reshape(b*s, *raw.shape[2:])
                            batch = raw[:8]
                            with torch.no_grad():
                                if stage1_arch in ["mlp", "pythae"]:
                                    out = stage1({"data": batch})
                                    recon = out.recon_x.clamp(0, 1)
                                else:
                                    out = stage1(batch)
                                    recon = out.recon_x.clamp(0, 1)
                            grid = vutils.make_grid(torch.cat([batch[:8], recon[:8]], dim=0), nrow=8, normalize=False)
                            wandb.log({"summary/stageA/final_recon_grid": wandb.Image(grid)})
                        elif 'model' in comp_paths:
                            print("[SUMMARY] ⚠️ Skipping Stage A recon grid logging because data.train_path is not set")
                    except Exception as e:
                        print(f"[SUMMARY] ⚠️ Could not log Stage A recon grid: {e}")

                    # Stage C test metrics
                    if isinstance(self.results, dict) and 'test_results' in self.results and isinstance(self.results['test_results'], dict) and len(self.results['test_results']):
                        payload = {}
                        for k, v in self.results['test_results'].items():
                            if isinstance(v, torch.Tensor):
                                try:
                                    v = v.item()
                                except Exception:
                                    continue
                            payload[f"summary/stageC/{k}"] = v
                        if payload:
                            wandb.log(payload)
                    # keep run open
                except Exception as e:
                    print(f"[SUMMARY] ⚠️ WandB summary logging failed: {e}")

            # Restore base configuration after Stage C
            self.config = stage_c_base
            cfg = self.config
    
    def _create_model_config(self, model_name: str) -> DictConfig:
        """Create configuration for a specific model variant."""
        config = OmegaConf.structured(self.config)
        
        # Apply model-specific overrides
        if model_name == 'vanilla_vae':
            config.model.n_flows = 0
            config.model.riemannian_beta = 0.0
            config.model.posterior.type = 'gaussian'
            config.model.sampling.use_riemannian = False
            config.model.sampling.method = 'standard'
            config.model.loop.mode = 'open'
            config.model.loop.penalty = 0.0
        elif model_name == 'riemannian_flow_vae':
            # Use default Riemannian configuration
            pass
        
        # Apply experiment overrides
        if hasattr(self.config.experiment, 'training_override'):
            config.training.update(self.config.experiment.training_override)
        
        if hasattr(self.config.experiment, 'visualization_override'):
            config.visualization.update(self.config.experiment.visualization_override)
        
        return config
    
    def _setup_wandb(self, run_name: str) -> Optional[WandbLogger]:
        """Setup Weights & Biases logging."""
        if self.config.wandb.mode == "disabled":
            return None
        
        # For pipeline experiments, include stage information in the run name
        if hasattr(self.config.experiment, 'type') and self.config.experiment.type == "pipeline":
            # Extract architecture and latent_dim for consistent naming
            architecture = getattr(self.config.experiment.stage1, 'architecture', 'mlp')
            latent_dim = getattr(self.config.experiment.stage1, 'latent_dim', 16)
            full_run_name = f"pipeline_stage2_rlvae_{architecture}_ld{latent_dim}"
        else:
            # Create unique run name for other experiments
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            full_run_name = f"{run_name}_{timestamp}"
        
        entity = getattr(self.config.wandb, 'entity', None)
        kwargs = dict(
            project=self.config.wandb.project,
            name=full_run_name,
            mode=self.config.wandb.mode,
            tags=self.config.wandb.get('tags', []),
            config=OmegaConf.to_container(self.config, resolve=True)
        )
        if entity is not None:
            kwargs['entity'] = entity
        wandb_logger = WandbLogger(**kwargs)
        
        return wandb_logger
    
    def _create_trainer(self, wandb_logger) -> L.Trainer:
        """Create Lightning trainer."""
        callbacks = []
        
        # Early stopping
        if hasattr(self.config.training, 'early_stopping'):
            early_stop = EarlyStopping(
                monitor=self.config.training.early_stopping.monitor,
                patience=self.config.training.early_stopping.patience,
                mode=self.config.training.early_stopping.mode
            )
            callbacks.append(early_stop)
        
        # Model checkpointing
        checkpoint = ModelCheckpoint(
            monitor=self.config.training.logging.monitor,
            save_top_k=self.config.training.logging.save_top_k,
            mode=self.config.training.logging.mode,
            dirpath=self.output_dir / "checkpoints",
            filename="{epoch:02d}-{val_loss:.3f}"
        )
        callbacks.append(checkpoint)
        
        # Create trainer with new config structure
        tt = self.config.training.trainer
        trainer_kwargs = {
            'max_epochs': getattr(tt, 'max_epochs', 3),
            'accelerator': getattr(tt, 'accelerator', 'auto'),
            'devices': getattr(tt, 'devices', 1),
            'precision': getattr(tt, 'precision', '16-mixed'),
            'log_every_n_steps': getattr(tt, 'log_every_n_steps', 10),
            'val_check_interval': getattr(tt, 'val_check_interval', 1.0),
            'num_sanity_val_steps': getattr(tt, 'num_sanity_val_steps', 0),
            'enable_progress_bar': getattr(tt, 'enable_progress_bar', True),
            'enable_model_summary': getattr(tt, 'enable_model_summary', True),
            'deterministic': getattr(tt, 'deterministic', False),
            'logger': wandb_logger,
            'callbacks': callbacks,
        }
        strategy = getattr(tt, 'strategy', None)
        if strategy is not None:
            trainer_kwargs['strategy'] = strategy
        
        trainer = L.Trainer(**trainer_kwargs)
        
        return trainer
    
    def _extract_comparison_metrics(self, test_results: Dict, metric_names: List[str]) -> Dict[str, float]:
        """Extract specific metrics for comparison."""
        extracted = {}
        
        for metric_name in metric_names:
            if metric_name in test_results:
                value = test_results[metric_name]
                if isinstance(value, torch.Tensor):
                    value = value.item()
                extracted[metric_name] = value
        
        return extracted
    
    def _save_results(self):
        """Save experiment results."""
        results_path = self.output_dir / "results.yaml"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_path, 'w') as f:
            OmegaConf.save(OmegaConf.create(self.results), f)
        
        print(f"💾 Results saved to: {results_path}")
    
    def _print_comparison_summary(self, summary: Dict[str, Any]):
        """Print comparison summary to console."""
        print(f"\n📊 COMPARISON SUMMARY")
        print("=" * 60)
        
        for model_name, metrics in summary.items():
            print(f"\n🎯 {model_name.upper()}:")
            for metric_name, value in metrics.items():
                if metric_name.endswith('_final'):
                    clean_name = metric_name.replace('_final', '')
                    print(f"   {clean_name}: {value:.4f}")

def get_stage_folder_name(stage, model_type, architecture, latent_dim, dataset_name="SPRITES"):
    """
    Create organized folder names for the three-stage pipeline.
    
    Args:
        stage: 'A', 'B', or 'C'
        model_type: 'RHVAE', 'VANILLA', 'PRECISION', etc.
        architecture: 'MLP', 'CNN', etc.
        latent_dim: latent dimension as int
        dataset_name: dataset name (default: "SPRITES")
    
    Returns:
        Folder name like 'A_RHVAE_MLP_16_SPRITES' or 'B_VANILLA_MLP_16_SPRITES'
    """
    return f"{stage}_{model_type}_{architecture.upper()}_{latent_dim}_{dataset_name}"

def get_stage_paths(cfg, stage, model_type, architecture, latent_dim, dataset_name=None):
    """
    Get organized paths for a specific stage.
    
    Args:
        cfg: configuration object
        stage: 'A', 'B', or 'C'
        model_type: model type for the stage
        architecture: architecture type
        latent_dim: latent dimension
    
    Returns:
        dict with organized paths
    """
    if dataset_name is None:
        try:
            dataset_name = str(getattr(cfg.data, 'dataset', 'SPRITES')).upper()
        except Exception:
            dataset_name = "SPRITES"
    stage_folder = get_stage_folder_name(stage, model_type, architecture, latent_dim, dataset_name)
    base_dir = Path(cfg.output_dir) / "stages" / stage_folder
    base_dir.mkdir(parents=True, exist_ok=True)
    
    if stage == 'A':
        return {
            'base_dir': base_dir,
            'encoder_path': base_dir / 'encoder.pt',
            'decoder_path': base_dir / 'decoder.pt',
            'model_path': base_dir / 'model.pt',
            'metric_path': base_dir / 'metric.pt',
            'config_path': base_dir / 'config.yaml'
        }
    elif stage == 'B':
        return {
            'base_dir': base_dir,
            'metric_path': base_dir / 'metric.pt',
            'samples_path': base_dir / 'rhmc_samples.pt',
            'config_path': base_dir / 'config.yaml'
        }
    elif stage == 'C':
        return {
            'base_dir': base_dir,
            'checkpoint_path': base_dir / 'checkpoint.pt',
            'config_path': base_dir / 'config.yaml'
        }
    else:
        raise ValueError(f"Unknown stage: {stage}")

def find_stage_a_data(cfg, architecture, latent_dim):
    """
    Automatically find Stage A data from organized folders.
    
    Args:
        cfg: configuration object
        architecture: architecture type
        latent_dim: latent dimension
    
    Returns:
        dict with Stage A paths or None if not found
    """
    stages_dir = Path(cfg.output_dir) / "stages"
    if not stages_dir.exists():
        return None
    try:
        dataset_name = str(getattr(cfg.data, 'dataset', 'SPRITES')).upper()
    except Exception:
        dataset_name = "SPRITES"
    dataset_candidates = [dataset_name]
    if "SPRITES" not in dataset_candidates:
        dataset_candidates.append("SPRITES")
    
    # Decide preference order based on configured Stage A model
    prefer_vanilla = False
    try:
        model_choice = str(getattr(cfg.experiment.stage_a, 'model', '')).lower()
        prefer_vanilla = (model_choice.startswith('vanilla'))
    except Exception:
        prefer_vanilla = False
    possible_folders = []
    for ds_name in dataset_candidates:
        if prefer_vanilla:
            possible_folders.extend([
                get_stage_folder_name('A', 'VANILLA', architecture, latent_dim, ds_name),
                get_stage_folder_name('A', 'RHVAE', architecture, latent_dim, ds_name)
            ])
        else:
            possible_folders.extend([
                get_stage_folder_name('A', 'RHVAE', architecture, latent_dim, ds_name),
                get_stage_folder_name('A', 'VANILLA', architecture, latent_dim, ds_name)
            ])
    
    # Add debug info
    print(f"[Stage B] Looking for Stage A data in folders: {possible_folders}")
    
    for folder_name in possible_folders:
        stage_a_dir = stages_dir / folder_name
        if stage_a_dir.exists():
            # Check if it has the required files
            config_path = stage_a_dir / 'config.yaml'
            
            # Look for encoder and decoder files (handle both timestamped and simple naming, and both .pt and .pkl)
            encoder_files = (list(stage_a_dir.glob('encoder_*.pt')) + list(stage_a_dir.glob('encoder.pt')) + 
                           list(stage_a_dir.glob('encoder_*.pkl')) + list(stage_a_dir.glob('encoder.pkl')))
            decoder_files = (list(stage_a_dir.glob('decoder_*.pt')) + list(stage_a_dir.glob('decoder.pt')) + 
                           list(stage_a_dir.glob('decoder_*.pkl')) + list(stage_a_dir.glob('decoder.pkl')))
            metric_files = (list(stage_a_dir.glob('metric*.pt')) + list(stage_a_dir.glob('metric.pt')) + 
                          list(stage_a_dir.glob('metric*.pkl')) + list(stage_a_dir.glob('metric.pkl')))
            
            if encoder_files and decoder_files and config_path.exists():
                # Pick the most recent encoder/decoder by modification time
                encoder_path = max(encoder_files, key=lambda p: p.stat().st_mtime)
                decoder_path = max(decoder_files, key=lambda p: p.stat().st_mtime)
                metric_path = metric_files[0] if metric_files else None
                print(f"[Stage B] Found Stage A data in: {stage_a_dir}")
                return {
                    'base_dir': stage_a_dir,
                    'metric_path': metric_path,
                    'encoder_path': encoder_path,
                    'decoder_path': decoder_path,
                    'config_path': config_path
                }

    print(f"[Stage B] No Stage A data found for {architecture}_{latent_dim}")
    # Fallback: generic stageA directory if staged runs are missing
    try:
        fallback_dir = Path(cfg.output_dir) / "stageA"
        enc_fallback = fallback_dir / "encoder.pt"
        dec_fallback = fallback_dir / "decoder.pt"
        cfg_fallback = fallback_dir / "config.yaml"
        if enc_fallback.exists() and dec_fallback.exists():
            print(f"[Stage B] Using fallback Stage A data in {fallback_dir}")
            return {
                'base_dir': fallback_dir,
                'metric_path': None,
                'encoder_path': enc_fallback,
                'decoder_path': dec_fallback,
                'config_path': cfg_fallback if cfg_fallback.exists() else None,
            }
    except Exception:
        pass
    return None

def find_stage_b_data(cfg, architecture, latent_dim, metric_type):
    """
    Automatically find Stage B data from organized folders.
    
    Args:
        cfg: configuration object
        architecture: architecture type
        latent_dim: latent dimension
        metric_type: metric type (RHVAE, PRECISION, etc.)
    
    Returns:
        dict with Stage B paths or None if not found
    """
    stages_dir = Path(cfg.output_dir) / "stages"
    if not stages_dir.exists():
        return None
    try:
        dataset_name = str(getattr(cfg.data, 'dataset', 'SPRITES')).upper()
    except Exception:
        dataset_name = "SPRITES"
    dataset_candidates = [dataset_name]
    if "SPRITES" not in dataset_candidates:
        dataset_candidates.append("SPRITES")
    
    # Look for Stage B folder
    stage_b_dir = None
    metric_path = None
    config_path = None
    for ds_name in dataset_candidates:
        folder_name = get_stage_folder_name('B', metric_type, architecture, latent_dim, ds_name)
        candidate_dir = stages_dir / folder_name
        metric_candidate = candidate_dir / 'metric.pt'
        config_candidate = candidate_dir / 'config.yaml'
        if candidate_dir.exists() and metric_candidate.exists() and config_candidate.exists():
            stage_b_dir = candidate_dir
            metric_path = metric_candidate
            config_path = config_candidate
            break
    
    if stage_b_dir is not None:
        print(f"[Stage C] Found Stage B data in: {stage_b_dir}")
        return {
            'base_dir': stage_b_dir,
            'metric_path': metric_path,
            'config_path': config_path
        }
    
    print(f"[Stage C] No Stage B data found for {metric_type}_{architecture}_{latent_dim}")
    return None

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(config: DictConfig) -> None:
    """Main entry point with Hydra configuration."""
    
    print("🧠 RlVAE Experiment Runner with Hydra")
    print("=" * 50)
    
    # Print configuration
    print("📋 Configuration:")
    from omegaconf import OmegaConf
    print(OmegaConf.to_yaml(config))
    
    # Print model.metric config for extra debugging
    print("[HYDRA DEBUG] config.model.metric:", OmegaConf.to_yaml(config.model.metric) if hasattr(config, 'model') and hasattr(config.model, 'metric') else 'N/A')
    
    # Run experiment
    runner = ExperimentRunner(config)
    runner.run()
    
    print("\n🏁 Experiment completed successfully!")


if __name__ == "__main__":
    main() 

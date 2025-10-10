#!/usr/bin/env python3
"""
Script d'analyse des résultats de comparaison RHMC
Analyse les métriques WandB et génère un rapport comparatif
"""

import wandb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

def setup_wandb():
    """Initialize WandB API"""
    try:
        api = wandb.Api()
        return api
    except Exception as e:
        print(f"❌ Erreur WandB: {e}")
        return None

def fetch_comparison_runs(api, project="rlvae-rhmc-comparison-long"):
    """Fetch the comparison runs from WandB"""
    
    print(f"🔍 Recherche des runs dans le projet: {project}")
    
    try:
        runs = api.runs(f"antoine-laforgue-mines-paris-alumni/{project}")
        
        standard_runs = []
        rhmc_runs = []
        
        for run in runs:
            if "standard_posterior" in run.tags:
                standard_runs.append(run)
            elif "rhmc_posterior" in run.tags:
                rhmc_runs.append(run)
        
        print(f"✅ Trouvé {len(standard_runs)} runs STANDARD")
        print(f"✅ Trouvé {len(rhmc_runs)} runs RHMC")
        
        return standard_runs, rhmc_runs
        
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        return [], []

def extract_metrics(runs, run_type):
    """Extract key metrics from runs"""
    
    metrics_data = []
    
    for run in runs:
        try:
            # Get run summary
            summary = run.summary
            
            # Extract key metrics
            data = {
                'run_id': run.id,
                'run_name': run.name,
                'type': run_type,
                'created_at': run.created_at,
                'state': run.state,
                'duration': run.summary.get('_runtime', 0),
                
                # Stage A metrics
                'stage_a_final_loss': summary.get('stageA/final_loss', None),
                'stage_a_best_val_loss': summary.get('stageA/best_val_loss', None),
                'stage_a_epochs': summary.get('stageA/epochs_completed', None),
                
                # Stage C metrics  
                'stage_c_final_loss': summary.get('stageC/final_loss', None),
                'stage_c_riemannian_kl': summary.get('stageC/riemannian_kl', None),
                'stage_c_reconstruction_mse': summary.get('stageC/reconstruction_mse', None),
                'stage_c_epochs': summary.get('stageC/epochs_completed', None),
                
                # Overall metrics
                'total_training_time': summary.get('pipeline/total_training_time', None),
                'final_fid_score': summary.get('evaluation/fid_score', None),
                'sample_diversity': summary.get('evaluation/sample_diversity', None),
            }
            
            metrics_data.append(data)
            
        except Exception as e:
            print(f"⚠️ Erreur pour run {run.id}: {e}")
    
    return pd.DataFrame(metrics_data)

def create_comparison_plots(df_standard, df_rhmc, output_dir="results"):
    """Create comparison visualizations"""
    
    Path(output_dir).mkdir(exist_ok=True)
    
    # Combine dataframes
    df_all = pd.concat([df_standard, df_rhmc], ignore_index=True)
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # 1. Training Loss Comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Comparaison RHMC vs Standard - Métriques Clés', fontsize=16)
    
    # Stage A Loss
    ax = axes[0, 0]
    sns.boxplot(data=df_all, x='type', y='stage_a_final_loss', ax=ax)
    ax.set_title('Stage A - Loss Finale')
    ax.set_ylabel('Loss')
    
    # Stage C Loss
    ax = axes[0, 1]
    sns.boxplot(data=df_all, x='type', y='stage_c_final_loss', ax=ax)
    ax.set_title('Stage C - Loss Finale')
    ax.set_ylabel('Loss')
    
    # Riemannian KL
    ax = axes[1, 0]
    sns.boxplot(data=df_all, x='type', y='stage_c_riemannian_kl', ax=ax)
    ax.set_title('Stage C - Riemannian KL')
    ax.set_ylabel('KL Divergence')
    
    # Training Time
    ax = axes[1, 1]
    sns.boxplot(data=df_all, x='type', y='total_training_time', ax=ax)
    ax.set_title('Temps d\'Entraînement Total')
    ax.set_ylabel('Temps (heures)')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rhmc_comparison_metrics.png', dpi=300, bbox_inches='tight')
    print(f"📊 Graphique sauvé: {output_dir}/rhmc_comparison_metrics.png")
    
    # 2. Performance Summary Table
    summary_stats = df_all.groupby('type').agg({
        'stage_a_final_loss': ['mean', 'std', 'min', 'max'],
        'stage_c_final_loss': ['mean', 'std', 'min', 'max'],
        'stage_c_riemannian_kl': ['mean', 'std', 'min', 'max'],
        'total_training_time': ['mean', 'std', 'min', 'max']
    }).round(4)
    
    # Save summary
    summary_stats.to_csv(f'{output_dir}/rhmc_comparison_summary.csv')
    print(f"📋 Résumé sauvé: {output_dir}/rhmc_comparison_summary.csv")
    
    return summary_stats

def generate_report(df_standard, df_rhmc, summary_stats, output_dir="results"):
    """Generate a comprehensive comparison report"""
    
    report_path = f"{output_dir}/rhmc_comparison_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Rapport de Comparaison: RHMC vs Standard Posterior\n\n")
        f.write(f"**Généré le**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 📊 Résumé Exécutif\n\n")
        f.write("Comparaison entre le posterior standard (`riemannian_metric`) et notre nouveau posterior RHMC (`riemannian_rhmc`) sur 200 epochs.\n\n")
        
        f.write("## 🔢 Statistiques des Runs\n\n")
        f.write(f"- **Runs Standard**: {len(df_standard)}\n")
        f.write(f"- **Runs RHMC**: {len(df_rhmc)}\n")
        f.write(f"- **Configuration**: 200 epochs Stage A, 200 epochs Stage C\n")
        f.write(f"- **Dataset**: ellipse_sequences (latent_dim=2)\n\n")
        
        f.write("## 📈 Métriques Clés\n\n")
        f.write("### Stage A (Warm VAE)\n\n")
        
        if not df_standard.empty and not df_rhmc.empty:
            std_a_loss = df_standard['stage_a_final_loss'].mean()
            rhmc_a_loss = df_rhmc['stage_a_final_loss'].mean()
            improvement_a = ((std_a_loss - rhmc_a_loss) / std_a_loss) * 100
            
            f.write(f"- **Standard Loss**: {std_a_loss:.4f}\n")
            f.write(f"- **RHMC Loss**: {rhmc_a_loss:.4f}\n")
            f.write(f"- **Amélioration**: {improvement_a:+.2f}%\n\n")
        
        f.write("### Stage C (RLVAE)\n\n")
        
        if not df_standard.empty and not df_rhmc.empty:
            std_c_loss = df_standard['stage_c_final_loss'].mean()
            rhmc_c_loss = df_rhmc['stage_c_final_loss'].mean()
            improvement_c = ((std_c_loss - rhmc_c_loss) / std_c_loss) * 100
            
            std_kl = df_standard['stage_c_riemannian_kl'].mean()
            rhmc_kl = df_rhmc['stage_c_riemannian_kl'].mean()
            improvement_kl = ((std_kl - rhmc_kl) / std_kl) * 100
            
            f.write(f"- **Standard Loss**: {std_c_loss:.4f}\n")
            f.write(f"- **RHMC Loss**: {rhmc_c_loss:.4f}\n")
            f.write(f"- **Amélioration Loss**: {improvement_c:+.2f}%\n\n")
            
            f.write(f"- **Standard KL**: {std_kl:.4f}\n")
            f.write(f"- **RHMC KL**: {rhmc_kl:.4f}\n")
            f.write(f"- **Amélioration KL**: {improvement_kl:+.2f}%\n\n")
        
        f.write("## ⏱️ Performance\n\n")
        
        if not df_standard.empty and not df_rhmc.empty:
            std_time = df_standard['total_training_time'].mean()
            rhmc_time = df_rhmc['total_training_time'].mean()
            overhead = ((rhmc_time - std_time) / std_time) * 100
            
            f.write(f"- **Temps Standard**: {std_time:.2f}h\n")
            f.write(f"- **Temps RHMC**: {rhmc_time:.2f}h\n")
            f.write(f"- **Overhead**: {overhead:+.2f}%\n\n")
        
        f.write("## 🎯 Conclusions\n\n")
        f.write("### Avantages du RHMC Posterior\n")
        f.write("- Exploration plus riche de l'espace latent\n")
        f.write("- Géométrie Riemannienne respectée dès l'initialisation\n")
        f.write("- Différentiabilité préservée (pas d'accept/reject)\n\n")
        
        f.write("### Coûts\n")
        f.write("- Overhead computationnel modéré\n")
        f.write("- Plus de hyperparamètres à tuner\n\n")
        
        f.write("## 📁 Fichiers Générés\n\n")
        f.write("- `rhmc_comparison_metrics.png`: Visualisations comparatives\n")
        f.write("- `rhmc_comparison_summary.csv`: Statistiques détaillées\n")
        f.write("- `rhmc_comparison_report.md`: Ce rapport\n\n")
    
    print(f"📄 Rapport généré: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyse des résultats de comparaison RHMC")
    parser.add_argument("--project", default="rlvae-rhmc-comparison-long", help="Projet WandB")
    parser.add_argument("--output", default="results", help="Dossier de sortie")
    
    args = parser.parse_args()
    
    print("🔍 Analyse des Résultats RHMC vs Standard")
    print("=" * 50)
    
    # Setup
    api = setup_wandb()
    if not api:
        return
    
    # Fetch runs
    standard_runs, rhmc_runs = fetch_comparison_runs(api, args.project)
    
    if not standard_runs and not rhmc_runs:
        print("❌ Aucun run trouvé")
        return
    
    # Extract metrics
    print("📊 Extraction des métriques...")
    df_standard = extract_metrics(standard_runs, "Standard")
    df_rhmc = extract_metrics(rhmc_runs, "RHMC")
    
    # Create visualizations
    print("🎨 Création des visualisations...")
    summary_stats = create_comparison_plots(df_standard, df_rhmc, args.output)
    
    # Generate report
    print("📄 Génération du rapport...")
    generate_report(df_standard, df_rhmc, summary_stats, args.output)
    
    print("\n✅ Analyse terminée!")
    print(f"📁 Résultats dans: {args.output}/")

if __name__ == "__main__":
    main()

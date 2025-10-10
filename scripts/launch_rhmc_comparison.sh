#!/bin/bash

# Script de lancement pour la comparaison RHMC
# Usage: ./scripts/launch_rhmc_comparison.sh

echo "🚀 Lancement Comparaison Posteriors RHMC - 200 Epochs"
echo "====================================================="

# Check if we're on a SLURM system
if command -v sbatch &> /dev/null; then
    echo "🖥️  Système SLURM détecté - Lancement via sbatch"
    
    # Create logs directory
    mkdir -p logs
    
    # Submit job
    job_id=$(sbatch scripts/run_rhmc_comparison_long.sbatch | grep -o '[0-9]*')
    
    if [ ! -z "$job_id" ]; then
        echo "✅ Job SLURM soumis avec succès!"
        echo "   Job ID: $job_id"
        echo "   Statut: squeue -j $job_id"
        echo "   Logs: logs/rhmc_comparison_long_${job_id}.out"
        echo ""
        echo "📊 Suivi en temps réel:"
        echo "   tail -f logs/rhmc_comparison_long_${job_id}.out"
        echo ""
        echo "🔗 WandB:"
        echo "   Projet: rlvae-rhmc-comparison-long"
        echo "   https://wandb.ai/antoine-laforgue-mines-paris-alumni/rlvae-rhmc-comparison-long"
    else
        echo "❌ Erreur lors de la soumission du job"
        exit 1
    fi
    
else
    echo "💻 Système local détecté - Lancement direct"
    echo "⚠️  Attention: Cela va prendre ~20-24h"
    echo ""
    
    read -p "Continuer? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🚀 Lancement des expériences..."
        
        # Create logs directory
        mkdir -p logs
        
        # Run the experiments directly
        bash scripts/run_rhmc_comparison_long.sbatch
    else
        echo "❌ Annulé par l'utilisateur"
        exit 0
    fi
fi

echo ""
echo "📋 Configuration:"
echo "  - Stage A: 200 epochs"
echo "  - Stage C: 200 epochs"
echo "  - Comparaison: riemannian_metric vs riemannian_rhmc"
echo "  - Data: ellipse_sequences (2D latent)"
echo ""
echo "⏱️  Temps estimé: 20-24 heures"
echo "🎯 Objectif: Comparaison robuste des posteriors"

#!/bin/bash
# Script de test rapide pour valider les corrections de configuration
# Usage: ./test_config_fix.sh

set -e

echo "═══════════════════════════════════════════════════════════════"
echo "TEST DE VALIDATION: Corrections Configuration Phase C"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Créer un répertoire pour les logs de test
TEST_DIR="test_diagnostic_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_DIR"

echo "📁 Logs seront sauvegardés dans: $TEST_DIR"
echo ""

# Activer les diagnostics
export RLVAE_DEBUG=1

echo "✓ RLVAE_DEBUG=1 activé"
echo ""

# Lancer le test (1 epoch, 2 batches)
echo "🚀 Lancement du test (1 epoch, 2 batches)..."
echo ""

python run_experiment.py \
  trainer.max_epochs=1 \
  trainer.limit_train_batches=2 \
  > "$TEST_DIR/test_full_log.txt" 2>&1

EXIT_CODE=$?

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "❌ ERREUR: Le test a échoué avec le code $EXIT_CODE"
    echo "   Voir les logs dans: $TEST_DIR/test_full_log.txt"
    exit $EXIT_CODE
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "ANALYSE DES RÉSULTATS"
echo "═══════════════════════════════════════════════════════════════"
echo ""

# Extraire les métriques clés
echo "📊 Métriques Clés Extraites:"
echo ""

# 1. KL Divergence (doit être positive!)
echo "1. KL DIVERGENCE:"
grep "FINAL KL LOSS" "$TEST_DIR/test_full_log.txt" | tail -1 || echo "   ⚠️  Non trouvé"
echo ""

# 2. RHMC Drift (ne doit plus être 0.0)
echo "2. RHMC DRIFT:"
grep "Total drift from z0" "$TEST_DIR/test_full_log.txt" | tail -1 || echo "   ⚠️  Non trouvé"
grep "Net change in" "$TEST_DIR/test_full_log.txt" | tail -1 || echo "   ⚠️  Non trouvé"
echo ""

# 3. Mahalanobis Distance (doit être ~2.0)
echo "3. MAHALANOBIS² (attendu: ~2.0):"
grep "Observed Mahal²" "$TEST_DIR/test_full_log.txt" | tail -1 || echo "   ⚠️  Non trouvé"
echo ""

# 4. Distance Ratio (doit être ~1.0)
echo "4. DISTANCE RATIO (attendu: ~1.0):"
grep "Ratio (actual/expected)" "$TEST_DIR/test_full_log.txt" | tail -1 || echo "   ⚠️  Non trouvé"
echo ""

# 5. log_q (doit être -3 à -5, pas -20 à -24)
echo "5. TOTAL LOG_Q (attendu: -3 à -5):"
grep "Total log_q:" "$TEST_DIR/test_full_log.txt" | tail -1 || echo "   ⚠️  Non trouvé"
echo ""

# Vérification automatique du succès
echo "═══════════════════════════════════════════════════════════════"
echo "VERDICT AUTOMATIQUE"
echo "═══════════════════════════════════════════════════════════════"
echo ""

KL_VALUE=$(grep "FINAL KL LOSS" "$TEST_DIR/test_full_log.txt" | tail -1 | grep -oP '[-+]?[0-9]*\.?[0-9]+' | tail -1 || echo "NaN")

if [[ "$KL_VALUE" =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
    if (( $(echo "$KL_VALUE > 0" | bc -l) )); then
        echo "✅ SUCCÈS: KL divergence est POSITIVE ($KL_VALUE)"
        echo "   → Les corrections ont fonctionné !"
        echo ""
        echo "🎉 Vous pouvez maintenant lancer le training complet:"
        echo "   python run_experiment.py"
        SUCCESS=true
    else
        echo "⚠️  ATTENTION: KL divergence est toujours NÉGATIVE ($KL_VALUE)"
        echo ""
        echo "📋 Actions recommandées:"
        echo "   1. Vérifier les logs complets: $TEST_DIR/test_full_log.txt"
        echo "   2. Chercher la section [INITIAL SAMPLING DIAGNOSTICS]"
        echo "   3. Ajuster rhmc_alpha selon les recommandations"
        echo ""
        
        # Extraire la distance ratio pour recommandation
        RATIO=$(grep "Ratio (actual/expected)" "$TEST_DIR/test_full_log.txt" | tail -1 | grep -oP '[0-9]+\.[0-9]+' || echo "NaN")
        
        if [[ "$RATIO" != "NaN" ]]; then
            if (( $(echo "$RATIO > 1.5" | bc -l) )); then
                echo "   → Distance ratio > 1.5: Essayer rhmc_alpha: 1.0 ou 2.0"
            elif (( $(echo "$RATIO < 0.5" | bc -l) )); then
                echo "   → Distance ratio < 0.5: Essayer rhmc_alpha: 0.2 ou 0.3"
            else
                echo "   → Distance ratio OK, problème ailleurs. Voir diagnostics complets."
            fi
        fi
        SUCCESS=false
    fi
else
    echo "❌ ERREUR: Impossible d'extraire la valeur de KL"
    echo "   Voir les logs: $TEST_DIR/test_full_log.txt"
    SUCCESS=false
fi

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "FICHIERS GÉNÉRÉS"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📄 Log complet: $TEST_DIR/test_full_log.txt"
echo ""

# Créer un résumé
{
    echo "TEST DE VALIDATION - RÉSUMÉ"
    echo "Date: $(date)"
    echo ""
    echo "KL Divergence: $KL_VALUE"
    echo "Succès: $SUCCESS"
    echo ""
    echo "Voir test_full_log.txt pour les détails complets"
} > "$TEST_DIR/summary.txt"

echo "📄 Résumé: $TEST_DIR/summary.txt"
echo ""

if [ "$SUCCESS" = true ]; then
    echo "🎯 TEST RÉUSSI !"
    exit 0
else
    echo "⚠️  TEST INCOMPLET - Voir recommandations ci-dessus"
    exit 1
fi


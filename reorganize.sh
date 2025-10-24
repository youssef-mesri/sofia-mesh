#!/bin/bash
#
# Script de réorganisation du projet SOFIA
# Ce script déplace les fichiers selon le plan de réorganisation
#

set -e  # Exit on error

cd "$(dirname "$0")"

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║         Réorganisation du Projet SOFIA                         ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Couleurs
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Phase 1: Créer les répertoires
echo -e "${BLUE}[Phase 1]${NC} Création des répertoires..."

mkdir -p docs/images
mkdir -p docs/dev/archive
mkdir -p benchmarks/results
mkdir -p experiments
mkdir -p examples/visualizations
mkdir -p scripts

echo -e "${GREEN}✓${NC} Répertoires créés"
echo ""

# Phase 2: Déplacer la documentation
echo -e "${BLUE}[Phase 2]${NC} Déplacement de la documentation..."

# Images de docs/ vers docs/images/
if [ -d "docs" ]; then
    mv docs/*.png docs/images/ 2>/dev/null || true
    echo -e "${GREEN}✓${NC} Images de documentation déplacées"
fi

# Documentation vers docs/
mv CITATION.md docs/ 2>/dev/null || echo "  CITATION.md déjà déplacé ou absent"
mv CODE_OF_CONDUCT.md docs/ 2>/dev/null || echo "  CODE_OF_CONDUCT.md déjà déplacé ou absent"
mv CONTRIBUTING.md docs/ 2>/dev/null || echo "  CONTRIBUTING.md déjà déplacé ou absent"
mv PUBLICATION_GUIDE.md docs/ 2>/dev/null || echo "  PUBLICATION_GUIDE.md déjà déplacé ou absent"
mv QUICK_START.md docs/ 2>/dev/null || echo "  QUICK_START.md déjà déplacé ou absent"

# Documentation dev
mv MOVING_MODULES.md docs/dev/ 2>/dev/null || echo "  MOVING_MODULES.md déjà déplacé ou absent"
mv REFACTORING_VISUAL.txt docs/dev/ 2>/dev/null || echo "  REFACTORING_VISUAL.txt déjà déplacé ou absent"

# Archives
mv PREPARATION_STATUS.md docs/dev/archive/ 2>/dev/null || echo "  PREPARATION_STATUS.md déjà déplacé ou absent"
mv PUBLICATION_COMPLETE.md docs/dev/archive/ 2>/dev/null || echo "  PUBLICATION_COMPLETE.md déjà déplacé ou absent"
mv SESSION_SUMMARY.md docs/dev/archive/ 2>/dev/null || echo "  SESSION_SUMMARY.md déjà déplacé ou absent"

echo -e "${GREEN}✓${NC} Documentation déplacée"
echo ""

# Phase 3: Déplacer les benchmarks
echo -e "${BLUE}[Phase 3]${NC} Déplacement des benchmarks..."

mv benchmark_*.py benchmarks/ 2>/dev/null || echo "  Benchmarks déjà déplacés ou absents"
mv batch_benchmark_results.json benchmarks/results/ 2>/dev/null || echo "  batch_benchmark_results.json déjà déplacé ou absent"
mv phase2_results.json benchmarks/results/ 2>/dev/null || echo "  phase2_results.json déjà déplacé ou absent"

echo -e "${GREEN}✓${NC} Benchmarks déplacés"
echo ""

# Phase 4: Déplacer les expérimentations
echo -e "${BLUE}[Phase 4]${NC} Déplacement des expérimentations RL..."

mv rl-ym.py experiments/ 2>/dev/null || echo "  rl-ym.py déjà déplacé ou absent"
mv remesh_environment.py experiments/ 2>/dev/null || echo "  remesh_environment.py déjà déplacé ou absent"
mv remesh_trainer.py experiments/ 2>/dev/null || echo "  remesh_trainer.py déjà déplacé ou absent"
mv smart_ppo_agent.py experiments/ 2>/dev/null || echo "  smart_ppo_agent.py déjà déplacé ou absent"
mv smart_ppo_agent_generic.py experiments/ 2>/dev/null || echo "  smart_ppo_agent_generic.py déjà déplacé ou absent"
mv test_remesh_env.py experiments/ 2>/dev/null || echo "  test_remesh_env.py déjà déplacé ou absent"

echo -e "${GREEN}✓${NC} Expérimentations déplacées"
echo ""

# Phase 5: Déplacer les images de résultats
echo -e "${BLUE}[Phase 5]${NC} Déplacement des visualisations d'exemples..."

mv *_result.png examples/visualizations/ 2>/dev/null || echo "  Visualisations déjà déplacées ou absentes"

echo -e "${GREEN}✓${NC} Visualisations déplacées"
echo ""

# Phase 6: Déplacer les scripts utilitaires
echo -e "${BLUE}[Phase 6]${NC} Déplacement des scripts utilitaires..."

mv verify_publication.py scripts/ 2>/dev/null || echo "  verify_publication.py déjà déplacé ou absent"
mv test_examples.sh scripts/ 2>/dev/null || echo "  test_examples.sh déjà déplacé ou absent"

echo -e "${GREEN}✓${NC} Scripts déplacés"
echo ""

# Phase 7: Gérer le README
echo -e "${BLUE}[Phase 7]${NC} Gestion du README..."

if [ -f "README_NEW.md" ] && [ -f "README.md" ]; then
    echo -e "${YELLOW}⚠${NC}  README.md et README_NEW.md existent tous les deux"
    mv README.md README_OLD.md
    echo -e "  → README.md sauvegardé en README_OLD.md"
    cp README_NEW.md README.md
    echo -e "  → README_NEW.md copié vers README.md"
fi

echo -e "${GREEN}✓${NC} README mis à jour"
echo ""

# Phase 8: Créer les README manquants
echo -e "${BLUE}[Phase 8]${NC} Création des README manquants..."

# Créer docs/README.md
cat > docs/README.md << 'EOF'
# SOFIA Documentation

This directory contains all project documentation.

## 📚 User Documentation

- **[CITATION.md](CITATION.md)** - How to cite SOFIA in your work
- **[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)** - Community guidelines
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - How to contribute to SOFIA
- **[PUBLICATION_GUIDE.md](PUBLICATION_GUIDE.md)** - Complete guide for publishing SOFIA
- **[QUICK_START.md](QUICK_START.md)** - Quick start guide for publication

## 🖼️ Images

The `images/` directory contains all documentation images and demos.

## 🛠️ Developer Documentation

See the `dev/` directory for:
- **[MOVING_MODULES.md](dev/MOVING_MODULES.md)** - Module organization
- **[REFACTORING_VISUAL.txt](dev/REFACTORING_VISUAL.txt)** - Refactoring notes
- **archive/** - Historical preparation documents

## 📖 Additional Documentation

- **Examples:** See `../examples/README.md`
- **Benchmarks:** See `../benchmarks/README.md`
- **Experiments:** See `../experiments/README.md`
EOF

echo -e "${GREEN}✓${NC} docs/README.md créé"

# Créer benchmarks/README.md
cat > benchmarks/README.md << 'EOF'
# SOFIA Benchmarks

This directory contains performance benchmarks for SOFIA's mesh operations.

## 📊 Benchmark Scripts

- **benchmark_boundary_loops.py** - Boundary loop performance
- **benchmark_comprehensive_validation.py** - Complete validation suite
- **benchmark_editor_incremental.py** - Incremental editor operations
- **benchmark_grid_optimization.py** - Grid-based optimizations
- **benchmark_incremental.py** - Incremental operations
- **benchmark_incremental_fair.py** - Fair comparison benchmarks
- **benchmark_numba.py** - Numba acceleration tests
- **benchmark_numba_comparison.py** - Python vs Numba comparison
- **benchmark_numba_direct.py** - Direct Numba integration
- **benchmark_real_world.py** - Real-world scenarios
- **benchmark_refinement_hotpaths.py** - Refinement hotpath analysis
- **benchmark_refinement_real_world.py** - Real-world refinement
- **benchmark_refinement_subprocess.py** - Subprocess-based refinement

## 📁 Results

Benchmark results are stored in `results/`:
- **batch_benchmark_results.json** - Batch operation results
- **phase2_results.json** - Phase 2 optimization results

## 🚀 Running Benchmarks

```bash
# Run a specific benchmark
python benchmark_<name>.py

# Run all benchmarks (takes time)
for bench in benchmark_*.py; do python "$bench"; done
```

## 📈 Interpreting Results

Results typically include:
- Execution time (seconds)
- Operations per second
- Memory usage
- Comparison with baseline

See individual benchmark files for detailed metrics.
EOF

echo -e "${GREEN}✓${NC} benchmarks/README.md créé"

# Créer experiments/README.md
cat > experiments/README.md << 'EOF'
# SOFIA Experiments

This directory contains experimental code and research projects.

## 🧪 Reinforcement Learning Experiments

### RL Environment
- **remesh_environment.py** - OpenAI Gym environment for mesh remeshing
- **test_remesh_env.py** - Tests for the RL environment

### RL Agents
- **rl-ym.py** - Main RL training script
- **remesh_trainer.py** - Training utilities
- **smart_ppo_agent.py** - PPO agent for remeshing
- **smart_ppo_agent_generic.py** - Generic PPO implementation

## 🎯 Purpose

These experiments explore:
- Automated mesh quality improvement using RL
- Learning optimal remeshing strategies
- Adaptive mesh refinement policies
- Quality-aware mesh operations

## ⚠️ Status

**Experimental Code** - Not production-ready

This code is for research and experimentation. It may:
- Have incomplete documentation
- Require additional dependencies
- Change without notice
- Not be fully tested

## 🚀 Usage

```bash
# Install RL dependencies (not in main requirements)
pip install gym stable-baselines3

# Run basic RL training
python rl-ym.py

# Test the environment
python test_remesh_env.py
```

## 📚 References

If you use this code in your research, please cite the main SOFIA project
and acknowledge that this is experimental work.
EOF

echo -e "${GREEN}✓${NC} experiments/README.md créé"
echo ""

# Statistiques finales
echo "═══════════════════════════════════════════════════════════════"
echo -e "${GREEN}✓ Réorganisation terminée!${NC}"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "📊 Nouvelle structure:"
echo "  • docs/            : Documentation complète"
echo "  • benchmarks/      : Tous les benchmarks"
echo "  • experiments/     : Code expérimental (RL)"
echo "  • examples/        : Exemples avec visualisations"
echo "  • scripts/         : Scripts utilitaires"
echo ""
echo "📁 Fichiers à la racine:"
ls -1 | grep -E '^[^.].*\.(md|py|toml|txt|ini|cfg)$' | wc -l | xargs echo "  •"
echo ""
echo "⚠️  Actions restantes:"
echo "  1. Vérifier que tout fonctionne: python scripts/verify_publication.py"
echo "  2. Tester les exemples: bash scripts/test_examples.sh"
echo "  3. Supprimer README_NEW.md et README_OLD.md si tout va bien"
echo "  4. Mettre à jour les chemins dans docs/PUBLICATION_GUIDE.md"
echo ""

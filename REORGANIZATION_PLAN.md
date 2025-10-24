# 📋 Plan de Réorganisation du Projet SOFIA

## 🎯 Objectif
Nettoyer et réorganiser le projet pour une structure professionnelle et maintenable.

## 📊 Analyse de la Situation Actuelle

### Problèmes Identifiés
1. **48 fichiers à la racine** (trop encombré)
2. **Documentation dispersée** (docs/ contient seulement des images)
3. **Doublons de README** (README.md vs README_NEW.md)
4. **Fichiers temporaires** (benchmarks, RL, résultats JSON)
5. **Images de résultats éparpillées** (PNG à la racine)

## 🗂️ Structure Proposée

```
sofia/
├── README.md                    # README principal (à mettre à jour)
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── setup.py
├── setup.cfg
├── pytest.ini
├── conftest.py
├── requirements.txt
├── requirements-dev.txt
├── Makefile
├── .gitignore
│
├── .github/
│   ├── workflows/
│   └── ISSUE_TEMPLATE/
│
├── docs/                        # 📖 TOUTE LA DOCUMENTATION
│   ├── README.md                # Index de la doc
│   ├── CITATION.md
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── PUBLICATION_GUIDE.md
│   ├── QUICK_START.md
│   ├── images/                  # Images de démo existantes
│   └── dev/                     # Documentation développeur
│       ├── MOVING_MODULES.md
│       ├── REFACTORING_VISUAL.txt
│       └── archive/
│           ├── PREPARATION_STATUS.md
│           ├── PUBLICATION_COMPLETE.md
│           └── SESSION_SUMMARY.md
│
├── examples/                    # ✅ Déjà bien organisé
│   ├── README.md
│   ├── *.py                     # 8 exemples
│   └── visualizations/          # NOUVEAU - images des résultats
│       └── *.png
│
├── benchmarks/                  # 🔧 NOUVEAU - tous les benchmarks
│   ├── README.md
│   ├── results/                 # NOUVEAU - résultats JSON
│   │   ├── batch_benchmark_results.json
│   │   └── phase2_results.json
│   └── *.py                     # Tous les benchmark_*.py
│
├── experiments/                 # 🧪 NOUVEAU - RL et expérimentations
│   ├── README.md
│   ├── rl-ym.py
│   ├── remesh_environment.py
│   ├── remesh_trainer.py
│   ├── smart_ppo_agent.py
│   ├── smart_ppo_agent_generic.py
│   └── test_remesh_env.py
│
├── configs/                     # ⚙️ Configurations
│   ├── greedy_cfg.json
│   └── patch_cfg.json
│
├── scripts/                     # 🛠️ Scripts utilitaires
│   ├── verify_publication.py   # DÉPLACÉ
│   └── test_examples.sh        # DÉPLACÉ
│
├── sofia/                       # ✅ Code source principal
│   ├── __init__.py
│   └── core/
│
├── tests/                       # ✅ Tests unitaires
│
├── demos/                       # ✅ Démos existantes
│
└── utilities/                   # ✅ Utilitaires existants
```

## 📝 Actions à Réaliser

### Phase 1: Préparation des Répertoires
- [ ] Créer `docs/images/`
- [ ] Créer `docs/dev/`
- [ ] Créer `docs/dev/archive/`
- [ ] Créer `benchmarks/`
- [ ] Créer `benchmarks/results/`
- [ ] Créer `experiments/`
- [ ] Créer `examples/visualizations/`

### Phase 2: Déplacer la Documentation
- [ ] Déplacer toutes les images de docs/ → docs/images/
- [ ] Déplacer CITATION.md → docs/
- [ ] Déplacer CODE_OF_CONDUCT.md → docs/
- [ ] Déplacer CONTRIBUTING.md → docs/
- [ ] Déplacer PUBLICATION_GUIDE.md → docs/
- [ ] Déplacer QUICK_START.md → docs/
- [ ] Déplacer MOVING_MODULES.md → docs/dev/
- [ ] Déplacer REFACTORING_VISUAL.txt → docs/dev/
- [ ] Déplacer fichiers de préparation → docs/dev/archive/

### Phase 3: Déplacer les Benchmarks
- [ ] Déplacer tous les benchmark_*.py → benchmarks/
- [ ] Déplacer batch_benchmark_results.json → benchmarks/results/
- [ ] Déplacer phase2_results.json → benchmarks/results/
- [ ] Créer benchmarks/README.md

### Phase 4: Déplacer les Expérimentations
- [ ] Déplacer rl-ym.py → experiments/
- [ ] Déplacer remesh_*.py → experiments/
- [ ] Déplacer smart_ppo_*.py → experiments/
- [ ] Déplacer test_remesh_env.py → experiments/
- [ ] Créer experiments/README.md

### Phase 5: Réorganiser les Images
- [ ] Déplacer *_result.png → examples/visualizations/
- [ ] Mettre à jour les chemins dans examples/*.py si nécessaire

### Phase 6: Déplacer les Scripts
- [ ] Déplacer verify_publication.py → scripts/
- [ ] Déplacer test_examples.sh → scripts/
- [ ] Mettre à jour les chemins dans PUBLICATION_GUIDE.md

### Phase 7: Nettoyer les Doublons
- [ ] Remplacer README.md par README_NEW.md
- [ ] Supprimer README_NEW.md
- [ ] Supprimer README_OLD.md (si existe)

### Phase 8: Créer les README manquants
- [ ] Créer docs/README.md (index de la documentation)
- [ ] Créer benchmarks/README.md
- [ ] Créer experiments/README.md

### Phase 9: Mise à jour des Références
- [ ] Mettre à jour les chemins dans PUBLICATION_GUIDE.md
- [ ] Mettre à jour les chemins dans QUICK_START.md
- [ ] Mettre à jour verify_publication.py pour les nouveaux chemins
- [ ] Mettre à jour test_examples.sh pour les nouveaux chemins

### Phase 10: Nettoyage Final
- [ ] Vérifier qu'il ne reste que les fichiers essentiels à la racine
- [ ] Tester que tous les exemples fonctionnent encore
- [ ] Tester verify_publication.py
- [ ] Mettre à jour .gitignore si nécessaire

## ✅ Fichiers qui DOIVENT Rester à la Racine

Fichiers essentiels uniquement:
- README.md (principal)
- LICENSE
- CITATION.cff
- pyproject.toml
- setup.py
- setup.cfg
- pytest.ini
- conftest.py
- requirements.txt
- requirements-dev.txt
- Makefile
- .gitignore

**Total: ~12 fichiers à la racine** (contre 48+ actuellement)

## 📊 Résultat Attendu

**Avant:** 48+ fichiers à la racine, documentation éparpillée
**Après:** 12 fichiers à la racine, structure claire et professionnelle

### Bénéfices
✅ Structure claire et navigable
✅ Documentation centralisée dans docs/
✅ Expérimentations séparées du code principal
✅ Benchmarks organisés avec résultats
✅ Visualisations groupées par type
✅ Plus facile à maintenir
✅ Plus professionnel pour la publication

## 🚀 Ordre d'Exécution Recommandé

1. Créer tous les répertoires d'abord
2. Déplacer les fichiers (pas de suppression pour l'instant)
3. Tester que tout fonctionne
4. Nettoyer les doublons
5. Mettre à jour les références
6. Vérification finale

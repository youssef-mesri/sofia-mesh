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

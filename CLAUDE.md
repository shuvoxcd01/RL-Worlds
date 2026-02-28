# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**RL-Worlds** is a Python package (`rl_worlds`) providing a collection of Gymnasium-compatible reinforcement learning environments. It is published to PyPI.

## Development Setup

Install the package in editable mode from the repo root:
```bash
pip install -e .
```

Dependencies: `gymnasium==1.0.0`, `numpy==1.24.4`, Python >= 3.8.

## Validating Environments

Use the Gymnasium env checker script to validate an environment conforms to the Gymnasium API:
```bash
python src/env_checker.py
```
Edit the script to switch which environment is checked (lines are commented/uncommented).

## Architecture

### Package structure

- `src/rl_worlds/__init__.py` — Registers all environments with Gymnasium via `gymnasium.envs.registration.register`. This is where environment IDs, entry points, and default kwargs are defined.
- `src/rl_worlds/envs/__init__.py` — Re-exports all environment classes.
- `src/rl_worlds/envs/` — One file per environment class.
- `src/rl_worlds/string_observation_space.py` — Custom `gymnasium.spaces.Space` subclass used by environments with string/letter-based observations (RandomWalk, TunnelWorld).

### Environment hierarchy

- `GridWorldEnv` (`grid_world.py`) — Base grid environment. Supports an optional `force_grid` (numpy array) and `force_direction` kwarg to apply wind-like forces. Action space: Discrete(4) (up/right/down/left). Observation: `Tuple` of two `Box` scalars (row, col).
- `WindyGridWorldEnv` (`windy_grid_world.py`) — Subclass of `GridWorldEnv`. Presets a wind force grid based on the Sutton & Barto windy gridworld example.
- `RandomWalkEnv` (`random_walk.py`) — MRP (actions are ignored; direction is random). Uses `StringObservationSpace` for letter states (A–E by default) or integer states. +1 reward at right terminal, 0 elsewhere.
- `ThousandStatesRandomWalkEnv` (`thousand_states_random_walk.py`) — Subclass of `RandomWalkEnv` with 1000 states, numeric observations, and multi-step jumps (1–100 steps per transition).
- `TunnelWorld` (`tunnel_world.py`) — 4-state linear environment (A–D). Action space: Discrete(2) (left/right). Supports both string and numeric observations via `numeric_observation` flag. Asymmetric reward structure.

### Registered environment IDs

| ID | Class | Notable defaults |
|----|-------|-----------------|
| `rl_worlds/RandomWalk-v0` | `RandomWalkEnv` | 5 states, string obs |
| `rl_worlds/ThousandStatesRandomWalk-v0` | `ThousandStatesRandomWalkEnv` | 1000 states, numeric obs |
| `rl_worlds/GridWorld-v0` | `GridWorldEnv` | 7×10 grid, max 50 steps |
| `rl_worlds/WindyGridWorld-v0` | `WindyGridWorldEnv` | 7×10 grid, max 50 steps |
| `rl_worlds/TunnelWorld-v0` | `TunnelWorld` | max 10 steps |

### Adding a new environment

1. Create `src/rl_worlds/envs/<env_name>.py` with a class extending `gym.Env`.
2. Export it from `src/rl_worlds/envs/__init__.py`.
3. Register it in `src/rl_worlds/__init__.py` with a unique ID.

## Versioning

Version is set in `pyproject.toml`. After bumping it, sync to `CITATION.cff`:
```bash
python scripts/sync_citation_version.py
```

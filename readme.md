# 🕷️ Marcuspider — Spider Solitaire Reinforcement Learning Agent

## Project Overview

**Marcuspider** is a reinforcement learning project that trains an AI agent to play  
**Spider Solitaire (single-suit variant)**.

The project implements a custom game environment that faithfully models Spider Solitaire rules
and trains an agent using **Maskable Proximal Policy Optimization (Maskable PPO)** with action masking,
ensuring that only legal moves are considered during learning and inference.

---

## Goals

- Build a **rule-correct and extensible** Spider Solitaire environment
- Train a stable and effective agent for **single-suit Spider Solitaire**
- Support:
  - Legal action masking
  - Automatic sequence completion detection (K → A)
  - Card flipping and dealing mechanics
- Provide a **live testing script** for evaluating model decisions in real games

---

## Core Approach

### Reinforcement Learning

- Algorithm: **Maskable PPO** (`sb3-contrib`)
- Action masking is used to prevent illegal moves and reduce exploration space

### Observation Space

- 10 tableau columns
- Up to 30 cards per column
- Each card is represented by:
  - Card value (`1–13`, or `-1` for face-down cards)
  - Face-up flag (`0/1`)
- Card suit is omitted because the environment targets the **single-suit** variant

### Action Space

- `0–99`: Move a valid descending face-up sequence from column `src` to column `dest`
- `100`: Deal one new card to each column

---

## Game Rules Modeled

- Single-suit Spider Solitaire
- A move is legal if:
  - Cards form a strictly descending sequence
  - All cards are face-up
  - All cards share the same suit (always true in single-suit mode)
- Automatic rule handling:
  - Flip the next card after a successful move
  - Remove completed sequences of **13 cards (K → A)**
  - Dealing is only allowed when all columns are non-empty

---

## Project Structure

Marcuspider/
├── logic.py # Core Spider Solitaire environment
├── train.py # RL training script (Maskable PPO)
├── verify_V3.py # Live testing & human-assisted verification script
├── verify_real_game.py # Experimental real-game testing script
├── testGPU.py # GPU availability check
├── models/ # Trained models (generated locally)
└── README.md


---

## About verify_V3.py

`verify_V3.py` is a **human-in-the-loop testing tool** designed to:

- Apply the AI agent’s decisions to a real Spider Solitaire game
- Allow manual input for revealed and dealt card values
- Keep the internal game state synchronized with real gameplay

> Note: This script is intended for **testing and verification only**  
> and does not affect the correctness of the training environment (`logic.py`).

---

## Current Status

- ✅ Single-suit Spider Solitaire environment implemented and verified
- ✅ Agent can be trained stably and produces reasonable strategies
- 🚧 Live verification script (`verify_V3.py`) is still under active refinement
- 🔄 Environment design supports future extensions

---

## Future Work

- Support for **two-suit and four-suit** Spider Solitaire
- Improved reward shaping focused on win rate
- Automated real-game state recognition (computer vision)
- Strategy analysis and visualization tools

---

## Notes

This project focuses on **environment design and rule-consistent reinforcement learning**
rather than minimal code size, prioritizing clarity, extensibility, and correctness.


# Quick Start Guide

Get AlphaEvo running in 5 minutes.

## Prerequisites

- Python 3.10+
- pip

## Installation

```bash
# From PyPI (when published)
pip install alphaevo

# From source
git clone https://github.com/ZhuLinsen/alphaevo.git
cd alphaevo
pip install -e .
```

## First-time Setup

```bash
alphaevo init
```

This creates `~/.alphaevo/config.yaml` and initializes the SQLite database.

## Run the Demo

```bash
# Synthetic data — no network, no API keys needed
alphaevo demo

# Real market data (requires network)
alphaevo demo --real
alphaevo demo --real --market cn   # A-share market
```

## Basic Commands

```bash
# List available strategies
alphaevo strategy list

# Run a single strategy backtest
alphaevo run trend_pullback_rebound_v1

# View the leaderboard
alphaevo leaderboard

# Show version
alphaevo version
```

## Next Steps

- [02 — Your First Strategy](02_first_strategy.md)
- [03 — Understanding the DSL](03_understand_dsl.md)
- [05 — Evolution Guide](05_evolution_guide.md)

---

*⚠️ AlphaEvo is a research tool, not investment advice.*

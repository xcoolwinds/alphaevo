<div align="center">

<img src="sources/alphaevo_terminal_v1_agent_evolution.svg" alt="AlphaEvo Logo" width="680">

# 🧬 AlphaEvo

**An Open-Source Self-Evolving Stock Strategy Research Agent**

*Backtest stock strategies and improve them with LLM-guided iterations*

[![GitHub stars](https://img.shields.io/github/stars/ZhuLinsen/alphaevo?style=social)](https://github.com/ZhuLinsen/alphaevo/stargazers)
[![CI](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml/badge.svg)](https://github.com/ZhuLinsen/alphaevo/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](Dockerfile)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

[**Overview**](#-overview) · [**Core Capabilities**](#-core-capabilities) · [**Quick Start**](#-quick-start) · [**Architecture**](#-architecture) · [**Validation**](#-real-world-validation-april-10-2026) · [**CLI**](#-cli-commands)

[English](README.md) | [中文](docs/README_CN.md)

</div>

---

## ✨ Overview

AlphaEvo backtests stock strategies on historical market data and iteratively improves them with LLM-guided reflection. It combines executable YAML DSL, anti-overfitting evaluation, research logs, and evolution trees into a single research workflow.

## 🧠 Core Capabilities

- **Backtesting and evaluation**: run strategies on real data with sampling, multi-metric scoring, and anti-overfitting checks.
- **LLM-guided strategy evolution**: diagnose failure modes, propose targeted mutations, and re-test whether the new version is actually better.
- **Traceable research workflow**: keep reports, LLM evidence, evolution trees, and trajectory exports for every iteration.

## 🚀 Quick Start

### Try it in 30 seconds from source (no API key needed!)

```bash
git clone https://github.com/ZhuLinsen/alphaevo.git
cd alphaevo
pip install -e .
alphaevo demo
```

<p align="center">
  <img src="docs/demo.gif" alt="AlphaEvo Demo" width="720">
</p>

This runs a complete self-evolution loop with synthetic data:

```
🔬 Evolution: 4 rounds of self-improvement

Round 1 │ v1 │ Win: 100%  Signals: 7   │ Score: 39.2%
  🔓 Lower volume threshold to capture more trades
  🔓 Lower relative strength threshold
Round 2 │ v2 │ Win: 86%   Signals: 21  │ Score: 44.0%  ↑ +4.8%
Round 3 │ v3 │ Win: 85%   Signals: 27  │ Score: 56.1%  ↑ +12.2%  🏆
Round 4 │ v4 │ Win: 85%   Signals: 27  │ Score: 55.2%  ↓ -1.0%

📈 Strategy improved from 39.2% → 56.1% (+16.9%)
```

### Choose the right path

| Goal | Command | Data | LLM |
|------|---------|------|-----|
| Fast first-run demo | `alphaevo demo` | Synthetic | No |
| Real market data smoke test | `alphaevo demo --real` | Live yfinance / akshare | No |
| Fuller real-data backtest | `alphaevo run ma_crossover_v1` | Live yfinance | No |
| Flagship research-agent path | `alphaevo evolve <id> --method llm --output reports/` | Real data | Yes |

Real-data commands need a data adapter extra: install `pip install -e ".[data-yfinance]"` for the default US workflow, `pip install -e ".[data-akshare]"` for A-share, or `pip install -e ".[data-full]"` for both.

`alphaevo demo --real` is the quickest real-data smoke test: it uses live market data, prints the per-round hypothesis diagnosis, and may stop early when real evidence does not support another mutation. If you want a stronger first backtest with more signals on the default `yfinance` adapter, start with `alphaevo run ma_crossover_v1`.

### Current release scope

- **Flagship LLM proof path**: `rsi_reversion_v1` and `ma_crossover_v1` on real data.
- **Most reliable strategy families right now**: trend + reversal strategies whose core signals come from OHLCV / benchmark context.
- **Experimental families**: event + rotation strategies still rely partly on proxy context for news / sector-flow semantics, so treat them as research previews rather than the main launch proof.

### Full setup (with LLM evolution)

```bash
# From the cloned repo root, add LLM + default real-data support
pip install -e ".[llm,data-yfinance]"

# Set your LLM API key
export ALPHAEVO_API_KEY=your_api_key
export ALPHAEVO_LLM_MODEL=gemini/gemini-2.0-flash  # or openai/gpt-4o, etc.

# Run a real-data research loop on the default yfinance adapter
alphaevo run ma_crossover_v1

# Run the flagship LLM research path
alphaevo evolve rsi_reversion_v1 --method llm --rounds 3 --output reports/rsi_evolve/
```

If you also want the built-in A-share workflow, install `pip install -e ".[llm,data-full]"` or add `pip install -e ".[data-akshare]"`, then use `alphaevo run trend_pullback_rebound_v1 --adapter akshare`.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    CLI (Typer + Rich)                     │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│              Orchestrator (Pipeline)                      │
│   generate → sample → backtest → evaluate → reflect      │
│                    → evolve → leaderboard                 │
└──┬────────┬────────┬────────┬────────┬────────┬─────────┘
   │        │        │        │        │        │
┌──▼──┐ ┌──▼──┐ ┌───▼──┐ ┌──▼──┐ ┌───▼──┐ ┌──▼────────┐
│Data │ │Stgy │ │Sample│ │Back │ │Eval  │ │Reflection │
│Layer│ │Layer│ │Layer │ │test │ │Layer │ │Layer      │
└─────┘ └─────┘ └──────┘ └─────┘ └──────┘ └───────────┘
```

### Six-Layer Architecture

| Layer | Purpose |
|-------|---------|
| **Data Layer** | Multi-source market data (yfinance, akshare, or daily_stock_analysis plugin) |
| **Strategy Layer** | Dual-representation: human-readable + executable YAML DSL |
| **Sampler Layer** | Smart sampling by market regime, style, and strategy scope |
| **Backtest Engine** | Signal-level simulation with proper slippage and fees |
| **Evaluator Layer** | Multi-dimensional metrics + anti-overfitting checks |
| **Reflection Layer** | LLM failure attribution + LLM-first evolution with optional param-search fallback |

## 🧬 How Self-Evolution Works

```
Round 1: rsi_reversion_v1 — live yfinance data, LLM-only evolve
         → 0 signals
         → Confidence: 8.1%  ❌ non-functional

         LLM diagnosis: "Entry conditions are contradictory and too strict."
         Changes: entry.logic and→or, RSI 30→35

Round 2: rsi_reversion_v2 — strategy becomes tradable
         → 522 signals
         → Win rate: 52.7%, Avg return: +0.96%
         → Confidence: 39.2%

         LLM diagnosis: "The OR logic over-corrected and now admits noisy entries."
         Changes: RSI 35→32, volume_ratio 1.3→1.15, stop_loss pct→atr

Round 3: rsi_reversion_v3 — champion
         → 498 signals
         → Win rate: 52.6%, Avg return: +1.22%
         → Confidence: 56.3%  (+48.2pp from start)
```

> These are **real results** from an April 10, 2026 run using live yfinance data and a configured external LLM provider with `--method llm`. Not synthetic, not heuristic fallback.

## 📋 Strategy DSL Example

Strategies are stored as human-readable YAML, so the LLM can explain and mutate them without hiding the logic in code.

```yaml
meta:
  id: trend_pullback_rebound_v3
  name: 强趋势回踩放量反包
  version: 3
  category: trend

entry:
  logic: and
  conditions:
    - indicator: relative_strength_20d
      op: ">"
      value: 0.12

exit:
  stop_loss:
    type: atr
    atr_period: 21
    multiplier: 2.0
  take_profit:
    type: rr
    value: 2.0

params:
  tunable:
    - target: entry.conditions[indicator=relative_strength_20d].value
      range: [0.05, 0.20]
      step: 0.01
```

See [technical design](docs/technical_design.md) for the full DSL.

## 🔬 Real-World Validation (April 10, 2026)

We validate the system on **live yfinance data** with real `--method llm` runs, and we intentionally show both success and honest failure.

| Case | Start | Best Outcome | Why It Matters |
|------|-------|--------------|----------------|
| `rsi_reversion_v1` | `8.1%` | `56.3%` champion | LLM turned a zero-signal strategy into a tradable one in 3 rounds |
| `ma_crossover_v1` | `24.2%` | `24.2%` champion | LLM proposed changes, but anti-overfit rejected weak generalization |
| `sector_rotation_leader_v1` smoke | `11.3%` | `12.3%` tested | Candidate improved in-sample, but `train_val_gap=18.9%` blocked promotion |

This is part of the product story, not a footnote: AlphaEvo should be trusted more when it stops honestly than when it invents a prettier curve.

Real factor discovery is also live-tested:

- `alphaevo factor discover AAPL` proposed `3` factors, passed `3` through sandboxing, validated `2`, and registered `2`.
- Walkthrough: [Factor Discovery Walkthrough](docs/cookbook/07_factor_discovery_walkthrough.md)

Each real run can export:

- `<strategy_id>_research_report.md`
- `<strategy_id>_llm_evidence.md`
- `<strategy_id>_research_log.md`
- `trajectory/*.jsonl|json`

### Reproducible Benchmark Path

For a fixed-input benchmark suite, use:

```bash
python scripts/experiments/run_repro_benchmark.py \
  --adapter yfinance \
  --method llm \
  --rounds 3 \
  --output results/repro-benchmark/

python scripts/validate_real_data.py \
  --adapter yfinance \
  --days 365 \
  --output results/real-validation/
```

See the detailed write-up: [April 10, 2026 real LLM validation](docs/reports/2026-04-10-real-llm-validation.md)

Current limits:

- live external LLM providers can still introduce latency spikes and timeout-driven fallback
- some strategies, like `ma_crossover_v1`, show that a valid LLM diagnosis does not automatically survive anti-overfit checks
- discovered factors are research artifacts first and should still be reviewed before any production-style use

## 🧠 Trajectory Data Flywheel

Every real evolution run can export more than scores:

- `trajectory.jsonl` captures each round as `(state -> diagnosis -> hypothesis -> change -> outcome)`
- `sharegpt.jsonl` reformats the same run into SFT-style conversations
- `preference.jsonl` stores improved vs. non-improved steps for preference learning

That means AlphaEvo is not only a strategy optimizer. It is also a data engine
for training better strategy-research agents over time.

See: [Trajectory Data Flywheel](docs/trajectory_data_flywheel.md)

## 🛡️ Anti-Overfitting

AlphaEvo takes overfitting seriously:

- **Time Separation**: Train / Validation / Test periods strictly separated
- **Walk-Forward**: Rolling 12-month train → 1-month test windows
- **Complexity Penalty**: More conditions = lower score
- **Stability Check**: Performance must be consistent across years/sectors
- **Minimum Signals**: Strategies with < 30 signals get reliability discount
- **Parameter Sensitivity**: ±10% perturbation test, >30% decay = warning

## 🔌 Compatibility

### Standalone Mode
Works out of the box with `yfinance` or `akshare` for data.

### Plugin Mode (daily_stock_analysis)
Seamlessly integrates with daily_stock_analysis for multi-source data with automatic fallback.

```python
from alphaevo.data.adapters.dsa import DSAAdapter
data_manager = DataManager([DSAAdapter(dsa_path="/path/to/dsa")])
```

## 📊 CLI Commands

```bash
alphaevo demo                     # 🔥 Try instantly (no setup needed)
alphaevo demo --real              # Real data demo without API key
alphaevo run <id>                 # Full research loop
alphaevo evolve <id> --method llm --rounds 3 --output reports/  # LLM-first evolution
alphaevo factor discover <symbol> # LLM-driven factor discovery
alphaevo leaderboard              # Strategy rankings
alphaevo tree <id>                # Evolution tree visualization
```

For the full command surface, run `alphaevo --help`.

## 📚 Research Inspirations

- **FunSearch (Nature 2024)** — island-style parallel search and branch competition
- **OPRO (DeepMind 2023)** — optimizer-style prompt refinement with trajectory history
- **Voyager (2023)** — reusable skill/pattern libraries and long-horizon memory

AlphaEvo adapts these ideas to quantitative strategy research rather than
general coding or benchmark optimization.

## 🗺️ Roadmap

- [x] Phase 1: Strategy Research Loop (MVP) — backtest engine, indicators, evaluator
- [x] Phase 2: Self-Evolution Pipeline — LLM reflection, mutation, multi-round improvement
- [x] Phase 3: CLI & Orchestration — full command suite, strategy store, leaderboard
- [x] Phase 4: Open-Source Polish — CI/CD, docs, English templates, CHANGELOG
- [ ] Phase 5: Market Regime Adaptive Gating — environment detection, strategy routing
- [ ] Phase 6: Web UI Dashboard — visualization, interactive strategy exploration

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Especially looking for:**
- New strategy templates
- Data source adapters
- Evaluation metrics
- UI/visualization improvements

## ☕ Support & Contact

If this project helps your research, consider giving it a ⭐ and sharing it!

<table>
  <tr>
    <td align="center" width="220">
      <a href="https://www.xiaohongshu.com/user/profile/61594417000000000201fa68" target="_blank">
        <img src="./sources/xiaohongshu.png" width="160" alt="小红书"><br>
        <b>Xiaohongshu 📱</b>
      </a><br>
      <sub>Follow for quant strategy research updates</sub>
    </td>
    <td valign="top" style="padding-left: 24px">
      <b>📬 Contact & Collaboration</b><br><br>
      🐛 &nbsp;<a href="https://github.com/ZhuLinsen/alphaevo/issues">Submit an Issue</a> — Bug reports / feature requests<br>
      📧 &nbsp;<a href="mailto:zhuls345@gmail.com">zhuls345@gmail.com</a> — Business inquiries<br>
      🔗 &nbsp;<a href="https://github.com/ZhuLinsen/daily_stock_analysis">daily_stock_analysis</a> — Sister project, AI-powered daily stock analysis
    </td>
  </tr>
</table>

## ⚠️ Disclaimer

This project is for **educational and research purposes only**. It does not constitute investment advice. The authors are not responsible for any financial losses incurred from using this software. Always do your own research and consult qualified financial advisors before making investment decisions.

Past strategy performance does not guarantee future results. All backtesting results are simulated and may not reflect real market conditions.

## 📄 License

Apache-2.0 License — see [LICENSE](LICENSE) for details.

If you use or build upon this project, a credit with a link back to this repository is appreciated.

---

<div align="center">


</div>

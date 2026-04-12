# Factor Discovery Walkthrough

AlphaEvo can do more than mutate thresholds. It can also propose brand-new
factors, sandbox them, validate them statistically, and register the ones that
survive.

This walkthrough shows the intended end-to-end path.

## When To Use It

Use factor discovery when:

- a strategy looks directionally interesting but lacks a strong ranking signal
- simple parameter tuning is no longer enough
- you want AlphaEvo to search for a new indicator hypothesis instead of only adjusting existing ones

## Command

```bash
alphaevo factor discover AAPL
```

The command will:

1. ask the LLM for candidate factor hypotheses
2. generate executable factor code inside the sandbox
3. reject unsafe or invalid code
4. evaluate the surviving factors on historical data
5. register only the factors that pass validation

## What A Good Run Looks Like

Real April 10, 2026 `AAPL` validation:

| Symbol | Proposed | Sandbox Passed | Validation Passed | Registered |
|--------|----------|----------------|-------------------|------------|
| `AAPL` | 3 | 3 | 2 | 2 |

Registered factors from that run:

- `volatility_compressed_breakout_quality`
- `volume_confirmed_reversal`

## What Failure Looks Like

Factor discovery should also reject bad ideas.

A same-day `MSFT` rerun showed why this matters:

- after enforcing `expected_direction`, a negative-direction mismatch was rejected
- only `volatility_compression_breakout_score` remained eligible for registration

That is the desired behavior. AlphaEvo should stop weak factors before they
become reusable building blocks.

## Output Artifacts

Factor discovery can export:

- `factor_report.md`
- factor code and registration metadata in the local store

The Markdown report is the easiest artifact to share because it preserves:

- the original factor hypothesis
- expected direction
- sandbox result
- validation metrics
- final registration decision

## How To Position It

The interesting story is not "LLM wrote a factor."

The stronger story is:

- AlphaEvo proposed a factor hypothesis
- executed it safely in a sandbox
- tested it statistically
- refused the ones that did not survive validation

That is a research workflow, not a code-generation trick.

## Related Paths

- Real data smoke path: [06_real_data_walkthrough.md](06_real_data_walkthrough.md)
- Trajectory exports: [../trajectory_data_flywheel.md](../trajectory_data_flywheel.md)

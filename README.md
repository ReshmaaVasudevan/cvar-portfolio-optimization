# CVaR Portfolio Optimization

Convex optimization for portfolio risk minimization using 
Conditional Value-at-Risk (CVaR) as the risk measure. Implements 
the full Rockafellar-Uryasev (2000) formulation from scratch 
without any external solver.

## Overview
Optimizes a 5-asset US equity portfolio (AAPL, MSFT, GOOGL, AMZN, 
TSLA) to minimize tail risk using two approaches — a hand-coded 
gradient descent algorithm and an exact LP solver — and compares 
their results.

## Key Innovation
The VaR threshold ζ is treated as a **joint decision variable** 
optimized alongside portfolio weights w via gradient descent, 
rather than computed via np.percentile. This fully implements 
the Rockafellar-Uryasev formulation from scratch.

∂L/∂ζ = 1 − |tail| / ((1−α)S)

At optimum, ∂L/∂ζ = 0 when |tail| = (1−α)S — exactly the 
definition of VaR — so gradient descent naturally recovers VaR 
without any statistical library calls.

## Methods Compared
| Method | Approach | Optimality |
|--------|----------|------------|
| Joint PGD (from scratch) | Gradient descent + simplex projection | Approximate (within 0.001 of LP) |
| CVXPY LP Solver | Exact linear programme | Global optimum |

## Results
| Portfolio | Ann. Return | Ann. Vol | CVaR (95%) | Sharpe |
|-----------|-------------|----------|------------|--------|
| Equal-Weight | ~34% | ~33.5% | 0.0453 | 1.015 |
| Joint PGD | ~25.7% | ~30.8% | 0.0417 | 0.833 |
| CVXPY LP | ~25.2% | ~30.4% | 0.0409 | 0.830 |

- CVaR gap between PGD and CVXPY: **< 0.001** across all 60 
  tested configurations
- **18–22% CVaR reduction** over equal-weight baseline at α = 0.95
- PGD converged in **300–1,500 iterations** across all settings

## Efficient Frontier
Traced across 20 return targets × 3 confidence levels 
(α = 0.90, 0.95, 0.99), confirming the convex structure of the 
CVaR-return trade-off.

## Tech Stack
- Python, NumPy (PGD solver — no external solver dependency)
- CVXPY + SCS (exact LP benchmark)
- yfinance (historical data: 2020–2024)
- Matplotlib, Pandas

## How to Run
```bash
pip install numpy cvxpy yfinance matplotlib pandas
jupyter notebook CVaR_Portfolio_Allocation.ipynb
```

## Dataset
- Daily adjusted closing prices for AAPL, MSFT, GOOGL, AMZN, TSLA
- Source: Yahoo Finance via yfinance (Jan 2020 – Jan 2024)
- 1,005 historical scenarios + 1,000 Monte Carlo simulations 
  = 2,005 total scenarios

## References
- Rockafellar & Uryasev (2000). Optimization of Conditional 
  Value-at-Risk. Journal of Risk.
- Duchi et al. (2008). Efficient Projections onto the L1-Ball. 
  ICML.

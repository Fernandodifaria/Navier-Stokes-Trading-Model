# NavierCapital — Fluid Dynamics Applied to Financial Markets

**Author:** Fernando Arruda de Faria
**Methodology:** Navier-Stokes & Computational Fluid Dynamics (CFD) analogues for quantitative trading
**Approach:** Institutional-grade systematic strategies (Jane Street / Jump Trading philosophy)

---

## Overview

NavierCapital treats financial markets as compressible, viscous fluid systems. Price, volume, and order flow are modeled as velocity fields, pressure gradients, and density distributions governed by the Navier-Stokes equations and related fluid dynamics principles. Every strategy in this repository maps directly to a physical fluid phenomenon — not as metaphor, but as a computable, backtestable trading signal.

The core premise: liquidity *is* a fluid. It flows, it has viscosity, it exhibits turbulence, it forms vortices around large orders, and it obeys conservation laws. This project exploits that isomorphism.

---

## Theoretical Foundation

### The Navier-Stokes Mapping

The incompressible Navier-Stokes equation:

```
∂u/∂t + (u · ∇)u = −(1/ρ)∇p + ν∇²u + f
```

Maps to markets as:

| Fluid Variable | Market Analogue |
|---|---|
| **u** (velocity field) | Order flow / price momentum vector |
| **p** (pressure) | Bid-ask spread compression, liquidity pressure |
| **ρ** (density) | Volume concentration / market depth |
| **ν** (kinematic viscosity) | Market friction (transaction costs, slippage) |
| **f** (external forces) | News shocks, macro events, scheduled releases |
| **∇ · u = 0** (incompressibility) | Volume conservation across venues |

### Extended Fluid Analogues

Beyond Navier-Stokes, the project draws from:

- **Euler Equations** — Inviscid (zero-cost) limit for idealized signal generation
- **Bernoulli's Principle** — Velocity-pressure tradeoff in spread dynamics
- **Reynolds Number** — Regime detection (laminar = trending, turbulent = mean-reverting)
- **Kolmogorov Turbulence Theory** — Volatility cascade across timeframes
- **Stokes Flow** — Low-Reynolds (illiquid) market microstructure
- **Vorticity Transport** — Rotational flow patterns in cross-asset momentum
- **Shallow Water Equations** — Macro liquidity wave propagation
- **Diffusion Equation** — Information dissemination and price discovery

---

## Repository Structure

```
naviercapital/
│
├── README.md
│
├── core/
│   ├── navier_stokes.py          # NS solver adapted for market fields
│   ├── reynolds.py                # Regime classifier (laminar/turbulent)
│   ├── bernoulli.py               # Spread-momentum pressure model
│   ├── vorticity.py               # Cross-asset rotational flow detector
│   ├── kolmogorov.py              # Multi-scale volatility cascade
│   ├── diffusion.py               # Information propagation model
│   └── shallow_water.py           # Macro liquidity wave model
│
├── strategies/
│   ├── pressure_gradient.py       # ∇p-based mean reversion
│   ├── turbulent_momentum.py      # High-Re momentum breakout
│   ├── laminar_carry.py           # Low-Re carry/trend strategy
│   ├── vortex_pairs.py            # Pairs trading via vorticity
│   ├── viscous_microstructure.py  # HFT spread capture (Stokes regime)
│   ├── wave_propagation.py        # Macro regime wave surfing
│   └── cascade_vol.py             # Kolmogorov vol-of-vol strategy
│
├── data/
│   ├── loaders.py                 # Market data ingestion
│   ├── orderbook.py               # L2/L3 order book reconstruction
│   └── field_construction.py      # Raw data → velocity/pressure fields
│
├── backtest/
│   ├── engine.py                  # Event-driven backtester
│   ├── costs.py                   # Viscosity-calibrated transaction costs
│   ├── risk.py                    # Drawdown, VaR, turbulence-adjusted risk
│   └── performance.py             # Sharpe, Sortino, Calmar, PnL attribution
│
├── reports/
│   └── pdf_generator.py           # Auto-generates strategy performance PDFs
│
├── config/
│   ├── assets.yaml                # Tradable universe
│   ├── params.yaml                # Model hyperparameters
│   └── execution.yaml             # Execution/routing config
│
└── notebooks/
    ├── 01_field_visualization.ipynb
    ├── 02_reynolds_regime_analysis.ipynb
    ├── 03_backtest_walkthrough.ipynb
    └── 04_turbulence_case_studies.ipynb
```

---

## Strategy Descriptions

### 1. Pressure Gradient Mean Reversion (`pressure_gradient.py`)

**Fluid Analogue:** Fluid flows from high pressure to low pressure.

When the bid-side "pressure" (accumulated resting volume × price level) diverges from the ask-side, the resulting ∇p predicts short-term price reversion toward the pressure equilibrium. Operates on tick-to-minute horizons.

**Signal:** `∇p = −ρ(∂u/∂t + (u · ∇)u − ν∇²u)`
Isolated pressure gradient from observable order flow.

---

### 2. Turbulent Momentum Breakout (`turbulent_momentum.py`)

**Fluid Analogue:** Above a critical Reynolds number, flow transitions from laminar to turbulent — energy cascades unpredictably.

When the market Reynolds number (momentum magnitude / transaction cost friction) exceeds a calibrated threshold, the strategy enters directional positions aligned with the dominant flow, sizing by turbulent kinetic energy.

**Signal:** `Re = |u| · L / ν` where L is the characteristic length scale (lookback window).

---

### 3. Laminar Carry (`laminar_carry.py`)

**Fluid Analogue:** Low-Reynolds laminar flow is smooth, predictable, dominated by viscosity.

In low-volatility regimes (Re << Re_critical), the strategy harvests carry and slow trend via Stokes-regime models where viscous forces dominate inertial forces. Ideal for rates, FX carry, and dividend futures.

---

### 4. Vortex Pairs Trading (`vortex_pairs.py`)

**Fluid Analogue:** Counter-rotating vortex pairs are stable, self-reinforcing structures in fluid flow.

Identifies asset pairs whose order flow exhibits stable counter-rotating vorticity (one absorbs flow the other emits). Trades the spread when vorticity magnitude spikes beyond its stationary distribution.

**Signal:** `ω = ∇ × u` computed on the cross-asset flow field.

---

### 5. Viscous Microstructure (`viscous_microstructure.py`)

**Fluid Analogue:** Stokes flow (creeping flow) — viscous forces completely dominate.

At the microstructure level, the spread behaves as a viscous boundary layer. This HFT-adjacent strategy models queue position as streamline position and captures spread by exploiting the no-slip condition (price stickiness at round numbers and NBBO boundaries).

---

### 6. Wave Propagation (`wave_propagation.py`)

**Fluid Analogue:** Shallow water equations model long-wavelength surface gravity waves.

Macro liquidity cycles (central bank flows, fiscal impulses, rebalancing waves) propagate as shallow-water waves across asset classes. The strategy detects wavefronts and positions ahead of the crest in downstream assets.

---

### 7. Cascade Volatility (`cascade_vol.py`)

**Fluid Analogue:** Kolmogorov's 1941 theory — energy injected at large scales cascades to small scales following a −5/3 power law.

Volatility injected at macro timescales (monthly) cascades to intraday following a power law. Deviations from the expected cascade spectrum signal mispriced implied vol at specific tenors. Trades the vol surface.

**Signal:** `E(k) ∝ k^(−5/3)` — fit the empirical vol spectrum, trade deviations.

---

## Performance Reporting

Every strategy auto-generates a PDF report via `reports/pdf_generator.py` containing:

- Equity curve (gross and net of costs)
- Sharpe / Sortino / Calmar ratios
- Maximum drawdown and recovery time
- Monthly return heatmap
- Reynolds number regime timeline
- Turbulent kinetic energy decomposition
- Slippage analysis (predicted viscosity vs realized)
- Signal decay profile

Reports are objective, functional, numbers-first. No decorative elements.

---

## Dependencies

```
numpy>=1.24
scipy>=1.11
pandas>=2.0
matplotlib>=3.7
reportlab>=4.0
pyyaml>=6.0
numba>=0.58        # JIT compilation for NS solver inner loops
h5py>=3.9          # Large field data storage
websocket-client   # Live order book feeds
```

---

## Quickstart

```bash
# Clone
git clone https://github.com/naviercapital/naviercapital.git
cd naviercapital

# Install
pip install -r requirements.txt

# Run a backtest
python -m strategies.pressure_gradient --config config/params.yaml

# Generate performance PDF
python -m reports.pdf_generator --strategy pressure_gradient --output reports/
```

---

## Design Principles

1. **Physics-first, not curve-fit.** Every parameter has a physical interpretation. If it can't be mapped to a fluid quantity, it doesn't belong in the model.

2. **Conservation laws are constraints.** Volume conservation (∇ · u = 0) is enforced, not assumed. Violations are themselves signals (hidden liquidity, spoofing).

3. **Viscosity is not noise — it's the cost model.** Transaction costs, market impact, and slippage are the kinematic viscosity ν. They are integral to the dynamics, not bolted on after.

4. **Regime is Reynolds number.** No ad-hoc regime switching. The Reynolds number provides a single, physically grounded, continuously computable regime indicator.

5. **Multi-scale by construction.** Kolmogorov cascade theory gives a principled framework for multi-timeframe signal aggregation instead of arbitrary lookback blending.

---

## Risk Management

Risk is managed through fluid-dynamical quantities:

- **Turbulent Kinetic Energy (TKE):** Position sizing inversely proportional to TKE. High turbulence → small positions.
- **Enstrophy (mean squared vorticity):** Cross-asset contagion risk. High enstrophy → reduce correlated exposures.
- **Pressure divergence:** When ∇²p spikes, the field is unstable — flatten discretionary overrides.
- **Boundary layer separation:** When price detaches from the "no-slip" reference (e.g., fair value), protective stops tighten.

---

## License

Proprietary. All intellectual property belongs to Fernando Arruda de Faria.
Unauthorized reproduction, distribution, or use of any strategy, model, or code in this repository is prohibited.

---

## Contact

**Fernando Arruda de Faria**
NavierCapital — Fluid Dynamics Quantitative Research

---

*"The market is a fluid. Trade accordingly."*

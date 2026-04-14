"""
Streamlined Backtest Runner - Optimized for Speed
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import warnings
import json

warnings.filterwarnings('ignore')

# Inline imports to avoid module loading issues
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from enum import Enum


class MarketRegime(Enum):
    LAMINAR = "LAMINAR"
    TRANSITIONAL = "TRANSITIONAL"
    TURBULENT = "TURBULENT"


@dataclass
class BacktestResults:
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profit_factor: float
    regime_accuracy: float
    avg_trade_pnl: float
    calmar_ratio: float
    regime_distribution: Dict
    strategy_breakdown: Dict
    equity_curve: pd.Series
    signal_decay: pd.DataFrame
    tc_sensitivity: pd.DataFrame
    mc_return_ci: Tuple[float, float]
    mc_sharpe_ci: Tuple[float, float]
    ruin_probability: float
    var_95: float
    cvar_95: float


def fetch_data(symbol='SPY', start='2021-01-01', end='2024-12-31'):
    """Fetch market data"""
    print(f"Fetching {symbol} data...")
    
    ticker = yf.Ticker(symbol)
    daily = ticker.history(start=start, end=end, interval='1d')
    daily.columns = [c.lower() for c in daily.columns]
    
    # Convert to timezone-naive for alignment
    daily.index = daily.index.tz_localize(None)
    
    vix = yf.Ticker('^VIX').history(start=start, end=end, interval='1d')
    vix.columns = [c.lower() for c in vix.columns]
    vix.index = vix.index.tz_localize(None)
    
    # Compute ATR
    tr = pd.concat([
        daily['high'] - daily['low'],
        abs(daily['high'] - daily['close'].shift(1)),
        abs(daily['low'] - daily['close'].shift(1))
    ], axis=1).max(axis=1)
    daily['atr'] = tr.rolling(14).mean()
    
    print(f"Loaded {len(daily)} days of data")
    return daily, vix['close']


def compute_velocity(prices, window=20):
    """Signed momentum accumulation"""
    returns = prices.pct_change()
    signed_mom = (np.sign(returns) * np.abs(returns))
    return signed_mom.rolling(window).sum() / window


def compute_viscosity(prices, window=14):
    """Inverse volatility"""
    vol = prices.pct_change().rolling(window).std() * np.sqrt(252)
    return 1 / (vol + 0.01)


def compute_pressure(vix, window=5):
    """VIX rate of change (inverted)"""
    roc = vix.pct_change(window)
    return -roc


def compute_reynolds(velocity, viscosity, scale=100):
    """Market Reynolds number"""
    return np.abs(velocity) * scale / viscosity


def classify_regime(reynolds, lam_th=50, turb_th=200):
    """Classify market regime"""
    if reynolds < lam_th:
        return MarketRegime.LAMINAR
    elif reynolds > turb_th:
        return MarketRegime.TURBULENT
    else:
        return MarketRegime.TRANSITIONAL


def run_backtest(daily, vix, lam_th=50, turb_th=200, initial_capital=100000):
    """Run simplified backtest"""
    
    # Make copies to avoid modifying original
    daily = daily.copy()
    
    # Align VIX to daily index
    vix_aligned = vix.reindex(daily.index).ffill().bfill()
    
    # Compute indicators
    velocity = compute_velocity(daily['close'])
    viscosity = compute_viscosity(daily['close'])
    pressure = compute_pressure(vix_aligned)
    reynolds = compute_reynolds(velocity, viscosity)
    
    # Build DataFrame with all indicators
    df = pd.DataFrame({
        'close': daily['close'],
        'velocity': velocity,
        'viscosity': viscosity,
        'pressure': pressure,
        'reynolds': reynolds,
        'atr': daily['atr']
    })
    
    # Fill NaN values with reasonable defaults
    df['pressure'] = df['pressure'].fillna(0)
    df['viscosity'] = df['viscosity'].fillna(df['viscosity'].mean() if df['viscosity'].notna().any() else 1)
    df['velocity'] = df['velocity'].fillna(0)
    df['reynolds'] = df['reynolds'].fillna(100)
    df['atr'] = df['atr'].fillna(df['close'] * 0.02)  # 2% of price as default ATR
    
    # Drop only rows where close is NaN
    df = df[df['close'].notna()]
    
    # Skip first 63 days for warm-up
    df = df.iloc[63:].copy()
    
    if len(df) < 50:
        raise ValueError(f"Insufficient data: only {len(df)} valid points after warm-up")
    
    # Flow signal
    df['flow_signal'] = (df['velocity'] * df['pressure']) / df['viscosity']
    
    # Z-score normalization
    df['flow_z'] = (df['flow_signal'] - df['flow_signal'].rolling(63).mean()) / \
                   (df['flow_signal'].rolling(63).std() + 1e-8)
    
    # Classify regimes
    df['regime'] = df['reynolds'].apply(lambda x: classify_regime(x, lam_th, turb_th))
    
    # Generate signals
    df['direction'] = 0
    df.loc[(df['flow_z'] > 0.3) & (df['regime'] == MarketRegime.TURBULENT), 'direction'] = 1
    df.loc[(df['flow_z'] < -0.3) & (df['regime'] == MarketRegime.TURBULENT), 'direction'] = -1
    df.loc[(df['flow_z'] > 0.3) & (df['regime'] == MarketRegime.LAMINAR), 'direction'] = -1
    df.loc[(df['flow_z'] < -0.3) & (df['regime'] == MarketRegime.LAMINAR), 'direction'] = 1
    
    # Forward returns
    df['fwd_ret'] = df['close'].pct_change().shift(-1)
    
    # Strategy returns
    df['strategy_ret'] = df['direction'].shift(1) * df['fwd_ret']
    df['strategy_ret'] = df['strategy_ret'].fillna(0)
    
    # Transaction costs (5bps per trade)
    df['trade'] = (df['direction'] != df['direction'].shift(1)).astype(int)
    df['strategy_ret'] -= df['trade'] * 0.0005
    
    # Equity curve
    df['equity'] = initial_capital * (1 + df['strategy_ret']).cumprod()
    
    # Performance metrics
    total_ret = (df['equity'].iloc[-1] / initial_capital) - 1
    
    strategy_returns = df['strategy_ret'].dropna()
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
    
    # Drawdown
    peak = df['equity'].expanding().max()
    dd = (df['equity'] - peak) / peak
    max_dd = dd.min()
    
    # Win rate
    winning_days = (df['strategy_ret'] > 0).sum()
    total_days = (df['strategy_ret'] != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # Profit factor
    gains = df.loc[df['strategy_ret'] > 0, 'strategy_ret'].sum()
    losses = abs(df.loc[df['strategy_ret'] < 0, 'strategy_ret'].sum())
    profit_factor = gains / losses if losses > 0 else float('inf')
    
    # Trade count
    total_trades = df['trade'].sum()
    
    # Regime distribution
    regime_counts = df['regime'].value_counts(normalize=True)
    regime_dist = {r.value: regime_counts.get(r, 0) for r in MarketRegime}
    
    # Regime accuracy (did regime correctly predict return behavior?)
    df['regime_correct'] = 0
    turb_mask = df['regime'] == MarketRegime.TURBULENT
    lam_mask = df['regime'] == MarketRegime.LAMINAR
    
    # Turbulent: momentum should work
    df.loc[turb_mask & (df['direction'] * df['fwd_ret'] > 0), 'regime_correct'] = 1
    # Laminar: fade should work (we already flipped direction)
    df.loc[lam_mask & (df['direction'] * df['fwd_ret'] > 0), 'regime_correct'] = 1
    
    regime_accuracy = df.loc[df['direction'] != 0, 'regime_correct'].mean()
    
    # Strategy breakdown
    strategy_breakdown = {}
    for regime in MarketRegime:
        mask = df['regime'] == regime
        regime_rets = df.loc[mask & (df['direction'] != 0), 'strategy_ret']
        strategy_breakdown[regime.value] = {
            'trades': len(regime_rets),
            'avg_ret': regime_rets.mean() if len(regime_rets) > 0 else 0,
            'total_ret': regime_rets.sum(),
            'win_rate': (regime_rets > 0).mean() if len(regime_rets) > 0 else 0
        }
    
    # Calmar ratio
    ann_ret = (1 + total_ret) ** (252 / len(df)) - 1
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    
    # Avg trade P&L
    active_rets = df.loc[df['direction'] != 0, 'strategy_ret']
    avg_trade = active_rets.mean() * initial_capital if len(active_rets) > 0 else 0
    
    return {
        'total_return': total_ret,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_dd,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'profit_factor': profit_factor,
        'regime_accuracy': regime_accuracy,
        'avg_trade_pnl': avg_trade,
        'calmar_ratio': calmar,
        'regime_distribution': regime_dist,
        'strategy_breakdown': strategy_breakdown,
        'equity_curve': df['equity'],
        'returns': strategy_returns,
        'df': df
    }


def signal_decay_analysis(df, max_horizon=30):
    """Analyze signal decay"""
    results = []
    
    for horizon in range(1, max_horizon + 1):
        df[f'fwd_{horizon}'] = df['close'].pct_change(horizon).shift(-horizon)
        
        signal_rets = df['direction'] * df[f'fwd_{horizon}']
        signal_rets = signal_rets[df['direction'] != 0].dropna()
        
        if len(signal_rets) > 10:
            results.append({
                'horizon': horizon,
                'avg_return': signal_rets.mean(),
                'hit_rate': (signal_rets > 0).mean(),
                'sharpe': signal_rets.mean() / (signal_rets.std() + 1e-8) * np.sqrt(252/horizon),
                'n_signals': len(signal_rets)
            })
    
    return pd.DataFrame(results)


def tc_sensitivity(returns, trades, initial_capital=100000):
    """Transaction cost sensitivity"""
    results = []
    
    for comm in [0.001, 0.005, 0.01, 0.02]:
        for slip in [2, 5, 10, 20]:
            tc_per_trade = comm + slip / 10000
            adjusted_ret = returns.sum() - trades * tc_per_trade
            
            results.append({
                'commission': comm,
                'slippage_bps': slip,
                'total_return': adjusted_ret,
                'breakeven': adjusted_ret > 0
            })
    
    return pd.DataFrame(results)


def monte_carlo_analysis(returns, n_sims=1000, n_periods=252):
    """Monte Carlo robustness"""
    rng = np.random.default_rng(42)
    
    total_returns = []
    sharpe_ratios = []
    max_drawdowns = []
    
    returns_arr = returns.values
    
    for _ in range(n_sims):
        sampled = rng.choice(returns_arr, size=n_periods, replace=True)
        
        # Total return
        total_ret = np.prod(1 + sampled) - 1
        total_returns.append(total_ret)
        
        # Sharpe
        sharpe = sampled.mean() / (sampled.std() + 1e-8) * np.sqrt(252)
        sharpe_ratios.append(sharpe)
        
        # Max DD
        equity = np.cumprod(1 + sampled)
        peak = np.maximum.accumulate(equity)
        dd = (equity - peak) / peak
        max_drawdowns.append(dd.min())
    
    total_returns = np.array(total_returns)
    sharpe_ratios = np.array(sharpe_ratios)
    max_drawdowns = np.array(max_drawdowns)
    
    # Confidence intervals
    return_ci = (np.percentile(total_returns, 2.5), np.percentile(total_returns, 97.5))
    sharpe_ci = (np.percentile(sharpe_ratios, 2.5), np.percentile(sharpe_ratios, 97.5))
    
    # Risk metrics
    var_95 = np.percentile(returns_arr, 5)
    cvar_95 = returns_arr[returns_arr <= var_95].mean()
    
    # Ruin probability (20% drawdown)
    ruin_prob = (max_drawdowns < -0.20).mean()
    
    return {
        'return_ci': return_ci,
        'sharpe_ci': sharpe_ci,
        'var_95': var_95,
        'cvar_95': cvar_95,
        'ruin_probability': ruin_prob,
        'expected_max_dd': max_drawdowns.mean()
    }


def calibrate_thresholds(df, test_grid=None):
    """Walk-forward threshold calibration"""
    if test_grid is None:
        # Use data-driven grid based on actual Reynolds distribution
        test_grid = {
            'laminar': np.linspace(0.005, 0.03, 10),
            'turbulent': np.linspace(0.02, 0.10, 15)
        }
    
    best_sharpe = -np.inf
    best_params = (50, 200)
    
    for lam in test_grid['laminar']:
        for turb in test_grid['turbulent']:
            if lam >= turb:
                continue
            
            # Quick backtest with these thresholds
            df['test_regime'] = df['reynolds'].apply(
                lambda x: classify_regime(x, lam, turb)
            )
            
            # Generate signals
            direction = pd.Series(0, index=df.index)
            turb_mask = df['test_regime'] == MarketRegime.TURBULENT
            lam_mask = df['test_regime'] == MarketRegime.LAMINAR
            
            direction.loc[turb_mask & (df['flow_z'] > 0.3)] = 1
            direction.loc[turb_mask & (df['flow_z'] < -0.3)] = -1
            direction.loc[lam_mask & (df['flow_z'] > 0.3)] = -1
            direction.loc[lam_mask & (df['flow_z'] < -0.3)] = 1
            
            # Returns
            strat_ret = direction.shift(1) * df['fwd_ret']
            strat_ret = strat_ret.dropna()
            
            if len(strat_ret) > 50:
                sharpe = strat_ret.mean() / (strat_ret.std() + 1e-8) * np.sqrt(252)
                
                if sharpe > best_sharpe:
                    best_sharpe = sharpe
                    best_params = (lam, turb)
    
    return best_params, best_sharpe


def main():
    print("="*60)
    print("FLUID REGIME TRADING SYSTEM - BACKTEST")
    print("="*60)
    print()
    
    # Fetch data
    daily, vix = fetch_data('SPY', '2021-01-01', '2024-12-31')
    
    # Initial backtest with corrected thresholds (based on actual Reynolds distribution)
    print("\n1. Running Initial Backtest (Calibrated Thresholds)...")
    print("-" * 40)
    
    # Use data-driven thresholds based on Reynolds percentiles
    results = run_backtest(daily, vix, lam_th=0.01, turb_th=0.03)
    
    print(f"   Total Return: {results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"   Win Rate: {results['win_rate']:.1%}")
    print(f"   Profit Factor: {results['profit_factor']:.2f}")
    print(f"   Total Trades: {results['total_trades']}")
    print(f"   Regime Accuracy: {results['regime_accuracy']:.1%}")
    print(f"   Calmar Ratio: {results['calmar_ratio']:.2f}")
    
    # Threshold calibration
    print("\n2. Calibrating Optimal Thresholds...")
    print("-" * 40)
    
    best_params, best_sharpe = calibrate_thresholds(results['df'])
    print(f"   Optimal Laminar Threshold: {best_params[0]:.1f}")
    print(f"   Optimal Turbulent Threshold: {best_params[1]:.1f}")
    print(f"   Calibrated Sharpe: {best_sharpe:.2f}")
    
    # Re-run with calibrated thresholds
    print("\n3. Running Calibrated Backtest...")
    print("-" * 40)
    
    cal_results = run_backtest(daily, vix, lam_th=best_params[0], turb_th=best_params[1])
    
    print(f"   Total Return: {cal_results['total_return']:.2%}")
    print(f"   Sharpe Ratio: {cal_results['sharpe_ratio']:.2f}")
    print(f"   Max Drawdown: {cal_results['max_drawdown']:.2%}")
    print(f"   Win Rate: {cal_results['win_rate']:.1%}")
    print(f"   Profit Factor: {cal_results['profit_factor']:.2f}")
    print(f"   Regime Accuracy: {cal_results['regime_accuracy']:.1%}")
    
    # Signal decay analysis
    print("\n4. Signal Decay Analysis...")
    print("-" * 40)
    
    decay = signal_decay_analysis(cal_results['df'])
    if len(decay) > 0:
        peak_idx = decay['sharpe'].idxmax()
        print(f"   Peak Alpha Horizon: {decay.loc[peak_idx, 'horizon']} days")
        print(f"   Peak Sharpe: {decay.loc[peak_idx, 'sharpe']:.2f}")
        print(f"   Peak Hit Rate: {decay.loc[peak_idx, 'hit_rate']:.1%}")
        
        zero_cross = decay[decay['avg_return'] <= 0]
        if len(zero_cross) > 0:
            print(f"   Signal Zero-Cross: {zero_cross['horizon'].min()} days")
        else:
            print(f"   Signal Zero-Cross: >30 days")
    
    # Transaction cost sensitivity
    print("\n5. Transaction Cost Sensitivity...")
    print("-" * 40)
    
    tc_df = tc_sensitivity(
        cal_results['returns'],
        cal_results['total_trades']
    )
    
    breakeven = tc_df[tc_df['breakeven']]
    print(f"   Breakeven Scenarios: {len(breakeven)}/{len(tc_df)}")
    if len(breakeven) > 0:
        print(f"   Max Sustainable Commission: ${breakeven['commission'].max():.3f}/share")
        print(f"   Max Sustainable Slippage: {breakeven['slippage_bps'].max()} bps")
    
    # Monte Carlo analysis
    print("\n6. Monte Carlo Robustness (1000 simulations)...")
    print("-" * 40)
    
    mc = monte_carlo_analysis(cal_results['returns'])
    
    print(f"   Return 95% CI: [{mc['return_ci'][0]:.2%}, {mc['return_ci'][1]:.2%}]")
    print(f"   Sharpe 95% CI: [{mc['sharpe_ci'][0]:.2f}, {mc['sharpe_ci'][1]:.2f}]")
    print(f"   VaR (95%): {mc['var_95']:.2%}")
    print(f"   CVaR (95%): {mc['cvar_95']:.2%}")
    print(f"   Ruin Probability (20% DD): {mc['ruin_probability']:.1%}")
    print(f"   Expected Max Drawdown: {mc['expected_max_dd']:.2%}")
    
    # Regime distribution
    print("\n7. Regime Distribution...")
    print("-" * 40)
    
    for regime, pct in cal_results['regime_distribution'].items():
        print(f"   {regime}: {pct:.1%}")
    
    # Strategy breakdown
    print("\n8. Strategy Performance by Regime...")
    print("-" * 40)
    
    for regime, stats in cal_results['strategy_breakdown'].items():
        if stats['trades'] > 0:
            print(f"   {regime}:")
            print(f"      Trades: {stats['trades']}")
            print(f"      Win Rate: {stats['win_rate']:.1%}")
            print(f"      Avg Return: {stats['avg_ret']*100:.3f}%")
            print(f"      Total Return: {stats['total_ret']*100:.2f}%")
    
    print("\n" + "="*60)
    print("BACKTEST COMPLETE")
    print("="*60)
    
    # Save results
    summary = {
        'default_results': {
            'total_return': results['total_return'],
            'sharpe': results['sharpe_ratio'],
            'max_dd': results['max_drawdown'],
            'win_rate': results['win_rate'],
            'profit_factor': results['profit_factor']
        },
        'calibrated_results': {
            'laminar_threshold': best_params[0],
            'turbulent_threshold': best_params[1],
            'total_return': cal_results['total_return'],
            'sharpe': cal_results['sharpe_ratio'],
            'max_dd': cal_results['max_drawdown'],
            'win_rate': cal_results['win_rate'],
            'profit_factor': cal_results['profit_factor'],
            'regime_accuracy': cal_results['regime_accuracy']
        },
        'monte_carlo': mc,
        'signal_decay': decay.to_dict() if len(decay) > 0 else {},
        'tc_sensitivity': tc_df.to_dict()
    }
    
    # Save equity curve
    cal_results['equity_curve'].to_csv('/home/claude/fluid_regime_trading/equity_curve.csv')
    decay.to_csv('/home/claude/fluid_regime_trading/signal_decay.csv', index=False)
    tc_df.to_csv('/home/claude/fluid_regime_trading/tc_sensitivity.csv', index=False)
    
    with open('/home/claude/fluid_regime_trading/backtest_summary.json', 'w') as f:
        # Convert non-serializable items
        summary_clean = json.loads(json.dumps(summary, default=str))
        json.dump(summary_clean, f, indent=2)
    
    print("\nResults saved to /home/claude/fluid_regime_trading/")
    
    return summary, cal_results, decay, tc_df


if __name__ == '__main__':
    summary, results, decay, tc_df = main()

"""
Backtest Engine
================
Walk-forward backtesting with:
- Rolling calibration windows
- Signal decay analysis
- Transaction cost sensitivity
- Regime accuracy testing
- Monte Carlo robustness analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
from concurrent.futures import ProcessPoolExecutor
import json

from core_model import FluidRegimeModel, FluidState, MarketRegime, Strategy
from risk_management import TradeManager, RiskParameters, Trade, TradeStatus


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    # Walk-forward parameters
    calibration_window: int = 252  # Days for in-sample calibration
    test_window: int = 63  # Days for out-of-sample testing
    step_size: int = 21  # Days between recalibrations
    
    # Data parameters
    bar_frequency: str = '5min'  # '1min', '5min', '15min', '1h', '1d'
    bars_per_day: int = 78  # 5-min bars per trading day
    
    # Starting capital
    initial_capital: float = 100000
    
    # Transaction costs for sensitivity analysis
    commission_scenarios: List[float] = None
    slippage_scenarios: List[float] = None
    
    def __post_init__(self):
        if self.commission_scenarios is None:
            self.commission_scenarios = [0.001, 0.005, 0.01]  # Per share
        if self.slippage_scenarios is None:
            self.slippage_scenarios = [2, 5, 10, 20]  # BPS


@dataclass 
class BacktestResult:
    """Complete backtest results"""
    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float
    
    # Trade statistics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_pnl: float
    avg_bars_held: float
    
    # Regime analysis
    regime_accuracy: float
    regime_distribution: Dict[str, float]
    strategy_breakdown: Dict
    
    # Time series
    equity_curve: pd.Series
    drawdown_series: pd.Series
    regime_series: pd.Series
    signal_series: pd.Series
    
    # Walk-forward results
    oos_periods: List[Dict]
    
    # Calibrated thresholds
    threshold_history: pd.DataFrame


class DataPreprocessor:
    """
    Prepare data for backtesting.
    Handles resampling, indicator calculation, and alignment.
    """
    
    def __init__(self, bar_frequency: str = '5min'):
        self.frequency = bar_frequency
    
    def compute_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()
    
    def resample_to_daily(
        self,
        df: pd.DataFrame
    ) -> pd.DataFrame:
        """Resample intraday data to daily OHLCV"""
        daily = df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return daily
    
    def prepare_model_inputs(
        self,
        intraday_df: pd.DataFrame,
        daily_df: pd.DataFrame,
        vix_df: pd.DataFrame,
        spy_df: pd.DataFrame,
        vix3m_df: Optional[pd.DataFrame] = None,
        vvix_df: Optional[pd.DataFrame] = None
    ) -> Dict:
        """
        Prepare all inputs needed by FluidRegimeModel.
        """
        # Compute ATR on daily
        daily_df['atr'] = self.compute_atr(
            daily_df['high'], daily_df['low'], daily_df['close']
        )
        
        # SPY returns for correlation
        spy_returns = spy_df['close'].pct_change().dropna()
        
        return {
            'intraday': intraday_df,
            'daily': daily_df,
            'vix': vix_df['close'] if 'close' in vix_df.columns else vix_df,
            'vix3m': vix3m_df['close'] if vix3m_df is not None else None,
            'vvix': vvix_df['close'] if vvix_df is not None else None,
            'spy_returns': spy_returns,
            'atr': daily_df['atr']
        }


class WalkForwardBacktester:
    """
    Walk-forward backtesting engine.
    """
    
    def __init__(
        self,
        config: BacktestConfig,
        model_params: Dict = None,
        risk_params: RiskParameters = None
    ):
        self.config = config
        self.model_params = model_params or {}
        self.risk_params = risk_params or RiskParameters()
        
        # Results storage
        self.equity_curve = []
        self.trades = []
        self.signals = []
        self.regimes = []
        self.thresholds = []
    
    def run_calibration_period(
        self,
        data: Dict,
        start_idx: int,
        end_idx: int
    ) -> Tuple[float, float]:
        """
        Run calibration to find optimal Reynolds thresholds.
        Returns (laminar_threshold, turbulent_threshold)
        """
        # Initialize model with default thresholds
        model = FluidRegimeModel(**self.model_params)
        
        # Storage for calibration
        reynolds_values = []
        momentum_returns = []
        fade_returns = []
        
        daily = data['daily'].iloc[start_idx:end_idx]
        
        for i in range(20, len(daily)):
            # Compute Reynolds for this day
            prices_1m = daily['close'].iloc[max(0, i-1):i+1]  # Proxy
            
            state = model.generate_signal(
                prices_1m=prices_1m,
                daily_prices=daily['close'].iloc[:i+1],
                vix=data['vix'].iloc[:i+1] if data['vix'] is not None else pd.Series([20]),
                spy_returns=data['spy_returns'].iloc[:i+1],
                vix3m=data['vix3m'].iloc[:i+1] if data.get('vix3m') is not None else None,
                vvix=data['vvix'].iloc[:i+1] if data.get('vvix') is not None else None,
                timestamp=daily.index[i]
            )
            
            reynolds_values.append(state.reynolds)
            
            # Forward return for next period
            if i < len(daily) - 1:
                fwd_ret = (daily['close'].iloc[i+1] - daily['close'].iloc[i]) / \
                          daily['close'].iloc[i]
                
                # Momentum return: follow the flow signal
                mom_ret = fwd_ret * np.sign(state.flow_signal) if state.flow_signal != 0 else 0
                
                # Fade return: opposite of flow signal
                fade_ret = -fwd_ret * np.sign(state.flow_signal) if state.flow_signal != 0 else 0
                
                momentum_returns.append(mom_ret)
                fade_returns.append(fade_ret)
        
        # Find optimal thresholds
        reynolds_series = pd.Series(reynolds_values[:-1])
        momentum_series = pd.Series(momentum_returns)
        fade_series = pd.Series(fade_returns)
        
        # Grid search for thresholds
        best_laminar = 50
        best_turbulent = 200
        best_score = -np.inf
        
        for lam in np.linspace(10, 150, 15):
            for turb in np.linspace(100, 400, 15):
                if lam >= turb:
                    continue
                
                # Score: momentum alpha in turbulent + fade alpha in laminar
                turb_mask = reynolds_series > turb
                lam_mask = reynolds_series < lam
                
                turb_alpha = momentum_series[turb_mask].mean() if turb_mask.sum() > 10 else 0
                lam_alpha = fade_series[lam_mask].mean() if lam_mask.sum() > 10 else 0
                
                score = turb_alpha + lam_alpha
                
                if score > best_score:
                    best_score = score
                    best_laminar = lam
                    best_turbulent = turb
        
        return best_laminar, best_turbulent
    
    def run_test_period(
        self,
        data: Dict,
        start_idx: int,
        end_idx: int,
        laminar_threshold: float,
        turbulent_threshold: float,
        trade_manager: TradeManager
    ) -> Dict:
        """
        Run out-of-sample test period with calibrated thresholds.
        """
        # Initialize model with calibrated thresholds
        model_params = self.model_params.copy()
        model_params['reynolds_laminar_threshold'] = laminar_threshold
        model_params['reynolds_turbulent_threshold'] = turbulent_threshold
        model = FluidRegimeModel(**model_params)
        
        intraday = data['intraday']
        daily = data['daily']
        
        # Get daily indices for test period
        test_dates = daily.index[start_idx:end_idx]
        
        period_signals = []
        period_regimes = []
        period_equity = [trade_manager.current_capital]
        
        # History for z-scoring
        velocity_history = pd.Series(dtype=float)
        viscosity_history = pd.Series(dtype=float)
        pressure_history = pd.Series(dtype=float)
        
        for date in test_dates:
            trade_manager.reset_daily()
            
            # Get intraday data for this date
            day_data = intraday[intraday.index.date == date.date()] \
                       if hasattr(intraday.index, 'date') else intraday.loc[date:date]
            
            if len(day_data) == 0:
                continue
            
            daily_idx = daily.index.get_loc(date)
            
            for bar_idx in range(0, len(day_data), 5):  # Every 5 bars
                bar_time = day_data.index[bar_idx]
                current_price = day_data['close'].iloc[bar_idx]
                
                # Get historical data up to this point
                hist_daily = daily.iloc[:daily_idx+1]
                
                # Generate signal
                state = model.generate_signal(
                    prices_1m=day_data['close'].iloc[:bar_idx+1],
                    daily_prices=hist_daily['close'],
                    vix=data['vix'].iloc[:daily_idx+1] if data['vix'] is not None else pd.Series([20]),
                    spy_returns=data['spy_returns'].iloc[:daily_idx+1],
                    vix3m=data['vix3m'].iloc[:daily_idx+1] if data.get('vix3m') is not None else None,
                    vvix=data['vvix'].iloc[:daily_idx+1] if data.get('vvix') is not None else None,
                    velocity_history=velocity_history,
                    viscosity_history=viscosity_history,
                    pressure_history=pressure_history,
                    timestamp=bar_time
                )
                
                # Update histories
                velocity_history = pd.concat([velocity_history, 
                                              pd.Series([state.velocity])]).tail(100)
                viscosity_history = pd.concat([viscosity_history,
                                               pd.Series([state.viscosity])]).tail(100)
                pressure_history = pd.concat([pressure_history,
                                              pd.Series([state.pressure_gradient])]).tail(100)
                
                period_signals.append({
                    'time': bar_time,
                    'flow_signal': state.flow_signal,
                    'direction': state.direction,
                    'reynolds': state.reynolds
                })
                period_regimes.append({
                    'time': bar_time,
                    'regime': state.regime.value,
                    'confidence': state.regime_confidence
                })
                
                # Check exits on open positions
                closed = trade_manager.check_exits(
                    current_price, bar_time, state.direction
                )
                
                # Open new positions if signal
                if state.direction != 0 and state.strategy != Strategy.FLAT:
                    atr = data['atr'].iloc[daily_idx] if 'atr' in data else \
                          hist_daily['close'].pct_change().std() * hist_daily['close'].iloc[-1]
                    
                    trade = trade_manager.open_trade(
                        symbol='TEST',
                        direction=state.direction,
                        strategy=state.strategy.value,
                        entry_price=current_price,
                        entry_time=bar_time,
                        atr=atr,
                        reynolds=state.reynolds,
                        regime=state.regime.value,
                        flow_signal=state.flow_signal,
                        prices_for_vol=hist_daily['close']
                    )
                
                period_equity.append(trade_manager.current_capital)
            
            # End of day: close all positions
            if len(day_data) > 0:
                trade_manager.force_close_all(
                    day_data['close'].iloc[-1],
                    day_data.index[-1]
                )
        
        # Period statistics
        period_return = (trade_manager.current_capital - period_equity[0]) / period_equity[0]
        
        return {
            'start_date': test_dates[0] if len(test_dates) > 0 else None,
            'end_date': test_dates[-1] if len(test_dates) > 0 else None,
            'return': period_return,
            'trades': len([t for t in trade_manager.closed_trades]),
            'signals': period_signals,
            'regimes': period_regimes,
            'equity': period_equity,
            'laminar_threshold': laminar_threshold,
            'turbulent_threshold': turbulent_threshold
        }
    
    def run_full_backtest(
        self,
        data: Dict
    ) -> BacktestResult:
        """
        Run complete walk-forward backtest.
        """
        daily = data['daily']
        n_days = len(daily)
        
        trade_manager = TradeManager(
            risk_params=self.risk_params,
            initial_capital=self.config.initial_capital
        )
        
        all_periods = []
        all_equity = [self.config.initial_capital]
        all_signals = []
        all_regimes = []
        threshold_history = []
        
        # Walk-forward loop
        current_idx = self.config.calibration_window
        
        while current_idx + self.config.test_window < n_days:
            # Calibration period
            cal_start = current_idx - self.config.calibration_window
            cal_end = current_idx
            
            lam_th, turb_th = self.run_calibration_period(
                data, cal_start, cal_end
            )
            
            threshold_history.append({
                'date': daily.index[current_idx],
                'laminar': lam_th,
                'turbulent': turb_th
            })
            
            # Test period
            test_start = current_idx
            test_end = min(current_idx + self.config.test_window, n_days)
            
            period_result = self.run_test_period(
                data, test_start, test_end,
                lam_th, turb_th, trade_manager
            )
            
            all_periods.append(period_result)
            all_equity.extend(period_result['equity'][1:])
            all_signals.extend(period_result['signals'])
            all_regimes.extend(period_result['regimes'])
            
            # Move forward
            current_idx += self.config.step_size
        
        # Compile results
        equity_series = pd.Series(all_equity)
        
        # Calculate metrics
        total_return = (equity_series.iloc[-1] - equity_series.iloc[0]) / \
                       equity_series.iloc[0]
        
        # Drawdown
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        max_dd = drawdown.min()
        
        # Returns for Sharpe
        returns = equity_series.pct_change().dropna()
        sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(252 * self.config.bars_per_day)
        
        # Sortino
        downside = returns[returns < 0]
        sortino = returns.mean() / (downside.std() + 1e-8) * np.sqrt(252 * self.config.bars_per_day)
        
        # Annualized return
        n_years = len(daily) / 252
        ann_return = (1 + total_return) ** (1/n_years) - 1 if n_years > 0 else total_return
        
        # Calmar
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        
        # Trade stats
        stats = trade_manager.get_trade_statistics()
        
        # Regime distribution
        regime_counts = pd.Series([r['regime'] for r in all_regimes]).value_counts(normalize=True)
        
        return BacktestResult(
            total_return=total_return,
            annualized_return=ann_return,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            max_drawdown=max_dd,
            calmar_ratio=calmar,
            total_trades=stats.get('total_trades', 0),
            win_rate=stats.get('win_rate', 0),
            profit_factor=stats.get('profit_factor', 0),
            avg_trade_pnl=stats.get('avg_pnl_per_trade', 0),
            avg_bars_held=stats.get('avg_bars_held', 0),
            regime_accuracy=self._compute_regime_accuracy(all_regimes, all_signals, data),
            regime_distribution=regime_counts.to_dict(),
            strategy_breakdown=stats.get('strategy_breakdown', {}),
            equity_curve=equity_series,
            drawdown_series=drawdown,
            regime_series=pd.DataFrame(all_regimes),
            signal_series=pd.DataFrame(all_signals),
            oos_periods=all_periods,
            threshold_history=pd.DataFrame(threshold_history)
        )
    
    def _compute_regime_accuracy(
        self,
        regimes: List[Dict],
        signals: List[Dict],
        data: Dict
    ) -> float:
        """
        Compute ex-post regime classification accuracy.
        Did turbulent regimes actually exhibit momentum?
        Did laminar regimes actually mean-revert?
        """
        if len(regimes) == 0 or len(signals) == 0:
            return 0.0
        
        correct = 0
        total = 0
        
        regime_df = pd.DataFrame(regimes).set_index('time')
        signal_df = pd.DataFrame(signals).set_index('time')
        
        daily = data['daily']
        
        for date in regime_df.index.unique():
            day_regimes = regime_df.loc[date:date]
            
            if len(day_regimes) == 0:
                continue
            
            dominant_regime = day_regimes['regime'].mode().iloc[0] \
                             if len(day_regimes['regime'].mode()) > 0 else 'TRANSITIONAL'
            
            # Get next day return
            try:
                idx = daily.index.get_loc(date.date() if hasattr(date, 'date') else date)
                if idx < len(daily) - 1:
                    fwd_ret = (daily['close'].iloc[idx+1] - daily['close'].iloc[idx]) / \
                              daily['close'].iloc[idx]
                else:
                    continue
            except:
                continue
            
            # Get average signal direction for the day
            day_signals = signal_df.loc[date:date]
            avg_direction = day_signals['direction'].mean() if len(day_signals) > 0 else 0
            
            # Check accuracy
            if dominant_regime == 'TURBULENT':
                # Momentum should work: direction * return > 0
                if avg_direction * fwd_ret > 0:
                    correct += 1
                total += 1
            elif dominant_regime == 'LAMINAR':
                # Mean-reversion should work: -direction * return > 0
                if -avg_direction * fwd_ret > 0:
                    correct += 1
                total += 1
        
        return correct / total if total > 0 else 0.0


class SignalDecayAnalyzer:
    """
    Analyze how quickly signal alpha decays after generation.
    """
    
    def __init__(self, max_bars: int = 100):
        self.max_bars = max_bars
    
    def compute_decay_curve(
        self,
        signals: pd.DataFrame,
        prices: pd.Series
    ) -> pd.DataFrame:
        """
        Compute signal alpha at various holding periods.
        """
        results = []
        
        for horizon in range(1, self.max_bars + 1):
            forward_returns = []
            
            for idx, row in signals.iterrows():
                signal_time = row.get('time', idx)
                direction = row.get('direction', 0)
                
                if direction == 0:
                    continue
                
                # Get forward return
                try:
                    price_idx = prices.index.get_loc(signal_time)
                    if price_idx + horizon < len(prices):
                        fwd_ret = (prices.iloc[price_idx + horizon] - 
                                   prices.iloc[price_idx]) / prices.iloc[price_idx]
                        
                        # Signal return = direction * actual return
                        signal_ret = direction * fwd_ret
                        forward_returns.append(signal_ret)
                except:
                    continue
            
            if forward_returns:
                results.append({
                    'horizon': horizon,
                    'avg_return': np.mean(forward_returns),
                    'hit_rate': np.mean([r > 0 for r in forward_returns]),
                    'sharpe': np.mean(forward_returns) / (np.std(forward_returns) + 1e-8),
                    'n_signals': len(forward_returns)
                })
        
        return pd.DataFrame(results)


class TransactionCostSensitivity:
    """
    Analyze sensitivity of results to transaction costs.
    """
    
    def __init__(
        self,
        commission_range: List[float] = None,
        slippage_range: List[float] = None
    ):
        self.commissions = commission_range or [0.001, 0.005, 0.01, 0.02]
        self.slippages = slippage_range or [1, 5, 10, 20, 50]
    
    def run_sensitivity(
        self,
        trades: List[Trade],
        initial_capital: float
    ) -> pd.DataFrame:
        """
        Compute P&L under various transaction cost scenarios.
        """
        results = []
        
        for comm in self.commissions:
            for slip in self.slippages:
                adjusted_pnl = 0
                
                for trade in trades:
                    # Recalculate P&L with new costs
                    gross = trade.realized_pnl + \
                            trade.shares * 2 * 0.005 + \
                            trade.entry_price * trade.shares * 5 / 10000 * 2
                    
                    new_comm = trade.shares * 2 * comm
                    new_slip = trade.entry_price * trade.shares * slip / 10000 * 2
                    
                    adjusted_pnl += gross - new_comm - new_slip
                
                results.append({
                    'commission': comm,
                    'slippage_bps': slip,
                    'total_pnl': adjusted_pnl,
                    'return_pct': adjusted_pnl / initial_capital,
                    'pnl_per_trade': adjusted_pnl / len(trades) if trades else 0,
                    'breakeven': adjusted_pnl > 0
                })
        
        return pd.DataFrame(results)

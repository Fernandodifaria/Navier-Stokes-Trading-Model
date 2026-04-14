"""
Entry/Exit Logic and Risk Management
=====================================
Professional-grade trade management with:
- Multi-level stop losses (initial, trailing, time-based)
- Dynamic profit targets
- Kelly criterion position sizing
- Volatility-adjusted sizing
- Maximum drawdown controls
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta


class TradeStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    CLOSED_PROFIT = "CLOSED_PROFIT"
    CLOSED_LOSS = "CLOSED_LOSS"
    CLOSED_TRAILING = "CLOSED_TRAILING"
    CLOSED_TIME = "CLOSED_TIME"
    CLOSED_SIGNAL = "CLOSED_SIGNAL"


@dataclass
class Trade:
    """Complete trade record"""
    trade_id: int
    symbol: str
    direction: int  # 1 = long, -1 = short
    strategy: str
    entry_time: pd.Timestamp
    entry_price: float
    position_size: float  # Fraction of capital
    shares: int
    
    # Risk parameters at entry
    initial_stop: float
    profit_target: float
    trailing_stop_pct: float
    max_hold_bars: int
    
    # State
    status: TradeStatus = TradeStatus.OPEN
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    high_water_mark: float = 0
    bars_held: int = 0
    
    # P&L
    realized_pnl: float = 0
    realized_pnl_pct: float = 0
    max_adverse_excursion: float = 0  # Worst unrealized loss
    max_favorable_excursion: float = 0  # Best unrealized gain
    
    # Metadata
    entry_reynolds: float = 0
    entry_regime: str = ""
    entry_flow_signal: float = 0
    
    def update_excursions(self, current_price: float):
        """Track MAE and MFE"""
        if self.direction == 1:
            unrealized = (current_price - self.entry_price) / self.entry_price
        else:
            unrealized = (self.entry_price - current_price) / self.entry_price
        
        if unrealized < 0:
            self.max_adverse_excursion = min(self.max_adverse_excursion, unrealized)
        else:
            self.max_favorable_excursion = max(self.max_favorable_excursion, unrealized)
        
        # Update high water mark for trailing stop
        if unrealized > 0:
            self.high_water_mark = max(self.high_water_mark, unrealized)


@dataclass
class RiskParameters:
    """Risk management configuration"""
    # Position sizing
    max_position_pct: float = 0.10  # Max 10% of capital per trade
    max_total_exposure: float = 0.50  # Max 50% total exposure
    kelly_fraction: float = 0.25  # Fraction of Kelly to use
    
    # Stop losses
    initial_stop_atr_mult: float = 2.0  # Initial stop = 2 ATR
    trailing_stop_activation: float = 0.02  # Activate trailing after 2% profit
    trailing_stop_pct: float = 0.015  # Trail by 1.5%
    
    # Profit targets
    profit_target_atr_mult: float = 3.0  # Target = 3 ATR
    
    # Time stops
    max_hold_bars: int = 78  # 2 trading days in 5-min bars
    time_decay_start: int = 39  # Start reducing after 1 day
    
    # Drawdown controls
    max_daily_drawdown: float = 0.02  # 2% max daily loss
    max_total_drawdown: float = 0.05  # 5% max total drawdown
    
    # Transaction costs
    commission_per_share: float = 0.005
    slippage_bps: float = 5  # 5 bps slippage


class KellyCriterion:
    """
    Kelly criterion for optimal position sizing.
    Uses half-Kelly for safety margin.
    """
    
    def __init__(self, lookback: int = 100, min_trades: int = 20):
        self.lookback = lookback
        self.min_trades = min_trades
        self.trade_history: List[float] = []
    
    def add_trade(self, return_pct: float):
        """Add trade result to history"""
        self.trade_history.append(return_pct)
        if len(self.trade_history) > self.lookback:
            self.trade_history = self.trade_history[-self.lookback:]
    
    def compute_kelly_fraction(self) -> float:
        """
        Compute optimal Kelly fraction.
        f* = (p * b - q) / b
        where p = win rate, q = loss rate, b = avg win / avg loss
        """
        if len(self.trade_history) < self.min_trades:
            return 0.1  # Conservative default
        
        trades = np.array(self.trade_history)
        winners = trades[trades > 0]
        losers = trades[trades < 0]
        
        if len(winners) == 0 or len(losers) == 0:
            return 0.1
        
        win_rate = len(winners) / len(trades)
        loss_rate = 1 - win_rate
        
        avg_win = winners.mean()
        avg_loss = abs(losers.mean())
        
        if avg_loss == 0:
            return 0.1
        
        b = avg_win / avg_loss  # Win/loss ratio
        
        kelly = (win_rate * b - loss_rate) / b
        
        # Bound Kelly to reasonable range
        return np.clip(kelly, 0, 0.5)
    
    def get_position_size(
        self,
        kelly_multiplier: float = 0.25,
        max_size: float = 0.10
    ) -> float:
        """Get position size as fraction of capital"""
        kelly = self.compute_kelly_fraction()
        size = kelly * kelly_multiplier
        return min(size, max_size)


class VolatilityScaler:
    """
    Volatility-based position sizing.
    Target constant dollar volatility per position.
    """
    
    def __init__(
        self,
        target_vol: float = 0.01,  # 1% daily vol target per position
        vol_lookback: int = 20,
        vol_cap: float = 3.0  # Max vol multiplier
    ):
        self.target_vol = target_vol
        self.lookback = vol_lookback
        self.vol_cap = vol_cap
    
    def compute_position_size(
        self,
        prices: pd.Series,
        base_size: float = 0.10
    ) -> float:
        """
        Scale position size inversely with volatility.
        """
        if len(prices) < self.lookback:
            return base_size
        
        returns = prices.pct_change().dropna().tail(self.lookback)
        realized_vol = returns.std()
        
        if realized_vol == 0:
            return base_size
        
        # Scale factor: target_vol / realized_vol
        scale = self.target_vol / realized_vol
        scale = np.clip(scale, 1/self.vol_cap, self.vol_cap)
        
        return base_size * scale


class TradeManager:
    """
    Manages open positions with professional risk controls.
    """
    
    def __init__(
        self,
        risk_params: RiskParameters,
        initial_capital: float = 100000
    ):
        self.params = risk_params
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.daily_pnl = 0
        
        self.open_trades: Dict[int, Trade] = {}
        self.closed_trades: List[Trade] = []
        self.trade_counter = 0
        
        self.kelly = KellyCriterion()
        self.vol_scaler = VolatilityScaler()
        
        # Drawdown tracking
        self.peak_capital = initial_capital
        self.current_drawdown = 0
        self.daily_start_capital = initial_capital
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_start_capital = self.current_capital
        self.daily_pnl = 0
    
    def check_drawdown_limits(self) -> bool:
        """Check if we've hit drawdown limits"""
        # Daily drawdown
        daily_dd = (self.daily_start_capital - self.current_capital) / \
                   self.daily_start_capital
        if daily_dd > self.params.max_daily_drawdown:
            return False
        
        # Total drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        total_dd = (self.peak_capital - self.current_capital) / self.peak_capital
        if total_dd > self.params.max_total_drawdown:
            return False
        
        return True
    
    def get_total_exposure(self) -> float:
        """Current total exposure as fraction of capital"""
        total = sum(t.position_size for t in self.open_trades.values())
        return total
    
    def can_open_trade(self, position_size: float) -> bool:
        """Check if we can open a new trade"""
        if not self.check_drawdown_limits():
            return False
        
        current_exposure = self.get_total_exposure()
        if current_exposure + position_size > self.params.max_total_exposure:
            return False
        
        return True
    
    def compute_stops_and_targets(
        self,
        entry_price: float,
        direction: int,
        atr: float
    ) -> Tuple[float, float]:
        """
        Compute initial stop and profit target.
        """
        stop_distance = atr * self.params.initial_stop_atr_mult
        target_distance = atr * self.params.profit_target_atr_mult
        
        if direction == 1:  # Long
            stop = entry_price - stop_distance
            target = entry_price + target_distance
        else:  # Short
            stop = entry_price + stop_distance
            target = entry_price - target_distance
        
        return stop, target
    
    def open_trade(
        self,
        symbol: str,
        direction: int,
        strategy: str,
        entry_price: float,
        entry_time: pd.Timestamp,
        atr: float,
        reynolds: float,
        regime: str,
        flow_signal: float,
        prices_for_vol: pd.Series
    ) -> Optional[Trade]:
        """
        Open a new trade with full risk management.
        """
        # Base position size from Kelly
        kelly_size = self.kelly.get_position_size(
            kelly_multiplier=self.params.kelly_fraction,
            max_size=self.params.max_position_pct
        )
        
        # Adjust for volatility
        vol_adjusted = self.vol_scaler.compute_position_size(
            prices_for_vol, kelly_size
        )
        
        position_size = min(vol_adjusted, self.params.max_position_pct)
        
        # Check if we can open
        if not self.can_open_trade(position_size):
            return None
        
        # Compute stops and targets
        initial_stop, profit_target = self.compute_stops_and_targets(
            entry_price, direction, atr
        )
        
        # Calculate shares
        dollar_size = self.current_capital * position_size
        shares = int(dollar_size / entry_price)
        
        if shares == 0:
            return None
        
        # Create trade
        self.trade_counter += 1
        trade = Trade(
            trade_id=self.trade_counter,
            symbol=symbol,
            direction=direction,
            strategy=strategy,
            entry_time=entry_time,
            entry_price=entry_price,
            position_size=position_size,
            shares=shares,
            initial_stop=initial_stop,
            profit_target=profit_target,
            trailing_stop_pct=self.params.trailing_stop_pct,
            max_hold_bars=self.params.max_hold_bars,
            entry_reynolds=reynolds,
            entry_regime=regime,
            entry_flow_signal=flow_signal
        )
        
        self.open_trades[trade.trade_id] = trade
        return trade
    
    def check_exits(
        self,
        current_price: float,
        current_time: pd.Timestamp,
        new_signal_direction: int
    ) -> List[Trade]:
        """
        Check all open trades for exit conditions.
        Returns list of closed trades.
        """
        closed = []
        
        for trade_id, trade in list(self.open_trades.items()):
            trade.bars_held += 1
            trade.update_excursions(current_price)
            
            exit_reason = None
            exit_price = current_price
            
            # 1. Initial stop loss
            if trade.direction == 1:  # Long
                if current_price <= trade.initial_stop:
                    exit_reason = TradeStatus.CLOSED_LOSS
            else:  # Short
                if current_price >= trade.initial_stop:
                    exit_reason = TradeStatus.CLOSED_LOSS
            
            # 2. Profit target
            if exit_reason is None:
                if trade.direction == 1:
                    if current_price >= trade.profit_target:
                        exit_reason = TradeStatus.CLOSED_PROFIT
                else:
                    if current_price <= trade.profit_target:
                        exit_reason = TradeStatus.CLOSED_PROFIT
            
            # 3. Trailing stop (only if in profit)
            if exit_reason is None and trade.high_water_mark > \
               self.params.trailing_stop_activation:
                
                if trade.direction == 1:
                    trail_price = trade.entry_price * (1 + trade.high_water_mark - 
                                                        trade.trailing_stop_pct)
                    if current_price <= trail_price:
                        exit_reason = TradeStatus.CLOSED_TRAILING
                else:
                    trail_price = trade.entry_price * (1 - trade.high_water_mark + 
                                                        trade.trailing_stop_pct)
                    if current_price >= trail_price:
                        exit_reason = TradeStatus.CLOSED_TRAILING
            
            # 4. Time stop
            if exit_reason is None and trade.bars_held >= trade.max_hold_bars:
                exit_reason = TradeStatus.CLOSED_TIME
            
            # 5. Signal reversal
            if exit_reason is None and new_signal_direction != 0 and \
               new_signal_direction != trade.direction:
                exit_reason = TradeStatus.CLOSED_SIGNAL
            
            # Close the trade
            if exit_reason is not None:
                trade = self._close_trade(trade, exit_price, current_time, exit_reason)
                closed.append(trade)
        
        return closed
    
    def _close_trade(
        self,
        trade: Trade,
        exit_price: float,
        exit_time: pd.Timestamp,
        status: TradeStatus
    ) -> Trade:
        """
        Close a trade and update capital.
        """
        trade.exit_price = exit_price
        trade.exit_time = exit_time
        trade.status = status
        
        # Calculate P&L
        if trade.direction == 1:
            gross_pnl = (exit_price - trade.entry_price) * trade.shares
        else:
            gross_pnl = (trade.entry_price - exit_price) * trade.shares
        
        # Transaction costs
        commission = trade.shares * 2 * self.params.commission_per_share
        slippage = trade.entry_price * trade.shares * self.params.slippage_bps / 10000 * 2
        
        trade.realized_pnl = gross_pnl - commission - slippage
        trade.realized_pnl_pct = trade.realized_pnl / (trade.entry_price * trade.shares)
        
        # Update capital
        self.current_capital += trade.realized_pnl
        self.daily_pnl += trade.realized_pnl
        
        # Update Kelly with trade result
        self.kelly.add_trade(trade.realized_pnl_pct)
        
        # Move to closed
        del self.open_trades[trade.trade_id]
        self.closed_trades.append(trade)
        
        return trade
    
    def force_close_all(
        self,
        current_price: float,
        current_time: pd.Timestamp
    ) -> List[Trade]:
        """Force close all open positions (end of day, etc.)"""
        closed = []
        for trade_id, trade in list(self.open_trades.items()):
            trade = self._close_trade(
                trade, current_price, current_time, TradeStatus.CLOSED_TIME
            )
            closed.append(trade)
        return closed
    
    def get_trade_statistics(self) -> Dict:
        """
        Compute comprehensive trade statistics.
        """
        if not self.closed_trades:
            return {}
        
        pnls = [t.realized_pnl for t in self.closed_trades]
        pnl_pcts = [t.realized_pnl_pct for t in self.closed_trades]
        
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        total_trades = len(self.closed_trades)
        win_rate = len(winners) / total_trades if total_trades > 0 else 0
        
        avg_win = np.mean(winners) if winners else 0
        avg_loss = abs(np.mean(losers)) if losers else 0
        
        profit_factor = sum(winners) / abs(sum(losers)) if losers else float('inf')
        
        # By exit type
        exit_counts = {}
        for trade in self.closed_trades:
            status = trade.status.value
            exit_counts[status] = exit_counts.get(status, 0) + 1
        
        # MAE/MFE analysis
        mae_values = [t.max_adverse_excursion for t in self.closed_trades]
        mfe_values = [t.max_favorable_excursion for t in self.closed_trades]
        
        # By strategy
        strategy_stats = {}
        for trade in self.closed_trades:
            strat = trade.strategy
            if strat not in strategy_stats:
                strategy_stats[strat] = {'trades': 0, 'pnl': 0, 'wins': 0}
            strategy_stats[strat]['trades'] += 1
            strategy_stats[strat]['pnl'] += trade.realized_pnl
            if trade.realized_pnl > 0:
                strategy_stats[strat]['wins'] += 1
        
        return {
            'total_trades': total_trades,
            'total_pnl': sum(pnls),
            'total_pnl_pct': (self.current_capital - self.initial_capital) / \
                             self.initial_capital,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'avg_pnl_per_trade': np.mean(pnls),
            'std_pnl': np.std(pnls),
            'sharpe_estimate': np.mean(pnl_pcts) / (np.std(pnl_pcts) + 1e-8) * \
                               np.sqrt(252),
            'max_drawdown': (self.peak_capital - min(
                self.initial_capital,
                min(t.realized_pnl for t in self.closed_trades) + self.initial_capital
            )) / self.peak_capital if self.closed_trades else 0,
            'avg_mae': np.mean(mae_values),
            'avg_mfe': np.mean(mfe_values),
            'avg_bars_held': np.mean([t.bars_held for t in self.closed_trades]),
            'exit_distribution': exit_counts,
            'strategy_breakdown': strategy_stats,
            'final_capital': self.current_capital,
            'kelly_fraction': self.kelly.compute_kelly_fraction()
        }

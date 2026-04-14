"""
Fluid Regime Trading Model - Core Engine
=========================================
Jane Street-style quantitative trading system using fluid dynamics metaphors
for regime classification and signal generation.

Key Variables:
- Velocity: Signed momentum accumulation (trend persistence)
- Viscosity: Inverse volatility (resistance to directional movement)
- Pressure Gradient: Multi-factor forcing (VIX derivatives, term structure, sentiment)
- Reynolds Number: Regime classifier (laminar vs turbulent)
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from enum import Enum


class MarketRegime(Enum):
    LAMINAR = "LAMINAR"          # Mean-reversion dominates
    TRANSITIONAL = "TRANSITIONAL"  # Uncertain, reduce exposure
    TURBULENT = "TURBULENT"       # Momentum dominates


class Strategy(Enum):
    MOMENTUM_LONG = "MOMENTUM_LONG"
    MOMENTUM_SHORT = "MOMENTUM_SHORT"
    FADE_LONG = "FADE_LONG"
    FADE_SHORT = "FADE_SHORT"
    FLAT = "FLAT"


@dataclass
class FluidState:
    """Complete state of the fluid regime model at a point in time"""
    timestamp: pd.Timestamp
    velocity: float
    velocity_zscore: float
    viscosity: float
    viscosity_zscore: float
    pressure_gradient: float
    pressure_zscore: float
    reynolds: float
    flow_signal: float
    regime: MarketRegime
    regime_confidence: float
    strategy: Strategy
    direction: int  # -1, 0, 1
    size_multiplier: float
    correlation_to_spy: float


class FluidRegimeModel:
    """
    Production-grade fluid regime trading model with:
    - Z-score normalization
    - Regime hysteresis
    - Multi-factor pressure gradients
    - Correlation overlay
    - Adaptive thresholds
    """
    
    def __init__(
        self,
        # Velocity parameters
        velocity_bars: int = 390,  # 1-min bars per trading day
        velocity_ema_span: int = 20,  # EMA smoothing for velocity
        
        # Viscosity parameters
        viscosity_lookback: int = 14,  # Days for volatility calc
        viscosity_ema_span: int = 5,
        
        # Pressure gradient parameters
        pressure_lookback: int = 5,  # Bars for rate of change
        
        # Normalization
        zscore_window: int = 63,  # ~3 months for z-score
        
        # Regime classification
        reynolds_laminar_threshold: float = 50,
        reynolds_turbulent_threshold: float = 200,
        regime_hysteresis_bars: int = 5,  # Bars to confirm regime change
        
        # Flow signal
        flow_dead_zone: float = 0.3,  # Z-score threshold for flat
        
        # Correlation filter
        correlation_window: int = 20,
        correlation_threshold: float = 0.8,
        
        # Position sizing
        base_position_size: float = 1.0,
        transitional_reduction: float = 0.5,
        high_correlation_reduction: float = 0.5
    ):
        self.velocity_bars = velocity_bars
        self.velocity_ema_span = velocity_ema_span
        self.viscosity_lookback = viscosity_lookback
        self.viscosity_ema_span = viscosity_ema_span
        self.pressure_lookback = pressure_lookback
        self.zscore_window = zscore_window
        self.reynolds_laminar = reynolds_laminar_threshold
        self.reynolds_turbulent = reynolds_turbulent_threshold
        self.hysteresis_bars = regime_hysteresis_bars
        self.flow_dead_zone = flow_dead_zone
        self.correlation_window = correlation_window
        self.correlation_threshold = correlation_threshold
        self.base_size = base_position_size
        self.transitional_reduction = transitional_reduction
        self.high_corr_reduction = high_correlation_reduction
        
        # State tracking for hysteresis
        self._regime_history: List[MarketRegime] = []
        self._current_regime: MarketRegime = MarketRegime.TRANSITIONAL
        self._bars_in_regime: int = 0
    
    def compute_velocity(
        self, 
        prices_1m: pd.Series,
        normalize: bool = True
    ) -> Tuple[float, float]:
        """
        Signed momentum accumulation over trading period.
        Returns (raw_velocity, normalized_velocity)
        """
        if len(prices_1m) < 2:
            return 0.0, 0.0
        
        returns = prices_1m.pct_change().dropna()
        
        # Signed momentum: sum of directional moves weighted by magnitude
        signed_momentum = (np.sign(returns) * np.abs(returns)).sum()
        
        # Normalize by time and ATR proxy
        atr_proxy = returns.abs().mean() + 1e-8
        raw_velocity = signed_momentum / (len(returns) * atr_proxy)
        
        # Apply EMA smoothing
        if hasattr(self, '_velocity_history'):
            alpha = 2 / (self.velocity_ema_span + 1)
            smoothed = alpha * raw_velocity + (1 - alpha) * self._velocity_history[-1]
        else:
            smoothed = raw_velocity
        
        return raw_velocity, smoothed
    
    def compute_viscosity(
        self,
        daily_prices: pd.Series,
        daily_high: Optional[pd.Series] = None,
        daily_low: Optional[pd.Series] = None
    ) -> Tuple[float, float]:
        """
        Inverse volatility - resistance to directional movement.
        High vol = low viscosity = momentum persists
        Low vol = high viscosity = mean-reversion dominates
        """
        if len(daily_prices) < self.viscosity_lookback:
            return 1.0, 0.0
        
        prices = daily_prices.tail(self.viscosity_lookback)
        returns = prices.pct_change().dropna()
        
        # Realized volatility (annualized)
        realized_vol = returns.std() * np.sqrt(252)
        
        # Range expansion factor
        if daily_high is not None and daily_low is not None:
            high = daily_high.tail(self.viscosity_lookback)
            low = daily_low.tail(self.viscosity_lookback)
            avg_range = ((high - low) / prices).mean()
        else:
            avg_range = returns.abs().mean() * 2  # Proxy
        
        # Viscosity = inverse of vol * range
        raw_viscosity = 1 / (realized_vol * avg_range + 1e-6)
        
        # Bound to reasonable range
        bounded = np.clip(raw_viscosity, 0.1, 50)
        
        return bounded, realized_vol
    
    def compute_pressure_gradient(
        self,
        vix: pd.Series,
        vix3m: Optional[pd.Series],
        vvix: Optional[pd.Series],
        put_call_ratio: Optional[pd.Series],
        price: pd.Series
    ) -> Tuple[float, Dict[str, float]]:
        """
        Multi-factor pressure gradient combining:
        1. VIX rate of change (inverted)
        2. VIX term structure (contango/backwardation)
        3. VVIX (vol-of-vol) rate of change
        4. Put/call ratio rate of change
        
        Returns: (composite_pressure, component_dict)
        """
        components = {}
        weights = {'vix_roc': 0.3, 'term_structure': 0.25, 
                   'vvix': 0.25, 'put_call': 0.2}
        
        # Price direction for signing
        price_direction = np.sign(
            price.iloc[-1] - price.iloc[-self.pressure_lookback]
        ) if len(price) >= self.pressure_lookback else 0
        
        # 1. VIX Rate of Change (inverted - falling VIX = bullish)
        if len(vix) >= self.pressure_lookback:
            vix_roc = (vix.iloc[-1] - vix.iloc[-self.pressure_lookback]) / \
                      (vix.iloc[-self.pressure_lookback] + 1e-6)
            # Invert: negative VIX change + positive price = bullish pressure
            components['vix_roc'] = -vix_roc * price_direction
        else:
            components['vix_roc'] = 0
            weights['vix_roc'] = 0
        
        # 2. Term Structure (VIX vs VIX3M)
        if vix3m is not None and len(vix3m) > 0 and len(vix) > 0:
            # Contango (VIX < VIX3M) = bullish, Backwardation = bearish
            term_spread = (vix3m.iloc[-1] - vix.iloc[-1]) / (vix.iloc[-1] + 1e-6)
            components['term_structure'] = term_spread
        else:
            components['term_structure'] = 0
            weights['term_structure'] = 0
        
        # 3. VVIX Rate of Change
        if vvix is not None and len(vvix) >= self.pressure_lookback:
            vvix_roc = (vvix.iloc[-1] - vvix.iloc[-self.pressure_lookback]) / \
                       (vvix.iloc[-self.pressure_lookback] + 1e-6)
            # High VVIX change = uncertainty = negative pressure
            components['vvix'] = -vvix_roc
        else:
            components['vvix'] = 0
            weights['vvix'] = 0
        
        # 4. Put/Call Ratio
        if put_call_ratio is not None and len(put_call_ratio) >= self.pressure_lookback:
            pcr_roc = (put_call_ratio.iloc[-1] - put_call_ratio.iloc[-self.pressure_lookback]) / \
                      (put_call_ratio.iloc[-self.pressure_lookback] + 1e-6)
            # Rising put/call = bearish pressure
            components['put_call'] = -pcr_roc
        else:
            components['put_call'] = 0
            weights['put_call'] = 0
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        # Composite pressure
        composite = sum(components[k] * weights[k] for k in components)
        
        return composite, components
    
    def compute_reynolds(
        self,
        velocity: float,
        viscosity: float,
        regime_duration: int = 20
    ) -> float:
        """
        Market Reynolds number: determines regime type
        Re = |v| × L / μ
        """
        return abs(velocity) * regime_duration / (viscosity + 1e-6)
    
    def classify_regime(
        self,
        reynolds: float,
        apply_hysteresis: bool = True
    ) -> Tuple[MarketRegime, float]:
        """
        Classify regime with hysteresis to prevent whipsaws.
        Returns (regime, confidence)
        """
        # Raw classification
        if reynolds < self.reynolds_laminar:
            raw_regime = MarketRegime.LAMINAR
            # Confidence: how far below threshold
            confidence = 1 - (reynolds / self.reynolds_laminar)
        elif reynolds > self.reynolds_turbulent:
            raw_regime = MarketRegime.TURBULENT
            # Confidence: how far above threshold
            confidence = min(1.0, (reynolds - self.reynolds_turbulent) / 
                           self.reynolds_turbulent)
        else:
            raw_regime = MarketRegime.TRANSITIONAL
            # Confidence in transitional is low by definition
            confidence = 0.5
        
        if not apply_hysteresis:
            return raw_regime, confidence
        
        # Apply hysteresis
        if raw_regime == self._current_regime:
            self._bars_in_regime += 1
            return self._current_regime, confidence
        else:
            self._regime_history.append(raw_regime)
            # Check if we have N consecutive bars of new regime
            if len(self._regime_history) >= self.hysteresis_bars:
                recent = self._regime_history[-self.hysteresis_bars:]
                if all(r == raw_regime for r in recent):
                    # Confirmed regime change
                    self._current_regime = raw_regime
                    self._bars_in_regime = 1
                    self._regime_history = []
                    return raw_regime, confidence
            
            # Not enough confirmation, stay in current regime
            return self._current_regime, confidence * 0.5  # Reduced confidence
    
    def compute_flow_signal(
        self,
        velocity: float,
        pressure: float,
        viscosity: float
    ) -> float:
        """
        Composite directional signal: Flow = (v × ∇P) / μ
        """
        return (velocity * pressure) / (viscosity + 1e-6)
    
    def zscore(
        self,
        value: float,
        history: pd.Series
    ) -> float:
        """Z-score normalization against rolling history"""
        if len(history) < self.zscore_window:
            return 0.0
        
        window = history.tail(self.zscore_window)
        mean = window.mean()
        std = window.std()
        
        if std < 1e-8:
            return 0.0
        
        return (value - mean) / std
    
    def compute_correlation_to_spy(
        self,
        asset_returns: pd.Series,
        spy_returns: pd.Series
    ) -> float:
        """Rolling correlation to SPY for macro regime detection"""
        if len(asset_returns) < self.correlation_window or \
           len(spy_returns) < self.correlation_window:
            return 0.0
        
        asset = asset_returns.tail(self.correlation_window)
        spy = spy_returns.tail(self.correlation_window)
        
        return asset.corr(spy)
    
    def determine_strategy(
        self,
        regime: MarketRegime,
        flow_signal_zscore: float
    ) -> Tuple[Strategy, int]:
        """
        Map regime + signal to trading strategy
        Returns (strategy, direction)
        """
        # Dead zone check
        if abs(flow_signal_zscore) < self.flow_dead_zone:
            return Strategy.FLAT, 0
        
        direction = int(np.sign(flow_signal_zscore))
        
        if regime == MarketRegime.TURBULENT:
            # Momentum regime: follow the signal
            if direction > 0:
                return Strategy.MOMENTUM_LONG, 1
            else:
                return Strategy.MOMENTUM_SHORT, -1
        
        elif regime == MarketRegime.LAMINAR:
            # Mean-reversion: fade the signal
            if direction > 0:
                return Strategy.FADE_SHORT, -1
            else:
                return Strategy.FADE_LONG, 1
        
        else:  # TRANSITIONAL
            return Strategy.FLAT, 0
    
    def compute_position_size(
        self,
        regime: MarketRegime,
        regime_confidence: float,
        correlation_to_spy: float,
        flow_signal_zscore: float
    ) -> float:
        """
        Dynamic position sizing based on regime confidence and conditions
        """
        base = self.base_size
        
        # Regime adjustment
        if regime == MarketRegime.TRANSITIONAL:
            base *= self.transitional_reduction
        
        # Confidence scaling
        base *= regime_confidence
        
        # Correlation adjustment (reduce in high-corr regimes)
        if abs(correlation_to_spy) > self.correlation_threshold:
            base *= self.high_corr_reduction
        
        # Signal strength scaling
        signal_strength = min(abs(flow_signal_zscore), 3.0) / 3.0
        base *= (0.5 + 0.5 * signal_strength)  # Scale 50%-100%
        
        return np.clip(base, 0, 1.0)
    
    def generate_signal(
        self,
        prices_1m: pd.Series,
        daily_prices: pd.Series,
        vix: pd.Series,
        spy_returns: pd.Series,
        vix3m: Optional[pd.Series] = None,
        vvix: Optional[pd.Series] = None,
        put_call_ratio: Optional[pd.Series] = None,
        daily_high: Optional[pd.Series] = None,
        daily_low: Optional[pd.Series] = None,
        velocity_history: Optional[pd.Series] = None,
        viscosity_history: Optional[pd.Series] = None,
        pressure_history: Optional[pd.Series] = None,
        timestamp: Optional[pd.Timestamp] = None
    ) -> FluidState:
        """
        Master signal generation function.
        Returns complete FluidState with all computed values.
        """
        # Compute raw values
        raw_velocity, smoothed_velocity = self.compute_velocity(prices_1m)
        raw_viscosity, realized_vol = self.compute_viscosity(
            daily_prices, daily_high, daily_low
        )
        raw_pressure, pressure_components = self.compute_pressure_gradient(
            vix, vix3m, vvix, put_call_ratio, daily_prices
        )
        
        # Z-score normalization
        velocity_z = self.zscore(smoothed_velocity, velocity_history) \
                     if velocity_history is not None else smoothed_velocity
        viscosity_z = self.zscore(raw_viscosity, viscosity_history) \
                      if viscosity_history is not None else 0
        pressure_z = self.zscore(raw_pressure, pressure_history) \
                     if pressure_history is not None else raw_pressure
        
        # Reynolds number (use z-scored velocity, raw viscosity)
        reynolds = self.compute_reynolds(velocity_z, raw_viscosity)
        
        # Regime classification with hysteresis
        regime, regime_confidence = self.classify_regime(reynolds)
        
        # Flow signal
        flow_signal = self.compute_flow_signal(
            velocity_z, pressure_z, raw_viscosity
        )
        
        # Correlation
        asset_returns = daily_prices.pct_change().dropna()
        correlation = self.compute_correlation_to_spy(asset_returns, spy_returns)
        
        # Strategy and direction
        strategy, direction = self.determine_strategy(regime, flow_signal)
        
        # Position sizing
        size_mult = self.compute_position_size(
            regime, regime_confidence, correlation, flow_signal
        )
        
        return FluidState(
            timestamp=timestamp or pd.Timestamp.now(),
            velocity=smoothed_velocity,
            velocity_zscore=velocity_z,
            viscosity=raw_viscosity,
            viscosity_zscore=viscosity_z,
            pressure_gradient=raw_pressure,
            pressure_zscore=pressure_z,
            reynolds=reynolds,
            flow_signal=flow_signal,
            regime=regime,
            regime_confidence=regime_confidence,
            strategy=strategy,
            direction=direction,
            size_multiplier=size_mult,
            correlation_to_spy=correlation
        )


class AdaptiveThresholdCalibrator:
    """
    Walk-forward calibration of Reynolds thresholds per asset.
    Finds thresholds where momentum vs mean-reversion alpha flips sign.
    """
    
    def __init__(
        self,
        calibration_window: int = 252,  # 1 year
        step_size: int = 21,  # Monthly recalibration
        threshold_grid: np.ndarray = None
    ):
        self.calibration_window = calibration_window
        self.step_size = step_size
        self.threshold_grid = threshold_grid or np.linspace(10, 500, 50)
    
    def compute_regime_returns(
        self,
        reynolds_series: pd.Series,
        returns: pd.Series,
        threshold: float,
        regime_type: str  # 'laminar' or 'turbulent'
    ) -> float:
        """
        Compute average returns in a given regime
        """
        if regime_type == 'laminar':
            mask = reynolds_series < threshold
        else:
            mask = reynolds_series > threshold
        
        regime_returns = returns[mask]
        if len(regime_returns) < 10:
            return np.nan
        
        return regime_returns.mean()
    
    def find_optimal_thresholds(
        self,
        reynolds_series: pd.Series,
        returns: pd.Series,
        momentum_returns: pd.Series,  # Returns from following momentum
        fade_returns: pd.Series  # Returns from fading
    ) -> Tuple[float, float]:
        """
        Find thresholds where strategy alpha flips sign.
        
        Laminar threshold: Below this, fade > momentum
        Turbulent threshold: Above this, momentum > fade
        """
        best_laminar = 50
        best_turbulent = 200
        best_laminar_spread = 0
        best_turbulent_spread = 0
        
        for threshold in self.threshold_grid:
            # For laminar: we want fade returns > momentum returns
            mask_below = reynolds_series < threshold
            if mask_below.sum() > 20:
                fade_below = fade_returns[mask_below].mean()
                mom_below = momentum_returns[mask_below].mean()
                spread = fade_below - mom_below
                if spread > best_laminar_spread:
                    best_laminar_spread = spread
                    best_laminar = threshold
            
            # For turbulent: we want momentum returns > fade returns
            mask_above = reynolds_series > threshold
            if mask_above.sum() > 20:
                mom_above = momentum_returns[mask_above].mean()
                fade_above = fade_returns[mask_above].mean()
                spread = mom_above - fade_above
                if spread > best_turbulent_spread:
                    best_turbulent_spread = spread
                    best_turbulent = threshold
        
        # Ensure laminar < turbulent
        if best_laminar >= best_turbulent:
            best_laminar = best_turbulent * 0.3
        
        return best_laminar, best_turbulent
    
    def walk_forward_calibration(
        self,
        reynolds_series: pd.Series,
        returns: pd.Series,
        momentum_returns: pd.Series,
        fade_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Rolling window calibration of thresholds
        """
        results = []
        
        for i in range(self.calibration_window, 
                       len(reynolds_series) - self.step_size, 
                       self.step_size):
            
            # Calibration window
            cal_re = reynolds_series.iloc[i-self.calibration_window:i]
            cal_ret = returns.iloc[i-self.calibration_window:i]
            cal_mom = momentum_returns.iloc[i-self.calibration_window:i]
            cal_fade = fade_returns.iloc[i-self.calibration_window:i]
            
            laminar_th, turbulent_th = self.find_optimal_thresholds(
                cal_re, cal_ret, cal_mom, cal_fade
            )
            
            results.append({
                'date': reynolds_series.index[i],
                'laminar_threshold': laminar_th,
                'turbulent_threshold': turbulent_th
            })
        
        return pd.DataFrame(results).set_index('date')

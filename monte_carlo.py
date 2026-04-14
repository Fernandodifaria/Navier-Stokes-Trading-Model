"""
Monte Carlo Robustness Testing
===============================
Test regime classifier robustness through:
- Bootstrapped parameter sensitivity
- Synthetic data generation with known regimes
- Ruin probability estimation
- Confidence intervals on performance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from concurrent.futures import ThreadPoolExecutor
import warnings

from core_model import FluidRegimeModel, MarketRegime


@dataclass
class MonteCarloConfig:
    """Monte Carlo simulation configuration"""
    n_simulations: int = 1000
    confidence_level: float = 0.95
    random_seed: int = 42
    
    # Parameter perturbation
    param_noise_std: float = 0.1  # 10% noise on parameters
    
    # Synthetic data generation
    n_synthetic_days: int = 252
    regime_switch_prob: float = 0.05  # Daily probability of regime switch
    
    # Risk metrics
    ruin_threshold: float = 0.20  # 20% drawdown = ruin


@dataclass
class MonteCarloResult:
    """Monte Carlo analysis results"""
    # Parameter sensitivity
    param_sensitivity: pd.DataFrame
    
    # Performance distribution
    return_distribution: np.ndarray
    sharpe_distribution: np.ndarray
    
    # Confidence intervals
    return_ci: Tuple[float, float]
    sharpe_ci: Tuple[float, float]
    
    # Risk metrics
    ruin_probability: float
    expected_max_drawdown: float
    var_95: float
    cvar_95: float
    
    # Regime classifier metrics
    regime_accuracy_distribution: np.ndarray
    regime_stability: float  # How often regime classification is consistent


class SyntheticDataGenerator:
    """
    Generate synthetic market data with known regime characteristics.
    Useful for testing if the model correctly identifies regimes.
    """
    
    def __init__(
        self,
        n_days: int = 252,
        bars_per_day: int = 78,
        regime_switch_prob: float = 0.05,
        seed: int = 42
    ):
        self.n_days = n_days
        self.bars_per_day = bars_per_day
        self.switch_prob = regime_switch_prob
        self.rng = np.random.default_rng(seed)
        
        # Regime parameters
        self.regime_params = {
            'LAMINAR': {
                'drift': 0.0001,
                'vol': 0.005,
                'mean_reversion': 0.3,
                'momentum_persistence': 0.1
            },
            'TURBULENT': {
                'drift': 0.0003,
                'vol': 0.015,
                'mean_reversion': 0.05,
                'momentum_persistence': 0.7
            },
            'TRANSITIONAL': {
                'drift': 0.0001,
                'vol': 0.010,
                'mean_reversion': 0.15,
                'momentum_persistence': 0.4
            }
        }
    
    def generate_regime_sequence(self) -> List[str]:
        """Generate sequence of regimes with realistic transitions"""
        regimes = ['TRANSITIONAL']
        current = 'TRANSITIONAL'
        
        transition_matrix = {
            'LAMINAR': {'LAMINAR': 0.90, 'TRANSITIONAL': 0.08, 'TURBULENT': 0.02},
            'TRANSITIONAL': {'LAMINAR': 0.15, 'TRANSITIONAL': 0.70, 'TURBULENT': 0.15},
            'TURBULENT': {'LAMINAR': 0.02, 'TRANSITIONAL': 0.08, 'TURBULENT': 0.90}
        }
        
        for _ in range(self.n_days - 1):
            probs = transition_matrix[current]
            states = list(probs.keys())
            probabilities = list(probs.values())
            current = self.rng.choice(states, p=probabilities)
            regimes.append(current)
        
        return regimes
    
    def generate_prices(
        self,
        regimes: List[str],
        initial_price: float = 100
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Generate intraday and daily price data based on regime sequence.
        Returns (intraday_df, daily_df)
        """
        all_prices = []
        all_times = []
        
        current_price = initial_price
        
        for day_idx, regime in enumerate(regimes):
            params = self.regime_params[regime]
            
            day_prices = [current_price]
            
            for bar in range(1, self.bars_per_day):
                # Mean-reverting momentum process
                prev_return = (day_prices[-1] - day_prices[-2]) / day_prices[-2] \
                              if len(day_prices) > 1 else 0
                
                # Momentum component
                momentum = params['momentum_persistence'] * prev_return
                
                # Mean-reversion component (to daily open)
                reversion = -params['mean_reversion'] * \
                            (day_prices[-1] - day_prices[0]) / day_prices[0]
                
                # Random shock
                shock = self.rng.normal(params['drift'], params['vol'])
                
                new_return = momentum + reversion + shock
                new_price = day_prices[-1] * (1 + new_return)
                day_prices.append(max(new_price, 0.01))
            
            all_prices.extend(day_prices)
            
            # Generate timestamps
            base_date = pd.Timestamp('2024-01-01') + pd.Timedelta(days=day_idx)
            day_times = pd.date_range(
                start=base_date + pd.Timedelta(hours=9, minutes=30),
                periods=self.bars_per_day,
                freq='5min'
            )
            all_times.extend(day_times)
            
            current_price = day_prices[-1]
        
        # Create intraday DataFrame
        intraday_df = pd.DataFrame({
            'close': all_prices,
            'open': all_prices,
            'high': np.array(all_prices) * (1 + self.rng.uniform(0, 0.002, len(all_prices))),
            'low': np.array(all_prices) * (1 - self.rng.uniform(0, 0.002, len(all_prices))),
            'volume': self.rng.integers(100000, 1000000, len(all_prices))
        }, index=all_times)
        
        # Aggregate to daily
        daily_df = intraday_df.resample('D').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        
        return intraday_df, daily_df
    
    def generate_vix(
        self,
        regimes: List[str]
    ) -> pd.Series:
        """Generate synthetic VIX based on regimes"""
        vix_values = []
        current_vix = 15
        
        regime_vix = {
            'LAMINAR': (12, 2),
            'TRANSITIONAL': (18, 4),
            'TURBULENT': (28, 8)
        }
        
        for regime in regimes:
            target, vol = regime_vix[regime]
            # Mean-reverting VIX
            shock = self.rng.normal(0, vol)
            current_vix = current_vix + 0.1 * (target - current_vix) + shock
            current_vix = np.clip(current_vix, 9, 80)
            vix_values.append(current_vix)
        
        dates = pd.date_range('2024-01-01', periods=len(regimes), freq='D')
        return pd.Series(vix_values, index=dates, name='vix')


class RegimeClassifierTester:
    """
    Test regime classifier accuracy on synthetic data with known regimes.
    """
    
    def __init__(
        self,
        model_params: Dict = None,
        n_trials: int = 100,
        seed: int = 42
    ):
        self.model_params = model_params or {}
        self.n_trials = n_trials
        self.seed = seed
    
    def run_accuracy_test(self) -> pd.DataFrame:
        """
        Run multiple trials of regime classification on synthetic data.
        """
        results = []
        
        for trial in range(self.n_trials):
            # Generate synthetic data with known regimes
            generator = SyntheticDataGenerator(
                n_days=252,
                seed=self.seed + trial
            )
            
            true_regimes = generator.generate_regime_sequence()
            intraday_df, daily_df = generator.generate_prices(true_regimes)
            vix = generator.generate_vix(true_regimes)
            
            # Run model
            model = FluidRegimeModel(**self.model_params)
            predicted_regimes = []
            
            spy_returns = daily_df['close'].pct_change().dropna()
            
            for i in range(20, len(daily_df)):
                state = model.generate_signal(
                    prices_1m=intraday_df['close'].iloc[i*78:(i+1)*78],
                    daily_prices=daily_df['close'].iloc[:i+1],
                    vix=vix.iloc[:i+1],
                    spy_returns=spy_returns.iloc[:i+1],
                    timestamp=daily_df.index[i]
                )
                predicted_regimes.append(state.regime.value)
            
            # Calculate accuracy
            true_subset = true_regimes[20:]
            
            # Overall accuracy
            accuracy = np.mean([p == t for p, t in zip(predicted_regimes, true_subset)])
            
            # Per-regime accuracy
            regime_accuracy = {}
            for regime in ['LAMINAR', 'TRANSITIONAL', 'TURBULENT']:
                mask = [t == regime for t in true_subset]
                if sum(mask) > 0:
                    regime_preds = [p for p, m in zip(predicted_regimes, mask) if m]
                    regime_accuracy[regime] = np.mean([p == regime for p in regime_preds])
                else:
                    regime_accuracy[regime] = np.nan
            
            results.append({
                'trial': trial,
                'overall_accuracy': accuracy,
                'laminar_accuracy': regime_accuracy['LAMINAR'],
                'transitional_accuracy': regime_accuracy['TRANSITIONAL'],
                'turbulent_accuracy': regime_accuracy['TURBULENT'],
                'n_predictions': len(predicted_regimes)
            })
        
        return pd.DataFrame(results)


class ParameterSensitivityAnalyzer:
    """
    Analyze sensitivity of model to parameter changes.
    """
    
    def __init__(
        self,
        base_params: Dict,
        param_ranges: Dict = None,
        n_samples: int = 100,
        seed: int = 42
    ):
        self.base_params = base_params
        self.param_ranges = param_ranges or self._default_ranges()
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)
    
    def _default_ranges(self) -> Dict:
        """Default parameter ranges for sensitivity analysis"""
        return {
            'reynolds_laminar_threshold': (20, 100),
            'reynolds_turbulent_threshold': (100, 400),
            'zscore_window': (21, 126),
            'regime_hysteresis_bars': (2, 10),
            'flow_dead_zone': (0.1, 0.5),
            'viscosity_lookback': (7, 28),
            'pressure_lookback': (3, 10)
        }
    
    def generate_parameter_samples(self) -> List[Dict]:
        """Generate parameter samples using Latin Hypercube Sampling"""
        samples = []
        
        for _ in range(self.n_samples):
            params = self.base_params.copy()
            
            for param, (low, high) in self.param_ranges.items():
                if param in params:
                    if isinstance(params[param], int):
                        params[param] = self.rng.integers(int(low), int(high))
                    else:
                        params[param] = self.rng.uniform(low, high)
            
            samples.append(params)
        
        return samples
    
    def run_sensitivity_analysis(
        self,
        data: Dict,
        evaluate_fn  # Function that takes params and data, returns metric
    ) -> pd.DataFrame:
        """
        Run sensitivity analysis across parameter samples.
        """
        samples = self.generate_parameter_samples()
        results = []
        
        for i, params in enumerate(samples):
            try:
                metric = evaluate_fn(params, data)
                result = {'sample': i, 'metric': metric}
                result.update(params)
                results.append(result)
            except Exception as e:
                warnings.warn(f"Sample {i} failed: {e}")
        
        return pd.DataFrame(results)


class RiskMetricsCalculator:
    """
    Calculate various risk metrics from return series.
    """
    
    def __init__(
        self,
        returns: np.ndarray,
        confidence_level: float = 0.95,
        ruin_threshold: float = 0.20
    ):
        self.returns = returns
        self.confidence = confidence_level
        self.ruin_threshold = ruin_threshold
    
    def compute_var(self) -> float:
        """Value at Risk"""
        return np.percentile(self.returns, (1 - self.confidence) * 100)
    
    def compute_cvar(self) -> float:
        """Conditional Value at Risk (Expected Shortfall)"""
        var = self.compute_var()
        return self.returns[self.returns <= var].mean()
    
    def compute_max_drawdown_distribution(
        self,
        n_paths: int = 1000,
        path_length: int = 252
    ) -> np.ndarray:
        """Bootstrap max drawdown distribution"""
        rng = np.random.default_rng(42)
        max_dds = []
        
        for _ in range(n_paths):
            # Bootstrap sample of returns
            sampled = rng.choice(self.returns, size=path_length, replace=True)
            
            # Compute equity curve
            equity = np.cumprod(1 + sampled)
            
            # Max drawdown
            peak = np.maximum.accumulate(equity)
            dd = (peak - equity) / peak
            max_dds.append(dd.max())
        
        return np.array(max_dds)
    
    def compute_ruin_probability(
        self,
        n_simulations: int = 10000,
        n_trades: int = 100
    ) -> float:
        """
        Estimate probability of hitting ruin threshold.
        """
        rng = np.random.default_rng(42)
        ruins = 0
        
        for _ in range(n_simulations):
            # Simulate sequence of trades
            trade_returns = rng.choice(self.returns, size=n_trades, replace=True)
            
            # Equity curve
            equity = np.cumprod(1 + trade_returns)
            
            # Check if we ever hit ruin
            min_equity = equity.min()
            if min_equity < (1 - self.ruin_threshold):
                ruins += 1
        
        return ruins / n_simulations
    
    def compute_kelly_fraction(self) -> float:
        """Optimal Kelly fraction"""
        wins = self.returns[self.returns > 0]
        losses = self.returns[self.returns < 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return 0.1
        
        p = len(wins) / len(self.returns)
        q = 1 - p
        b = wins.mean() / abs(losses.mean())
        
        kelly = (p * b - q) / b
        return np.clip(kelly, 0, 0.5)


class MonteCarloSimulator:
    """
    Main Monte Carlo simulation engine.
    """
    
    def __init__(
        self,
        config: MonteCarloConfig
    ):
        self.config = config
        self.rng = np.random.default_rng(config.random_seed)
    
    def bootstrap_returns(
        self,
        returns: np.ndarray,
        n_periods: int = 252
    ) -> np.ndarray:
        """Generate bootstrapped return paths"""
        paths = []
        
        for _ in range(self.config.n_simulations):
            sampled = self.rng.choice(returns, size=n_periods, replace=True)
            paths.append(sampled)
        
        return np.array(paths)
    
    def compute_performance_distribution(
        self,
        paths: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute distribution of performance metrics"""
        total_returns = []
        sharpe_ratios = []
        
        for path in paths:
            # Total return
            total_ret = np.prod(1 + path) - 1
            total_returns.append(total_ret)
            
            # Sharpe
            sharpe = path.mean() / (path.std() + 1e-8) * np.sqrt(252)
            sharpe_ratios.append(sharpe)
        
        return np.array(total_returns), np.array(sharpe_ratios)
    
    def compute_confidence_interval(
        self,
        values: np.ndarray
    ) -> Tuple[float, float]:
        """Compute confidence interval"""
        alpha = 1 - self.config.confidence_level
        lower = np.percentile(values, alpha/2 * 100)
        upper = np.percentile(values, (1 - alpha/2) * 100)
        return (lower, upper)
    
    def run_full_analysis(
        self,
        trade_returns: np.ndarray,
        model_params: Dict,
        data: Dict = None
    ) -> MonteCarloResult:
        """
        Run complete Monte Carlo analysis.
        """
        # Bootstrap return paths
        paths = self.bootstrap_returns(trade_returns)
        
        # Performance distributions
        return_dist, sharpe_dist = self.compute_performance_distribution(paths)
        
        # Confidence intervals
        return_ci = self.compute_confidence_interval(return_dist)
        sharpe_ci = self.compute_confidence_interval(sharpe_dist)
        
        # Risk metrics
        risk_calc = RiskMetricsCalculator(
            trade_returns,
            self.config.confidence_level,
            self.config.ruin_threshold
        )
        
        ruin_prob = risk_calc.compute_ruin_probability()
        var_95 = risk_calc.compute_var()
        cvar_95 = risk_calc.compute_cvar()
        max_dd_dist = risk_calc.compute_max_drawdown_distribution()
        expected_max_dd = max_dd_dist.mean()
        
        # Parameter sensitivity
        sensitivity_analyzer = ParameterSensitivityAnalyzer(
            model_params, n_samples=50, seed=self.config.random_seed
        )
        
        # Regime classifier testing
        if data is not None:
            regime_tester = RegimeClassifierTester(
                model_params, n_trials=50, seed=self.config.random_seed
            )
            regime_results = regime_tester.run_accuracy_test()
            regime_accuracy_dist = regime_results['overall_accuracy'].values
            regime_stability = regime_results['overall_accuracy'].std()
        else:
            # Use synthetic data
            regime_tester = RegimeClassifierTester(
                model_params, n_trials=50, seed=self.config.random_seed
            )
            regime_results = regime_tester.run_accuracy_test()
            regime_accuracy_dist = regime_results['overall_accuracy'].values
            regime_stability = 1 - regime_results['overall_accuracy'].std()
        
        return MonteCarloResult(
            param_sensitivity=pd.DataFrame(),  # Would be filled by sensitivity analysis
            return_distribution=return_dist,
            sharpe_distribution=sharpe_dist,
            return_ci=return_ci,
            sharpe_ci=sharpe_ci,
            ruin_probability=ruin_prob,
            expected_max_drawdown=expected_max_dd,
            var_95=var_95,
            cvar_95=cvar_95,
            regime_accuracy_distribution=regime_accuracy_dist,
            regime_stability=regime_stability
        )

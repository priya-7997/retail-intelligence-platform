"""
Intelligent Model Selection Engine for Retail Intelligence Platform
Automatically selects the best forecasting model based on data characteristics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelSelector:
    """Intelligent model selection based on data patterns and characteristics"""
    
    def __init__(self):
        self.model_scores = {}
        self.selection_criteria = {
            'prophet': {
                'seasonality_strength': 0.3,
                'trend_strength': 0.2,
                'data_points_min': 60,
                'missing_data_tolerance': 0.3
            },
            'xgboost': {
                'feature_count_min': 2,
                'data_points_min': 100,
                'non_linearity_threshold': 0.4,
                'missing_data_tolerance': 0.1
            },
            'arima': {
                'stationarity_threshold': 0.05,
                'data_points_min': 30,
                'autocorr_threshold': 0.3,
                'missing_data_tolerance': 0.05
            }
        }
    
    def select_best_model(self, df: pd.DataFrame, features_available: List[str] = None) -> Dict:
        """
        Analyze data and recommend the best forecasting model
        
        Args:
            df: Time series dataframe with 'ds' (date) and 'y' (values)
            features_available: List of additional feature columns
            
        Returns:
            Dictionary with model recommendation and analysis
        """
        try:
            logger.info("Analyzing data patterns for model selection...")
            
            # Ensure required columns exist
            if 'ds' not in df.columns or 'y' not in df.columns:
                raise ValueError("DataFrame must have 'ds' (date) and 'y' (value) columns")
            
            # Clean data
            df_clean = df.dropna(subset=['ds', 'y']).copy()
            df_clean = df_clean.sort_values('ds')
            
            # Analyze data characteristics
            data_analysis = self._analyze_data_characteristics(df_clean)
            
            # Score each model
            model_scores = self._score_models(data_analysis, features_available or [])
            
            # Select best model
            best_model = max(model_scores.keys(), key=lambda k: model_scores[k]['total_score'])
            
            # Generate recommendation explanation
            explanation = self._generate_explanation(best_model, model_scores, data_analysis)
            
            return {
                'recommended_model': best_model,
                'confidence': model_scores[best_model]['confidence'],
                'model_scores': model_scores,
                'data_analysis': data_analysis,
                'explanation': explanation,
                'fallback_models': self._get_fallback_order(model_scores)
            }
            
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            # Return safe default
            return {
                'recommended_model': 'prophet',
                'confidence': 0.5,
                'explanation': f"Error in analysis, defaulting to Prophet: {str(e)}",
                'model_scores': {},
                'data_analysis': {},
                'fallback_models': ['arima', 'xgboost']
            }
    
    def _analyze_data_characteristics(self, df: pd.DataFrame) -> Dict:
        """Comprehensive analysis of time series characteristics"""
        analysis = {}
        
        # Basic statistics
        analysis['data_points'] = len(df)
        analysis['date_range_days'] = (df['ds'].max() - df['ds'].min()).days
        analysis['missing_percentage'] = df['y'].isnull().sum() / len(df)
        
        # Time series properties
        y_values = df['y'].dropna()
        
        # Trend analysis
        analysis['trend_strength'] = self._calculate_trend_strength(df)
        analysis['trend_direction'] = 'increasing' if analysis['trend_strength'] > 0 else 'decreasing'
        
        # Seasonality analysis
        analysis['seasonality'] = self._detect_seasonality(df)
        
        # Stationarity test
        analysis['stationarity'] = self._test_stationarity(y_values)
        
        # Autocorrelation
        analysis['autocorrelation'] = self._calculate_autocorrelation(y_values)
        
        # Variance and volatility
        analysis['variance'] = float(y_values.var())
        analysis['coefficient_of_variation'] = float(y_values.std() / y_values.mean()) if y_values.mean() != 0 else 0
        
        # Distribution analysis
        analysis['skewness'] = float(stats.skew(y_values))
        analysis['kurtosis'] = float(stats.kurtosis(y_values))
        
        # Outlier detection
        analysis['outlier_percentage'] = self._detect_outliers(y_values)
        
        # Data frequency
        analysis['frequency'] = self._detect_frequency(df)
        
        logger.info(f"Data analysis complete: {analysis['data_points']} points, "
                   f"{analysis['seasonality']['strength']:.2f} seasonality, "
                   f"{analysis['trend_strength']:.2f} trend")
        
        return analysis
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> float:
        """Calculate the strength of the trend in the data"""
        if len(df) < 10:
            return 0.0
        
        # Linear regression on time vs values
        x = np.arange(len(df))
        y = df['y'].values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            return 0.0
        
        x_clean, y_clean = x[mask], y[mask]
        
        # Calculate correlation coefficient as trend strength
        if len(x_clean) > 1:
            correlation = np.corrcoef(x_clean, y_clean)[0, 1]
            return 0.0 if np.isnan(correlation) else abs(correlation)
        
        return 0.0
    
    def _detect_seasonality(self, df: pd.DataFrame) -> Dict:
        """Detect seasonal patterns in the data"""
        seasonality_info = {
            'strength': 0.0,
            'weekly_pattern': False,
            'monthly_pattern': False,
            'yearly_pattern': False,
            'dominant_frequency': None
        }
        
        if len(df) < 14:  # Need at least 2 weeks
            return seasonality_info
        
        # Add time components
        df_temp = df.copy()
        df_temp['day_of_week'] = df_temp['ds'].dt.dayofweek
        df_temp['day_of_month'] = df_temp['ds'].dt.day
        df_temp['month'] = df_temp['ds'].dt.month
        
        # Test weekly seasonality
        if len(df) >= 14:
            weekly_groups = df_temp.groupby('day_of_week')['y'].mean()
            weekly_var = weekly_groups.var()
            overall_var = df_temp['y'].var()
            if overall_var > 0:
                weekly_strength = weekly_var / overall_var
                seasonality_info['weekly_pattern'] = weekly_strength > 0.1
                seasonality_info['strength'] = max(seasonality_info['strength'], weekly_strength)
        
        # Test monthly seasonality
        if len(df) >= 60:  # Need at least 2 months
            monthly_groups = df_temp.groupby('day_of_month')['y'].mean()
            monthly_var = monthly_groups.var()
            if overall_var > 0:
                monthly_strength = monthly_var / overall_var
                seasonality_info['monthly_pattern'] = monthly_strength > 0.1
                seasonality_info['strength'] = max(seasonality_info['strength'], monthly_strength)
        
        # Test yearly seasonality
        if len(df) >= 365:
            yearly_groups = df_temp.groupby('month')['y'].mean()
            yearly_var = yearly_groups.var()
            if overall_var > 0:
                yearly_strength = yearly_var / overall_var
                seasonality_info['yearly_pattern'] = yearly_strength > 0.1
                seasonality_info['strength'] = max(seasonality_info['strength'], yearly_strength)
        
        # Determine dominant frequency
        if seasonality_info['yearly_pattern']:
            seasonality_info['dominant_frequency'] = 'yearly'
        elif seasonality_info['monthly_pattern']:
            seasonality_info['dominant_frequency'] = 'monthly'
        elif seasonality_info['weekly_pattern']:
            seasonality_info['dominant_frequency'] = 'weekly'
        
        return seasonality_info
    
    def _test_stationarity(self, series: pd.Series) -> Dict:
        """Test for stationarity using Augmented Dickey-Fuller test"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            # Remove NaN values
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return {'is_stationary': False, 'p_value': 1.0}
            
            result = adfuller(clean_series, autolag='AIC')
            
            return {
                'is_stationary': result[1] < 0.05,  # p-value < 0.05 indicates stationarity
                'p_value': result[1],
                'adf_statistic': result[0],
                'critical_values': result[4]
            }
        except Exception as e:
            logger.warning(f"Stationarity test failed: {e}")
            return {'is_stationary': False, 'p_value': 1.0}
    
    def _calculate_autocorrelation(self, series: pd.Series) -> float:
        """Calculate autocorrelation at lag 1"""
        try:
            clean_series = series.dropna()
            if len(clean_series) < 10:
                return 0.0
            
            return float(clean_series.autocorr(lag=1))
        except:
            return 0.0
    
    def _detect_outliers(self, series: pd.Series) -> float:
        """Detect percentage of outliers using IQR method"""
        try:
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = series[(series < lower_bound) | (series > upper_bound)]
            return len(outliers) / len(series)
        except:
            return 0.0
    
    def _detect_frequency(self, df: pd.DataFrame) -> str:
        """Detect the frequency of the time series"""
        if len(df) < 2:
            return 'unknown'
        
        # Calculate median time difference
        time_diffs = df['ds'].diff().dropna()
        median_diff = time_diffs.median()
        
        if median_diff <= pd.Timedelta(days=1):
            return 'daily'
        elif median_diff <= pd.Timedelta(days=7):
            return 'weekly'
        elif median_diff <= pd.Timedelta(days=31):
            return 'monthly'
        else:
            return 'irregular'
    
    def _score_models(self, analysis: Dict, features: List[str]) -> Dict:
        """Score each model based on data characteristics"""
        scores = {
            'prophet': {'scores': {}, 'total_score': 0, 'confidence': 0},
            'xgboost': {'scores': {}, 'total_score': 0, 'confidence': 0},
            'arima': {'scores': {}, 'total_score': 0, 'confidence': 0}
        }
        
        # Prophet scoring
        prophet_score = 0
        
        # Seasonality favor
        seasonality_strength = analysis.get('seasonality', {}).get('strength', 0)
        prophet_score += min(40, seasonality_strength * 100)
        scores['prophet']['scores']['seasonality'] = seasonality_strength * 100
        
        # Trend favor
        trend_strength = analysis.get('trend_strength', 0)
        prophet_score += min(20, trend_strength * 50)
        scores['prophet']['scores']['trend'] = trend_strength * 50
        
        # Data points
        data_points = analysis.get('data_points', 0)
        if data_points >= 60:
            prophet_score += 20
        elif data_points >= 30:
            prophet_score += 10
        scores['prophet']['scores']['data_sufficiency'] = min(20, data_points / 3)
        
        # Missing data tolerance
        missing_pct = analysis.get('missing_percentage', 0)
        if missing_pct <= 0.3:
            prophet_score += 20
        scores['prophet']['scores']['data_quality'] = max(0, 20 * (1 - missing_pct / 0.3))
        
        scores['prophet']['total_score'] = prophet_score
        scores['prophet']['confidence'] = min(1.0, prophet_score / 100)
        
        # XGBoost scoring
        xgboost_score = 0
        
        # Feature availability
        feature_count = len(features)
        if feature_count >= 3:
            xgboost_score += 40
        elif feature_count >= 1:
            xgboost_score += 20
        scores['xgboost']['scores']['features'] = min(40, feature_count * 13)
        
        # Data points (needs more data)
        if data_points >= 100:
            xgboost_score += 25
        elif data_points >= 50:
            xgboost_score += 15
        scores['xgboost']['scores']['data_sufficiency'] = min(25, data_points / 4)
        
        # Non-linearity (high variance favors XGBoost)
        cv = analysis.get('coefficient_of_variation', 0)
        if cv > 0.5:
            xgboost_score += 20
        elif cv > 0.2:
            xgboost_score += 10
        scores['xgboost']['scores']['complexity'] = min(20, cv * 40)
        
        # Data quality requirement (stricter)
        if missing_pct <= 0.1:
            xgboost_score += 15
        elif missing_pct <= 0.2:
            xgboost_score += 5
        scores['xgboost']['scores']['data_quality'] = max(0, 15 * (1 - missing_pct / 0.1))
        
        scores['xgboost']['total_score'] = xgboost_score
        scores['xgboost']['confidence'] = min(1.0, xgboost_score / 100)
        
        # ARIMA scoring
        arima_score = 0
        
        # Stationarity favor
        stationarity = analysis.get('stationarity', {})
        if stationarity.get('is_stationary', False):
            arima_score += 30
        else:
            arima_score += 10  # Can still work with differencing
        scores['arima']['scores']['stationarity'] = 30 if stationarity.get('is_stationary') else 10
        
        # Autocorrelation
        autocorr = abs(analysis.get('autocorrelation', 0))
        arima_score += min(25, autocorr * 50)
        scores['arima']['scores']['autocorrelation'] = autocorr * 50
        
        # Simplicity favor (less data points)
        if data_points <= 100:
            arima_score += 20
        elif data_points <= 200:
            arima_score += 10
        scores['arima']['scores']['simplicity'] = max(0, 20 * (1 - max(0, data_points - 50) / 150))
        
        # Data quality (strictest requirement)
        if missing_pct <= 0.05:
            arima_score += 25
        elif missing_pct <= 0.1:
            arima_score += 10
        scores['arima']['scores']['data_quality'] = max(0, 25 * (1 - missing_pct / 0.05))
        
        scores['arima']['total_score'] = arima_score
        scores['arima']['confidence'] = min(1.0, arima_score / 100)
        
        return scores
    
    def _generate_explanation(self, best_model: str, scores: Dict, analysis: Dict) -> str:
        """Generate human-readable explanation for model selection"""
        explanations = {
            'prophet': [
                f"Strong seasonality detected ({analysis.get('seasonality', {}).get('strength', 0):.2f})",
                f"Trend strength: {analysis.get('trend_strength', 0):.2f}",
                f"Handles missing data well ({analysis.get('missing_percentage', 0)*100:.1f}% missing)",
                "Excellent for retail sales with seasonal patterns"
            ],
            'xgboost': [
                "Multiple features available for complex modeling",
                f"High data variability (CV: {analysis.get('coefficient_of_variation', 0):.2f})",
                f"Sufficient data points ({analysis.get('data_points', 0)})",
                "Best for non-linear patterns and feature interactions"
            ],
            'arima': [
                f"Good autocorrelation ({analysis.get('autocorrelation', 0):.2f})",
                f"Stationarity: {'Yes' if analysis.get('stationarity', {}).get('is_stationary') else 'No'}",
                "Simple and interpretable model",
                "Ideal for clean, stable time series"
            ]
        }
        
        base_explanation = f"**{best_model.upper()} selected** (confidence: {scores[best_model]['confidence']:.1%})\n\n"
        base_explanation += "**Reasons:**\n"
        for reason in explanations[best_model]:
            base_explanation += f"• {reason}\n"
        
        return base_explanation
    
    def _get_fallback_order(self, scores: Dict) -> List[str]:
        """Get fallback models in order of preference"""
        sorted_models = sorted(scores.keys(), key=lambda k: scores[k]['total_score'], reverse=True)
        return sorted_models[1:]  # Return all except the best one

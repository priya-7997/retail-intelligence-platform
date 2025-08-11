"""
Three-Model Forecasting Engine for Retail Intelligence Platform
Implements Prophet, XGBoost, and ARIMA models with automatic selection
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
import warnings
import joblib
from pathlib import Path
import json

# Model imports
try:
    from prophet import Prophet
    from prophet.diagnostics import cross_validation, performance_metrics
except ImportError:
    Prophet = None

import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
except ImportError:
    ARIMA = None

from config import settings, format_currency

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class ForecastingEngine:
    """Unified forecasting engine with three models"""
    
    def __init__(self):
        self.models = {}
        self.model_metadata = {}
        self.indian_holidays = self._get_indian_holidays()
        self.models_dir = Path('backend/models/saved_models')
        self.metadata_path = self.models_dir / 'model_metadata.json'
        self._load_models_and_metadata()

    def _load_models_and_metadata(self):
        self.models_dir.mkdir(parents=True, exist_ok=True)
        # Load metadata
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r') as f:
                    self.model_metadata = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load model metadata: {e}")
                self.model_metadata = {}
        # Load models
        for model_id, meta in self.model_metadata.items():
            model_file = self.models_dir / f'{model_id}.pkl'
            if model_file.exists():
                try:
                    if meta.get('model_type') == 'xgboost':
                        # XGBoost: load dict with model, feature_columns, etc.
                        self.models[model_id] = joblib.load(model_file)
                    else:
                        # Prophet/ARIMA: load model object
                        self.models[model_id] = joblib.load(model_file)
                except Exception as e:
                    logger.error(f"Failed to load model {model_id}: {e}")

    def _save_model_and_metadata(self, model_id, model, model_type=None):
        # Save model
        model_file = self.models_dir / f'{model_id}.pkl'
        try:
            if model_type == 'xgboost' or (model_type is None and model_id.startswith('xgboost_')):
                # Save the full dict for XGBoost
                joblib.dump(model, model_file)
            else:
                # Save just the model object for Prophet/ARIMA
                joblib.dump(model, model_file)
        except Exception as e:
            logger.error(f"Failed to save model {model_id}: {e}")
        # Save metadata
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(self.model_metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model metadata: {e}")
        
    def _get_indian_holidays(self) -> pd.DataFrame:
        """Get Indian holidays for Prophet model"""
        holidays = pd.DataFrame({
            'holiday': [
                'Republic Day', 'Holi', 'Independence Day', 'Gandhi Jayanti', 
                'Diwali', 'Eid', 'Christmas', 'Dussehra'
            ],
            'ds': pd.to_datetime([
                '2024-01-26', '2024-03-25', '2024-08-15', '2024-10-02',
                '2024-11-01', '2024-04-10', '2024-12-25', '2024-10-12'
            ]),
            'lower_window': 0,
            'upper_window': 1,
        })
        return holidays
    
    def train_prophet(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train Prophet model for seasonal forecasting"""
        try:
            if Prophet is None:
                raise ImportError("Prophet not installed")
            
            logger.info("Training Prophet model...")
            
            # Prepare data
            train_df = df[['ds', 'y']].copy()
            train_df = train_df.dropna()
            
            # Configure Prophet with Indian market settings
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                holidays=self.indian_holidays,
                seasonality_mode='multiplicative',  # Better for sales data
                changepoint_prior_scale=0.05,  # Detect trend changes
                seasonality_prior_scale=10.0,  # Strong seasonality
                interval_width=0.8,  # 80% confidence intervals
                **kwargs
            )
            
            # Add custom seasonalities for Indian retail
            model.add_seasonality(
                name='monthly',
                period=30.5,
                fourier_order=5
            )
            
            # Fit the model
            model.fit(train_df)
            
            # Store model
            model_id = f"prophet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.models[model_id] = model
            self._save_model_and_metadata(model_id, model)
            
            # Calculate training metrics
            train_forecast = model.predict(train_df)
            metrics = self._calculate_metrics(train_df['y'], train_forecast['yhat'])
            
            # Store metadata
            self.model_metadata[model_id] = {
                'model_type': 'prophet',
                'training_samples': len(train_df),
                'features': ['ds', 'y'],
                'metrics': metrics,
                'trained_at': datetime.now().isoformat(),
                'parameters': {
                    'yearly_seasonality': True,
                    'weekly_seasonality': True,
                    'seasonality_mode': 'multiplicative'
                }
            }
            
            logger.info(f"Prophet model trained successfully. MAE: {metrics['mae']:.2f}")
            
            return {
                'success': True,
                'model_id': model_id,
                'metrics': metrics,
                'model': model
            }
            
        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_id': None
            }
    
    def train_xgboost(self, df: pd.DataFrame, features: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Train XGBoost model for complex pattern recognition"""
        try:
            logger.info("Training XGBoost model...")
            
            # Feature engineering
            df_features = self._engineer_features(df, features)
            
            # Prepare training data
            feature_cols = [col for col in df_features.columns if col not in ['ds', 'y']]
            if not feature_cols:
                raise ValueError("No features available for XGBoost training")
            
            X = df_features[feature_cols]
            y = df_features['y']
            
            # Remove rows with missing target
            mask = ~y.isnull()
            X, y = X[mask], y[mask]
            
            if len(X) < 20:
                raise ValueError("Insufficient data for XGBoost training")
            
            # Train-test split for time series
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            # Configure XGBoost
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                **kwargs
            )
            
            # Train model
            model.fit(
                X_train, y_train,
                eval_set=[(X_test, y_test)],
                verbose=False
            )
            
            # Store model
            model_id = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            xgb_dict = {
                'model': model,
                'feature_columns': feature_cols,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            self.models[model_id] = xgb_dict
            # Save the full dict for XGBoost
            self._save_model_and_metadata(model_id, xgb_dict, model_type='xgboost')
            
            # Calculate metrics
            y_pred = model.predict(X_test)
            metrics = self._calculate_metrics(y_test, y_pred)
            
            # Store metadata
            self.model_metadata[model_id] = {
                'model_type': 'xgboost',
                'training_samples': len(X_train),
                'features': feature_cols,
                'metrics': metrics,
                'trained_at': datetime.now().isoformat(),
                'feature_importance': dict(zip(feature_cols, model.feature_importances_))
            }
            
            logger.info(f"XGBoost model trained successfully. MAE: {metrics['mae']:.2f}")
            
            return {
                'success': True,
                'model_id': model_id,
                'metrics': metrics,
                'feature_importance': dict(zip(feature_cols, model.feature_importances_)),
                'model': model
            }
            
        except Exception as e:
            logger.error(f"XGBoost training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_id': None
            }
    
    def train_arima(self, df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
        """Train ARIMA model for traditional time series forecasting"""
        try:
            if ARIMA is None:
                raise ImportError("statsmodels not installed")
            
            logger.info("Training ARIMA model...")
            
            # Prepare data
            series = df.set_index('ds')['y'].dropna()
            
            if len(series) < 30:
                raise ValueError("Insufficient data for ARIMA training")
            
            # Auto-detect ARIMA parameters
            best_order = self._auto_arima_order(series)
            
            # Train ARIMA model
            model = ARIMA(series, order=best_order)
            fitted_model = model.fit()
            
            # Store model
            model_id = f"arima_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.models[model_id] = fitted_model
            self._save_model_and_metadata(model_id, fitted_model)
            
            # Calculate metrics on fitted values
            fitted_values = fitted_model.fittedvalues
            metrics = self._calculate_metrics(series, fitted_values)
            
            # Store metadata
            self.model_metadata[model_id] = {
                'model_type': 'arima',
                'training_samples': len(series),
                'features': ['y'],
                'metrics': metrics,
                'trained_at': datetime.now().isoformat(),
                'parameters': {
                    'order': best_order,
                    'aic': fitted_model.aic,
                    'bic': fitted_model.bic
                }
            }
            
            logger.info(f"ARIMA{best_order} model trained successfully. MAE: {metrics['mae']:.2f}")
            
            return {
                'success': True,
                'model_id': model_id,
                'metrics': metrics,
                'order': best_order,
                'model': fitted_model
            }
            
        except Exception as e:
            logger.error(f"ARIMA training failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_id': None
            }
    
    def _engineer_features(self, df: pd.DataFrame, additional_features: List[str] = None) -> pd.DataFrame:
        """Engineer time-based features for XGBoost"""
        df_features = df.copy()
        
        # Ensure datetime index
        df_features['ds'] = pd.to_datetime(df_features['ds'])
        
        # Time-based features
        df_features['year'] = df_features['ds'].dt.year
        df_features['month'] = df_features['ds'].dt.month
        df_features['day'] = df_features['ds'].dt.day
        df_features['dayofweek'] = df_features['ds'].dt.dayofweek
        df_features['dayofyear'] = df_features['ds'].dt.dayofyear
        df_features['week'] = df_features['ds'].dt.isocalendar().week
        df_features['quarter'] = df_features['ds'].dt.quarter
        
        # Cyclical encoding for better learning
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
        df_features['day_sin'] = np.sin(2 * np.pi * df_features['dayofweek'] / 7)
        df_features['day_cos'] = np.cos(2 * np.pi * df_features['dayofweek'] / 7)
        
        # Lag features
        for lag in [1, 7, 30]:
            df_features[f'lag_{lag}'] = df_features['y'].shift(lag)
        
        # Rolling statistics
        for window in [7, 30]:
            df_features[f'rolling_mean_{window}'] = df_features['y'].rolling(window).mean()
            df_features[f'rolling_std_{window}'] = df_features['y'].rolling(window).std()
        
        # Indian market specific features
        df_features['is_weekend'] = df_features['dayofweek'].isin([5, 6])  # Sat, Sun
        df_features['is_month_end'] = df_features['day'] >= 28
        df_features['is_quarter_end'] = df_features['month'].isin([3, 6, 9, 12]) & (df_features['day'] >= 28)
        
        # Holiday proximity (simplified)
        holiday_months = [1, 3, 8, 10, 11]  # Major Indian holidays
        df_features['holiday_month'] = df_features['month'].isin(holiday_months)
        
        # Include additional features if provided
        if additional_features:
            for feature in additional_features:
                if feature in df.columns:
                    df_features[feature] = df[feature]
        
        return df_features
    
    def _auto_arima_order(self, series: pd.Series) -> Tuple[int, int, int]:
        """Automatically determine best ARIMA order using grid search"""
        best_aic = float('inf')
        best_order = (1, 1, 1)
        
        # Grid search for optimal parameters
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = ARIMA(series, order=(p, d, q))
                        fitted = model.fit()
                        if fitted.aic < best_aic:
                            best_aic = fitted.aic
                            best_order = (p, d, q)
                    except:
                        continue
        
        return best_order
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
        """Calculate forecasting performance metrics"""
        # Align series
        if isinstance(y_true, pd.Series) and isinstance(y_pred, pd.Series):
            aligned = pd.concat([y_true, y_pred], axis=1, join='inner')
            y_true_clean = aligned.iloc[:, 0]
            y_pred_clean = aligned.iloc[:, 1]
        else:
            y_true_clean = np.array(y_true)
            y_pred_clean = np.array(y_pred)
        
        # Remove NaN values
        mask = ~(np.isnan(y_true_clean) | np.isnan(y_pred_clean))
        y_true_clean = y_true_clean[mask]
        y_pred_clean = y_pred_clean[mask]
        
        if len(y_true_clean) == 0:
            return {'mae': float('inf'), 'rmse': float('inf'), 'mape': float('inf'), 'r2': -1}
        
        # Calculate metrics
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        
        # MAPE calculation
        mape = np.mean(np.abs((y_true_clean - y_pred_clean) / np.where(y_true_clean != 0, y_true_clean, 1))) * 100
        
        # R² score
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return {
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'r2': float(r2)
        }
    
    def forecast(self, model_id: str, periods: int = 30) -> Dict[str, Any]:
        """Generate forecasts using trained model"""
        try:
            logger.info(f"Forecast request: model_id={model_id}, periods={periods}")
            if model_id not in self.models:
                logger.error(f"Model {model_id} not found in self.models")
                raise ValueError(f"Model {model_id} not found")

            model_type = self.model_metadata[model_id]['model_type']
            logger.info(f"Selected model_type: {model_type}")

            if model_type == 'prophet':
                logger.info("Calling _forecast_prophet...")
                result = self._forecast_prophet(model_id, periods)
            elif model_type == 'xgboost':
                logger.info("Calling _forecast_xgboost...")
                result = self._forecast_xgboost(model_id, periods)
            elif model_type == 'arima':
                logger.info("Calling _forecast_arima...")
                result = self._forecast_arima(model_id, periods)
            else:
                logger.error(f"Unknown model type: {model_type}")
                raise ValueError(f"Unknown model type: {model_type}")

            logger.info(f"Forecast result keys: {list(result.keys())}")
            if not result.get('success'):
                logger.error(f"Model prediction failed: {result.get('error')}")
            return result
        except Exception as e:
            logger.error(f"Forecasting failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _forecast_prophet(self, model_id: str, periods: int) -> Dict[str, Any]:
        """Generate Prophet forecast"""
        model = self.models[model_id]
        logger.info(f"Prophet model: {model}")
        future = model.make_future_dataframe(periods=periods)
        logger.info(f"Future dataframe shape: {future.shape}, columns: {future.columns.tolist()}")
        forecast = model.predict(future)
        logger.info(f"Forecast dataframe shape: {forecast.shape}, columns: {forecast.columns.tolist()}")
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
        logger.info(f"Prophet forecast data shape: {forecast_data.shape}, NaNs: {forecast_data.isnull().sum().to_dict()}")
        return {
            'success': True,
            'forecast': forecast_data.to_dict('records'),
            'model_type': 'prophet',
            'periods': periods
        }
    
    def _forecast_xgboost(self, model_id: str, periods: int) -> Dict[str, Any]:
        """Generate XGBoost forecast"""
        model_info = self.models[model_id]
        model = model_info['model']
        feature_cols = model_info['feature_columns']
        logger.info(f"XGBoost model: {model}, feature_cols: {feature_cols}")
        forecast_dates = pd.date_range(
            start=datetime.now().date(),
            periods=periods,
            freq='D'
        )
        future_df = pd.DataFrame({'ds': forecast_dates})
        future_features = self._engineer_features(future_df)
        logger.info(f"Future features shape: {future_features.shape}, columns: {future_features.columns.tolist()}")
        available_features = [col for col in feature_cols if col in future_features.columns]
        X_future = future_features[available_features].fillna(0)
        logger.info(f"X_future shape: {X_future.shape}, NaNs: {X_future.isnull().sum().to_dict()}")
        try:
            predictions = model.predict(X_future)
            logger.info(f"XGBoost predictions shape: {predictions.shape if hasattr(predictions, 'shape') else len(predictions)}")
        except Exception as e:
            logger.error(f"XGBoost prediction error: {e}")
            return {'success': False, 'error': str(e)}
        forecast_data = []
        for i, (date, pred) in enumerate(zip(forecast_dates, predictions)):
            forecast_data.append({
                'ds': date.strftime('%Y-%m-%d'),
                'yhat': float(pred),
                'yhat_lower': float(pred * 0.9),
                'yhat_upper': float(pred * 1.1)
            })
        logger.info(f"XGBoost forecast data length: {len(forecast_data)}")
        return {
            'success': True,
            'forecast': forecast_data,
            'model_type': 'xgboost',
            'periods': periods
        }
    
    def _forecast_arima(self, model_id: str, periods: int) -> Dict[str, Any]:
        """Generate ARIMA forecast"""
        model = self.models[model_id]
        logger.info(f"ARIMA model: {model}")
        try:
            forecast_result = model.forecast(steps=periods)
            conf_int = model.get_forecast(steps=periods).conf_int()
            logger.info(f"ARIMA forecast_result shape: {getattr(forecast_result, 'shape', len(forecast_result))}, conf_int shape: {conf_int.shape}")
        except Exception as e:
            logger.error(f"ARIMA prediction error: {e}")
            return {'success': False, 'error': str(e)}
        last_date = model.data.dates[-1] if hasattr(model.data, 'dates') else datetime.now()
        forecast_dates = pd.date_range(
            start=last_date + timedelta(days=1),
            periods=periods,
            freq='D'
        )
        forecast_data = []
        for i, date in enumerate(forecast_dates):
            forecast_data.append({
                'ds': date.strftime('%Y-%m-%d'),
                'yhat': float(forecast_result[i]),
                'yhat_lower': float(conf_int.iloc[i, 0]),
                'yhat_upper': float(conf_int.iloc[i, 1])
            })
        logger.info(f"ARIMA forecast data length: {len(forecast_data)}")
        return {
            'success': True,
            'forecast': forecast_data,
            'model_type': 'arima',
            'periods': periods
        }
    
    def save_model(self, model_id: str, filepath: str = None) -> str:
        """Save trained model to disk"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        if filepath is None:
            filepath = settings.MODEL_SAVE_DIR / f"{model_id}.pkl"
        else:
            filepath = Path(filepath)
        
        # Save model and metadata
        model_data = {
            'model': self.models[model_id],
            'metadata': self.model_metadata[model_id]
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model {model_id} saved to {filepath}")
        
        return str(filepath)
    
    def load_model(self, filepath: str) -> str:
        """Load trained model from disk"""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        model_data = joblib.load(filepath)
        
        # Generate new model ID
        model_id = f"loaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.models[model_id] = model_data['model']
        self.model_metadata[model_id] = model_data['metadata']
        
        logger.info(f"Model loaded as {model_id}")
        return model_id
    
    def compare_models(self, model_ids: List[str]) -> Dict[str, Any]:
        """Compare performance of multiple models"""
        comparison = {
            'models': {},
            'best_model': None,
            'ranking': []
        }
        
        for model_id in model_ids:
            if model_id in self.model_metadata:
                metadata = self.model_metadata[model_id]
                comparison['models'][model_id] = {
                    'type': metadata['model_type'],
                    'metrics': metadata['metrics'],
                    'training_samples': metadata['training_samples']
                }
        
        # Rank by MAE (lower is better)
        ranking = sorted(
            comparison['models'].items(),
            key=lambda x: x[1]['metrics'].get('mae', float('inf'))
        )
        
        comparison['ranking'] = [model_id for model_id, _ in ranking]
        comparison['best_model'] = ranking[0][0] if ranking else None
        
        return comparison
__all__ = ["router"]
def custom_json_response(content, status_code=200):
    return JSONResponse(content=json.loads(json.dumps(content, cls=CustomJSONEncoder)), status_code=status_code)
import json
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (pd.Timestamp, np.datetime64)):
            return str(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if hasattr(obj, 'isoformat'):
            try:
                return obj.isoformat()
            except Exception:
                return str(obj)
        return super().default(obj)

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

class AutoForecastRequest(BaseModel):
    file_id: str = Field(..., description="Processed file ID")
    forecast_periods: int = Field(30, description="Number of periods to forecast", ge=1, le=365)
from typing import Dict, List, Optional, Any
import json
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import json

from config import settings
from core.forecasting import ForecastingEngine
from core.model_selector import ModelSelector
from core import forecasting
from core.data_processor import RetailDataProcessor

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize engines
forecasting_engine = ForecastingEngine()
model_selector = ModelSelector()
data_processor = RetailDataProcessor()

# Store training results temporarily
training_results = {}
forecast_cache = {}

class ForecastRequest(BaseModel):
    file_id: str = Field(..., description="Processed file ID")
    model_type: Optional[str] = Field(None, description="Specific model to use (prophet/xgboost/arima)")
    forecast_periods: int = Field(30, description="Number of periods to forecast", ge=1, le=365)
    frequency: str = Field("daily", description="Forecast frequency")
    
class TrainingRequest(BaseModel):
    file_id: str = Field(..., description="Processed file ID")
    model_type: str = Field(..., description="Model type to train")
    parameters: Optional[Dict] = Field(None, description="Custom model parameters")

@router.post("/train")
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> JSONResponse:
    """
    
    Args:
        request: Training request with file ID and model type
        background_tasks: BackgroundTasks
    """
    try:
        # Validate file exists
        from api.upload import processing_results
        if request.file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        file_result = processing_results[request.file_id]
        if file_result.get("status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")
        # Generate training job ID
        import uuid
        job_id = str(uuid.uuid4())
        # Initialize training status
        training_results[job_id] = {
            "status": "started",
            "progress": 0,
            "file_id": request.file_id,
            "model_type": request.model_type,
            "started_at": datetime.now().isoformat()
        }
        # Start training in background
        background_tasks.add_task(
            train_model_background,
            job_id,
            request.file_id,
            request.model_type,
            request.parameters or {}
        )
        return JSONResponse(
            status_code=202,
            content={
                "success": True,
                "message": "Training started",
                "job_id": job_id,
                "model_type": request.model_type,
                "status": "training"
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training initiation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/train/status/{job_id}")
async def get_training_status(job_id: str) -> JSONResponse:
    """
    
    Args:
        job_id: Training job identifier
        
    Returns:
        Training status and progress
    """
    if job_id not in training_results:
        raise HTTPException(status_code=404, detail="Training job not found")

    result = training_results[job_id]

    return JSONResponse(content={
        "success": True,
        "job_id": job_id,
        "status": result.get("status"),
        "progress": result.get("progress", 0),
        "model_type": result.get("model_type"),
        "started_at": result.get("started_at"),
        "completed_at": result.get("completed_at"),
        "error": result.get("error"),
        "model_id": result.get("model_id")
    })

@router.post("/predict")
async def generate_forecast(request: ForecastRequest) -> JSONResponse:
    try:
        # Check cache first
        cache_key = f"{request.file_id}_{request.model_type}_{request.forecast_periods}"
        if cache_key in forecast_cache:
            cached_result = forecast_cache[cache_key]
            cache_time = datetime.fromisoformat(cached_result["generated_at"])
            if (datetime.now() - cache_time).seconds < 3600:
                logger.info(f"Returning cached forecast for {cache_key}")
                return JSONResponse(content=cached_result)

        from api.upload import processing_results
        if request.file_id not in processing_results:
            logger.error(f"File ID {request.file_id} not found in processing_results")
            raise HTTPException(status_code=404, detail="File not found")
        file_result = processing_results[request.file_id]
        if file_result.get("status") != "completed":
            logger.error(f"File ID {request.file_id} processing not completed")
            raise HTTPException(status_code=400, detail="File processing not completed")
        processed_path = file_result["results"]["processed_file_path"]
        df = pd.read_csv(processed_path)
        column_analysis = file_result["results"]["column_analysis"]
        # Ensure 'y' column exists for forecasting
        if 'y' not in df.columns:
            # Try to map common sales columns to 'y'
            for candidate in ['sales', 'amount', 'value', 'revenue']:
                if candidate in df.columns:
                    df['y'] = df[candidate]
                    break
        # If still no 'y', raise error
        if 'y' not in df.columns:
            logger.error("No 'y' or sales column found in uploaded data.")
            return JSONResponse(content={
                "success": False,
                "message": "No 'y' or sales column found in uploaded data. Please ensure your file has a 'y' or 'sales' column.",
                "available_columns": list(df.columns)
            }, status_code=400)
        df_agg = data_processor.aggregate_for_forecasting(df, column_analysis, request.frequency)
        if not request.model_type:
            model_selection = model_selector.select_best_model(df_agg)
            selected_model = model_selection["recommended_model"]
            selection_confidence = model_selection["confidence"]
        else:
            selected_model = request.model_type
            selection_confidence = 1.0

        # Find a trained model for the selected type
        model_id = None
        for mid, meta in forecasting_engine.model_metadata.items():
            if meta["model_type"] == selected_model and meta.get("trained_at"):
                model_id = mid
                break
        if not model_id:
            logger.error(f"No trained {selected_model} model found in model_metadata. Available: {[meta['model_type'] for meta in forecasting_engine.model_metadata.values()]}")
            return JSONResponse(content={
                "success": False,
                "message": f"No trained {selected_model} model found. Please train the model first using /train endpoint.",
                "model_type": selected_model,
                "available_models": [meta['model_type'] for meta in forecasting_engine.model_metadata.values()]
            }, status_code=400)

        forecast_result = forecasting_engine.forecast(model_id, request.forecast_periods)
        if not forecast_result["success"]:
            logger.error(f"Forecasting failed for model_id {model_id}: {forecast_result.get('error')}")
            return JSONResponse(content={
                "success": False,
                "error": forecast_result.get('error'),
                "message": f"Forecasting failed: {forecast_result.get('error')}",
                "model_type": selected_model,
                "model_id": model_id
            }, status_code=500)

        def serialize_forecast(forecast):
            for row in forecast:
                # Convert date
                if "ds" in row and not isinstance(row["ds"], str):
                    try:
                        row["ds"] = row["ds"].strftime("%Y-%m-%d")
                    except Exception:
                        row["ds"] = str(row["ds"])
                # Sanitize all float values
                for k, v in row.items():
                    if isinstance(v, float):
                        if np.isnan(v) or np.isinf(v):
                            row[k] = None
            return forecast

        response_data = {
            "success": True,
            "forecast": serialize_forecast(forecast_result["forecast"]),
            "model_used": selected_model,
            "model_id": model_id,
            "selection_confidence": selection_confidence,
            "training_metrics": forecasting_engine.model_metadata[model_id].get("metrics", {}),
            "forecast_periods": request.forecast_periods,
            "frequency": request.frequency,
            "generated_at": datetime.now().isoformat(),
            "model_performance": {
                "mae": forecasting_engine.model_metadata[model_id].get("metrics", {}).get("mae"),
                "rmse": forecasting_engine.model_metadata[model_id].get("metrics", {}).get("rmse"),
                "r2": forecasting_engine.model_metadata[model_id].get("metrics", {}).get("r2")
            }
        }
        # Add feature importance for XGBoost
        if selected_model == "xgboost" and "feature_importance" in forecasting_engine.model_metadata[model_id]:
            response_data["feature_importance"] = forecasting_engine.model_metadata[model_id]["feature_importance"]
        # Cache result
        forecast_cache[cache_key] = response_data
        logger.info(f"Forecast generated successfully for model_id {model_id}")
        return JSONResponse(content=response_data)
    except HTTPException as he:
        logger.error(f"Forecasting HTTPException: {he.detail}")
        raise
    except Exception as e:
        logger.error(f"Forecasting error: {e}")
        return JSONResponse(content={
            "success": False,
            "error": str(e),
            "message": f"Forecasting failed: {str(e)}"
        }, status_code=500)
        # Add feature importance for XGBoost
        if selected_model == "xgboost" and "feature_importance" in forecasting_engine.model_metadata[model_id]:
            response_data["feature_importance"] = forecasting_engine.model_metadata[model_id]["feature_importance"]
        # Cache result
        forecast_cache[cache_key] = response_data
        return JSONResponse(content=response_data)
        raise
    except Exception as e:
        error_msg = str(e) if str(e) else f"Unknown error: {type(e).__name__}"
        logger.error(f"Forecasting error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@router.post("/auto-forecast")
async def auto_forecast(request: AutoForecastRequest) -> JSONResponse:
    file_id = request.file_id
    forecast_periods = request.forecast_periods
    try:
        # Validate file
        from api.upload import processing_results
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_result = processing_results[file_id]
        if file_result.get("status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")
        
        # Load and prepare data
        processed_path = file_result["results"]["processed_file_path"]
        df = pd.read_csv(processed_path)
        column_analysis = file_result["results"]["column_analysis"]
        df_agg = data_processor.aggregate_for_forecasting(df, column_analysis, "daily")
        
        # Get model recommendation
        model_selection = model_selector.select_best_model(df_agg)
        recommended_model = model_selection["recommended_model"]
        
        logger.info(f"Auto-selected model: {recommended_model} (confidence: {model_selection['confidence']:.2f})")
        
        # Try recommended model first, fallback if needed
        models_to_try = [recommended_model] + model_selection.get("fallback_models", [])
        forecast_result = None

        # Use ForecastingEngine to train and forecast
        engine = forecasting.ForecastingEngine()
        train_methods = {
            "arima": engine.train_arima,
            "prophet": engine.train_prophet,
            "xgboost": engine.train_xgboost,
        }
        forecast_result = None
        model_id = None
        for model in models_to_try:
            try:
                train_func = train_methods.get(model)
                if not train_func:
                    continue
                train_out = train_func(df_agg)
                if not train_out.get("success"):
                    continue
                model_id = train_out["model_id"]
                forecast_out = engine.forecast(model_id, forecast_periods)
                if forecast_out.get("success"):
                    forecast_result = forecast_out
                    break
            except Exception as e:
                logger.warning(f"Model {model} failed: {e}")
                continue

        # Compute forecast summary for frontend
        def compute_forecast_summary(forecast):
            if not forecast or not isinstance(forecast, list):
                return {}
            try:
                import numpy as np
                yhat = np.array([row.get("yhat", 0) for row in forecast])
                total = float(np.sum(yhat))
                avg = float(np.mean(yhat))
                trend = "up" if yhat[-1] > yhat[0] else ("down" if yhat[-1] < yhat[0] else "stable")
                period_days = len(yhat)
                return {
                    "total_forecasted_revenue": round(total, 2),
                    "average_daily_revenue": round(avg, 2),
                    "trend_direction": trend,
                    "forecast_period_days": period_days
                }
            except Exception:
                return {}

        forecast_data = forecast_result["forecast"] if forecast_result and "forecast" in forecast_result else []
        summary = compute_forecast_summary(forecast_data)
        response_data = {
            "success": bool(forecast_result is not None),
            "forecast": forecast_data,
            "forecast_summary": summary,
            "model_used": recommended_model,
            "selection_confidence": model_selection.get("confidence", 0.8),
            "training_metrics": forecast_result.get("metrics", {}) if forecast_result else {},
        }
        return custom_json_response(response_data)
        for model_type in models_to_try:
            try:
                logger.info(f"Attempting forecast with {model_type}")
                
                # Train model
                if model_type == "prophet":
                    training_result = forecasting_engine.train_prophet(df_agg)
                elif model_type == "xgboost":
                    training_result = forecasting_engine.train_xgboost(df_agg)
                elif model_type == "arima":
                    training_result = forecasting_engine.train_arima(df_agg)
                else:
                    continue
                
                if training_result["success"]:
                    # Generate forecast
                    model_id = training_result["model_id"]
                    forecast_result = forecasting_engine.forecast(model_id, forecast_periods)
                    
                    if forecast_result["success"]:
                        # Success! Use this model
                        forecast_result.update({
                            "model_used": model_type,
                            "model_id": model_id,
                            "training_metrics": training_result.get("metrics", {}),
                            "model_selection": model_selection,
                            "fallback_used": model_type != recommended_model
                        })
                        break
                        
            except Exception as e:
                logger.warning(f"Model {model_type} failed: {e}")
                continue
        
        if not forecast_result or not forecast_result["success"]:
            raise HTTPException(status_code=500, detail="All forecasting models failed")
        
        # Enhanced response with business insights
        response_data = {
            "success": True,
            "forecast": forecast_result["forecast"],
            "model_used": forecast_result["model_used"],
            "model_selection_reasoning": model_selection["explanation"],
            "confidence": model_selection["confidence"],
            "training_metrics": forecast_result["training_metrics"],
            "data_analysis": model_selection["data_analysis"],
            "forecast_summary": _generate_forecast_summary(forecast_result["forecast"]),
            "generated_at": datetime.now().isoformat()
        }
        
        return custom_json_response(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Auto-forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_available_models() -> JSONResponse:
    
    try:
        models_info = {
            "prophet": {
                "name": "Prophet",
                "description": "Facebook's time series forecasting tool, excellent for seasonal data",
                "best_for": [
                    "Strong seasonal patterns",
                    "Holiday effects",
                    "Missing data tolerance",
                    "Retail sales forecasting"
                ],
                "requirements": {
                    "min_data_points": 60,
                    "seasonality": "preferred",
                    "missing_data_tolerance": "high"
                },
                "outputs": ["point_forecast", "confidence_intervals", "trend_components"]
            },
            "xgboost": {
                "name": "XGBoost",
                "description": "Gradient boosting for complex pattern recognition with multiple features",
                "best_for": [
                    "Multiple input features",
                    "Non-linear patterns",
                    "Feature importance analysis",
                    "Complex business rules"
                ],
                "requirements": {
                    "min_data_points": 100,
                    "features": "multiple_preferred",
                    "missing_data_tolerance": "low"
                },
                "outputs": ["point_forecast", "feature_importance", "confidence_intervals"]
            },
            "arima": {
                "name": "ARIMA",
                "description": "Classical time series model for stationary data with autocorrelation",
                "best_for": [
                    "Stationary time series",
                    "Simple trend patterns",
                    "Quick forecasting",
                    "Traditional time series analysis"
                ],
                "requirements": {
                    "min_data_points": 30,
                    "stationarity": "preferred",
                    "missing_data_tolerance": "very_low"
                },
                "outputs": ["point_forecast", "confidence_intervals", "model_diagnostics"]
            }
        }
        
        return JSONResponse(content={
            "success": True,
            "models": models_info,
            "recommendation": "Use auto-forecast endpoint for automatic model selection",
            "selection_criteria": {
                "data_characteristics": "Seasonality, trend, stationarity",
                "data_quality": "Completeness, consistency",
                "feature_availability": "Additional variables for XGBoost",
                "business_context": "Indian retail patterns and holidays"
            }
        })
        
    except Exception as e:
        logger.error(f"Model listing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models/compare/{file_id}")
async def compare_all_models(file_id: str) -> JSONResponse:
    
    try:
        # Validate file
        from api.upload import processing_results
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_result = processing_results[file_id]
        if file_result.get("status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")
        
        # Load data
        processed_path = file_result["results"]["processed_file_path"]
        df = pd.read_csv(processed_path)
        column_analysis = file_result["results"]["column_analysis"]
        df_agg = data_processor.aggregate_for_forecasting(df, column_analysis, "daily")
        
        # Train all models
        model_results = {}
        model_ids = []
        # Prophet
        try:
            prophet_result = forecasting_engine.train_prophet(df_agg)
            if prophet_result["success"]:
                # Remove non-serializable model object
                prophet_result.pop("model", None)
                model_results["prophet"] = prophet_result
                model_ids.append(prophet_result["model_id"])
        except Exception as e:
            model_results["prophet"] = {"success": False, "error": str(e)}
        # XGBoost
        try:
            xgboost_result = forecasting_engine.train_xgboost(df_agg)
            if xgboost_result["success"]:
                xgboost_result.pop("model", None)
                model_results["xgboost"] = xgboost_result
                model_ids.append(xgboost_result["model_id"])
        except Exception as e:
            model_results["xgboost"] = {"success": False, "error": str(e)}
        # ARIMA
        try:
            arima_result = forecasting_engine.train_arima(df_agg)
            if arima_result["success"]:
                arima_result.pop("model", None)
                model_results["arima"] = arima_result
                model_ids.append(arima_result["model_id"])
        except Exception as e:
            model_results["arima"] = {"success": False, "error": str(e)}
        
        # Compare models
        if model_ids:
            comparison = forecasting_engine.compare_models(model_ids)
        else:
            comparison = {"models": {}, "best_model": None, "ranking": []}
        
        # Prepare comparison summary
        comparison_summary = []
        for model_type, result in model_results.items():
            if result["success"]:
                metrics = result.get("metrics", {})
                comparison_summary.append({
                    "model": model_type,
                    "mae": metrics.get("mae", float('inf')),
                    "rmse": metrics.get("rmse", float('inf')),
                    "r2": metrics.get("r2", -1),
                    "training_time": result.get("training_time", "N/A"),
                    "suitable_for": _get_model_suitability(model_type)
                })
            else:
                comparison_summary.append({
                    "model": model_type,
                    "error": result.get("error", "Training failed"),
                    "suitable_for": _get_model_suitability(model_type)
                })
        
        # Sort by MAE (lower is better)
        successful_models = [m for m in comparison_summary if "error" not in m]
        failed_models = [m for m in comparison_summary if "error" in m]
        successful_models.sort(key=lambda x: x["mae"])
        
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "comparison": {
                "successful_models": successful_models,
                "failed_models": failed_models,
                "recommendation": successful_models[0] if successful_models else None
            },
            "detailed_results": model_results,
            "model_ranking": comparison.get("ranking", []),
            "generated_at": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Model comparison error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/models/{model_id}")
async def delete_trained_model(model_id: str) -> JSONResponse:
    try:
        # Remove from forecasting engine
        if model_id in forecasting_engine.models:
            del forecasting_engine.models[model_id]
        
        if model_id in forecasting_engine.model_metadata:
            del forecasting_engine.model_metadata[model_id]
        
        # Remove from training results
        training_jobs_to_remove = []
        for job_id, result in training_results.items():
            if result.get("model_id") == model_id:
                training_jobs_to_remove.append(job_id)
        
        for job_id in training_jobs_to_remove:
            del training_results[job_id]
        
        # Clear forecast cache for this model
        cache_keys_to_remove = []
        for cache_key in forecast_cache.keys():
            if model_id in str(forecast_cache[cache_key]):
                cache_keys_to_remove.append(cache_key)
        
        for cache_key in cache_keys_to_remove:
            del forecast_cache[cache_key]
        
        return JSONResponse(content={
            "success": True,
            "message": f"Model {model_id} deleted successfully"
        })
        
    except Exception as e:
        logger.error(f"Model deletion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def train_model_background(job_id: str, file_id: str, model_type: str, parameters: Dict):

    try:
        # Update status
        training_results[job_id]["status"] = "loading_data"
        training_results[job_id]["progress"] = 25
        
        # Load data
        from api.upload import processing_results
        file_result = processing_results[file_id]
        processed_path = file_result["results"]["processed_file_path"]
        df = pd.read_csv(processed_path)
        column_analysis = file_result["results"]["column_analysis"]
        
        # Prepare data
        df_agg = data_processor.aggregate_for_forecasting(df, column_analysis, "daily")
        
        # Update status
        training_results[job_id]["status"] = "training"
        training_results[job_id]["progress"] = 50
        
        # Train model
        if model_type == "prophet":
            result = forecasting_engine.train_prophet(df_agg, **parameters)
        elif model_type == "xgboost":
            result = forecasting_engine.train_xgboost(df_agg, **parameters)
        elif model_type == "arima":
            result = forecasting_engine.train_arima(df_agg, **parameters)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        # Update status
        training_results[job_id]["progress"] = 100
        
        if result["success"]:
            training_results[job_id].update({
                "status": "completed",
                "model_id": result["model_id"],
                "metrics": result.get("metrics", {}),
                "completed_at": datetime.now().isoformat()
            })
            logger.info(f"Training completed successfully for job: {job_id}")
        else:
            training_results[job_id].update({
                "status": "failed",
                "error": result.get("error", "Training failed"),
                "completed_at": datetime.now().isoformat()
            })
            logger.error(f"Training failed for job: {job_id}")
        
    except Exception as e:
        training_results[job_id].update({
            "status": "failed",
            "error": str(e),
            "progress": 100,
            "completed_at": datetime.now().isoformat()
        })
        logger.error(f"Background training error for job {job_id}: {e}")

def _generate_forecast_summary(forecast_data: List[Dict]) -> Dict[str, Any]:

    try:
        if not forecast_data:
            return {"error": "No forecast data"}
        
        values = [item["yhat"] for item in forecast_data]
        
        # Basic statistics
        total_forecasted = sum(values)
        avg_daily = total_forecasted / len(values)
        max_value = max(values)
        min_value = min(values)
        
        # Trend analysis
        first_week = values[:7] if len(values) >= 7 else values
        last_week = values[-7:] if len(values) >= 7 else values
        
        first_week_avg = sum(first_week) / len(first_week)
        last_week_avg = sum(last_week) / len(last_week)
        
        trend_change = 0
        if first_week_avg > 0:
            trend_change = ((last_week_avg - first_week_avg) / first_week_avg) * 100
        
        from config import format_currency
        
        return {
            "total_forecasted_revenue": format_currency(total_forecasted),
            "average_daily_revenue": format_currency(avg_daily),
            "highest_day": format_currency(max_value),
            "lowest_day": format_currency(min_value),
            "trend_change_percent": round(trend_change, 2),
            "trend_direction": "increasing" if trend_change > 0 else "decreasing",
            "forecast_period_days": len(values),
            "confidence_level": "80%"  # Standard confidence level
        }
        
    except Exception as e:
        logger.error(f"Forecast summary error: {e}")
        return {"error": "Unable to generate forecast summary"}

def _get_model_suitability(model_type: str) -> List[str]:

    suitability = {
        "prophet": [
            "Seasonal retail patterns",
            "Holiday impact analysis",
            "Missing data handling",
            "Long-term trend forecasting"
        ],
        "xgboost": [
            "Multiple feature analysis",
            "Price elasticity modeling",
            "Promotional impact",
            "Complex pattern recognition"
        ],
        "arima": [
            "Simple trend analysis",
            "Quick forecasting",
            "Stable time series",
            "Traditional statistical approach"
        ]
    }
    return suitability.get(model_type, ["General forecasting"])
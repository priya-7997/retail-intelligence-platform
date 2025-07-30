"""
Business Insights API for Retail Intelligence Platform
Generates actionable business insights from data and forecasts
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import pandas as pd
import logging
from datetime import datetime

from config import settings
from core.insights import InsightsGenerator

router = APIRouter()
logger = logging.getLogger(__name__)

# Initialize insights generator
insights_generator = InsightsGenerator()

# Cache insights results
insights_cache = {}

class InsightsRequest(BaseModel):
    file_id: str = Field(..., description="Processed file ID")
    include_forecast: bool = Field(True, description="Include forecast-based insights")
    insight_types: Optional[List[str]] = Field(None, description="Specific insight types to generate")

@router.post("/generate")
async def generate_insights(request: InsightsRequest) -> JSONResponse:
    """
    Generate comprehensive business insights from processed data
    
    Args:
        request: Insights generation request
        
    Returns:
        Complete business insights with actionable recommendations
    """
    try:
        # Check cache first
        cache_key = f"{request.file_id}_insights"
        if cache_key in insights_cache:
            cached_result = insights_cache[cache_key]
            cache_time = datetime.fromisoformat(cached_result["generated_at"])
            # Use cache if less than 30 minutes old
            if (datetime.now() - cache_time).seconds < 1800:
                return JSONResponse(content=cached_result)
        
        # Validate file exists
        from api.upload import processing_results
        if request.file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_result = processing_results[request.file_id]
        if file_result.get("status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")
        
        # Get processed data
        processed_data = file_result["results"]
        column_analysis = processed_data["column_analysis"]
        
        # Load processed DataFrame
        processed_path = processed_data["processed_file_path"]
        df = pd.read_csv(processed_path)
        processed_data["data"] = df
        
        # Get forecast data if requested
        forecast_results = None
        if request.include_forecast:
            try:
                # Try to get recent forecast for this file
                from api.forecast import forecast_cache
                for cache_key, forecast_data in forecast_cache.items():
                    if request.file_id in cache_key:
                        forecast_results = forecast_data
                        break
            except Exception as e:
                logger.warning(f"Could not load forecast data: {e}")
        
        # Generate insights
        insights_result = insights_generator.generate_comprehensive_insights(
            processed_data, 
            forecast_results, 
            column_analysis
        )
        
        if not insights_result["success"]:
            raise HTTPException(status_code=500, detail=f"Insights generation failed: {insights_result.get('error')}")
        
        # Enhance response with metadata
        response_data = {
            "success": True,
            "file_id": request.file_id,
            "insights": insights_result["insights"],
            "data_period": insights_result["data_period"],
            "generated_at": insights_result["generated_at"],
            "include_forecast": request.include_forecast,
            "forecast_available": forecast_results is not None,
            "summary": _generate_insights_summary(insights_result["insights"])
        }
        
        # Cache result
        insights_cache[f"{request.file_id}_insights"] = response_data
        
        return JSONResponse(content=response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Insights generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/summary/{file_id}")
async def get_insights_summary(file_id: str) -> JSONResponse:
    """
    Get a quick summary of key insights
    
    Args:
        file_id: Processed file identifier
        
    Returns:
        Executive summary of key insights
    """
    try:
        # Check if insights exist
        cache_key = f"{file_id}_insights"
        if cache_key in insights_cache:
            cached_insights = insights_cache[cache_key]["insights"]
            
            summary = {
                "executive_summary": cached_insights.get("executive_summary", [])[:3],
                "key_metrics": cached_insights.get("kpi_dashboard", {}),
                "top_recommendations": cached_insights.get("action_items", [])[:3],
                "risk_alerts": cached_insights.get("risk_alerts", [])[:2]
            }
            
            return JSONResponse(content={
                "success": True,
                "file_id": file_id,
                "summary": summary
            })
        
        # If no cached insights, generate basic summary from processed data
        from api.upload import processing_results
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_result = processing_results[file_id]
        if file_result.get("status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")
        
        # Quick summary from processed data
        processed_data = file_result["results"]
        summary_stats = processed_data.get("summary_stats", {})
        
        quick_summary = {
            "data_overview": {
                "total_records": processed_data.get("processed_shape", [0, 0])[0],
                "data_quality_score": f"{processed_data.get('quality_score', 0):.0f}%",
                "date_range": summary_stats.get("date_range", {})
            },
            "recommendations": processed_data.get("recommendations", [])[:3]
        }
        
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "quick_summary": quick_summary,
            "note": "Generate full insights for detailed analysis"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/types")
async def get_insight_types() -> JSONResponse:
    """
    Get available insight types and their descriptions
    
    Returns:
        List of available insight categories
    """
    try:
        insight_types = {
            "executive_summary": {
                "name": "Executive Summary",
                "description": "High-level business overview and key performance indicators",
                "includes": ["Revenue overview", "Growth analysis", "Data quality assessment"]
            },
            "sales_insights": {
                "name": "Sales Performance Analysis",
                "description": "Detailed analysis of sales patterns and performance metrics",
                "includes": ["Transaction patterns", "Revenue distribution", "Performance variability"]
            },
            "trend_analysis": {
                "name": "Trend Analysis",
                "description": "Historical trends and momentum indicators",
                "includes": ["Growth trends", "Momentum analysis", "Volatility assessment"]
            },
            "seasonality_insights": {
                "name": "Seasonality & Patterns",
                "description": "Seasonal patterns and time-based performance analysis",
                "includes": ["Weekly patterns", "Monthly trends", "Holiday effects", "Indian market patterns"]
            },
            "inventory_alerts": {
                "name": "Inventory Intelligence",
                "description": "Product performance and inventory optimization insights",
                "includes": ["Fast/slow movers", "ABC analysis", "Stock velocity"]
            },
            "revenue_opportunities": {
                "name": "Revenue Optimization",
                "description": "Opportunities to increase revenue and profitability",
                "includes": ["Cross-selling opportunities", "Pricing optimization", "Seasonal focus"]
            },
            "forecast_insights": {
                "name": "Forecast Analysis",
                "description": "Future predictions and planning insights",
                "includes": ["Revenue projections", "Growth forecasts", "Confidence analysis"]
            },
            "risk_alerts": {
                "name": "Risk Assessment",
                "description": "Potential business risks and warning indicators",
                "includes": ["Revenue concentration", "Declining trends", "Operational risks"]
            },
            "action_items": {
                "name": "Action Items",
                "description": "Specific, actionable recommendations prioritized by impact",
                "includes": ["Immediate actions", "Strategic initiatives", "Operational improvements"]
            }
        }
        
        return JSONResponse(content={
            "success": True,
            "insight_types": insight_types,
            "indian_market_specific": [
                "Festival season analysis (Diwali, Dussehra)",
                "Month-end salary impact",
                "Regional holiday considerations",
                "Weekend vs weekday patterns for Indian market"
            ]
        })
        
    except Exception as e:
        logger.error(f"Insight types error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/recommendations/{file_id}")
async def get_specific_recommendations(
    file_id: str,
    category: str = "all"
) -> JSONResponse:
    """
    Get specific recommendations for a category
    
    Args:
        file_id: Processed file identifier
        category: Recommendation category (sales, inventory, marketing, etc.)
        
    Returns:
        Targeted recommendations for the specified category
    """
    try:
        # Get insights from cache or generate
        cache_key = f"{file_id}_insights"
        if cache_key not in insights_cache:
            # Generate insights first
            request = InsightsRequest(file_id=file_id)
            await generate_insights(request)
        
        insights = insights_cache[cache_key]["insights"]
        
        # Filter recommendations by category
        recommendations = {
            "sales": {
                "category": "Sales Optimization",
                "recommendations": insights.get("sales_insights", []) + insights.get("trend_analysis", []),
                "action_items": [item for item in insights.get("action_items", []) if "sales" in item.lower() or "revenue" in item.lower()]
            },
            "inventory": {
                "category": "Inventory Management",
                "recommendations": insights.get("inventory_alerts", []),
                "action_items": [item for item in insights.get("action_items", []) if "inventory" in item.lower() or "stock" in item.lower()]
            },
            "marketing": {
                "category": "Marketing & Promotions",
                "recommendations": insights.get("seasonality_insights", []) + insights.get("revenue_opportunities", []),
                "action_items": [item for item in insights.get("action_items", []) if "marketing" in item.lower() or "promotion" in item.lower()]
            },
            "forecasting": {
                "category": "Demand Planning",
                "recommendations": insights.get("forecast_insights", []),
                "action_items": [item for item in insights.get("action_items", []) if "planning" in item.lower() or "forecast" in item.lower()]
            }
        }
        
        if category == "all":
            return JSONResponse(content={
                "success": True,
                "file_id": file_id,
                "all_recommendations": recommendations
            })
        elif category in recommendations:
            return JSONResponse(content={
                "success": True,
                "file_id": file_id,
                "category": category,
                "recommendations": recommendations[category]
            })
        else:
            raise HTTPException(status_code=400, detail=f"Invalid category: {category}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/export/{file_id}")
async def export_insights_report(file_id: str, format: str = "json") -> JSONResponse:
    """
    Export comprehensive insights report
    
    Args:
        file_id: Processed file identifier
        format: Export format (json, summary, executive)
        
    Returns:
        Formatted insights report for export
    """
    try:
        # Get insights
        cache_key = f"{file_id}_insights"
        if cache_key not in insights_cache:
            raise HTTPException(status_code=404, detail="Insights not found. Generate insights first.")
        
        insights_data = insights_cache[cache_key]
        insights = insights_data["insights"]
        
        if format == "executive":
            # Executive summary format
            report = {
                "report_title": "Retail Intelligence Executive Summary",
                "generated_for": file_id,
                "generated_at": insights_data["generated_at"],
                "data_period": insights_data["data_period"],
                "executive_summary": insights.get("executive_summary", []),
                "key_metrics": insights.get("kpi_dashboard", {}),
                "critical_actions": insights.get("action_items", [])[:5],
                "risk_alerts": insights.get("risk_alerts", []),
                "forecast_summary": insights.get("forecast_insights", [])[:3] if insights.get("forecast_insights") else []
            }
        elif format == "summary":
            # Condensed summary format
            report = {
                "report_type": "Business Summary",
                "file_id": file_id,
                "key_findings": {
                    "sales_performance": insights.get("sales_insights", [])[:3],
                    "trends": insights.get("trend_analysis", [])[:3],
                    "opportunities": insights.get("revenue_opportunities", [])[:3],
                    "immediate_actions": insights.get("action_items", [])[:3]
                },
                "metrics_snapshot": insights.get("kpi_dashboard", {}),
                "generated_at": insights_data["generated_at"]
            }
        else:
            # Full JSON format
            report = insights_data
        
        # Add export metadata
        report["export_info"] = {
            "exported_at": datetime.now().isoformat(),
            "export_format": format,
            "platform": "Retail Intelligence Platform",
            "version": settings.APP_VERSION
        }
        
        return JSONResponse(content={
            "success": True,
            "report": report,
            "export_format": format
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Export error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_insights_summary(insights: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate a summary of insights for quick overview
    
    Args:
        insights: Complete insights dictionary
        
    Returns:
        Summary statistics and key highlights
    """
    try:
        summary = {
            "total_insights": 0,
            "categories": {},
            "priority_items": [],
            "risk_level": "low"
        }
        
        # Count insights by category
        for category, items in insights.items():
            if isinstance(items, list):
                count = len(items)
                summary["categories"][category] = count
                summary["total_insights"] += count
                
                # Identify high-priority items
                if category == "risk_alerts" and count > 0:
                    summary["priority_items"].extend(items[:2])
                    summary["risk_level"] = "high" if count > 2 else "medium"
                elif category == "action_items" and count > 0:
                    summary["priority_items"].extend(items[:3])
        
        # Extract key metrics
        kpi_dashboard = insights.get("kpi_dashboard", {})
        if "financial_metrics" in kpi_dashboard:
            financial = kpi_dashboard["financial_metrics"]
            summary["key_metrics"] = {
                "total_revenue": financial.get("total_revenue", {}).get("value", "N/A"),
                "average_transaction": financial.get("average_transaction", {}).get("value", "N/A")
            }
        
        return summary
        
    except Exception as e:
        logger.error(f"Summary generation error: {e}")
        return {"error": "Unable to generate summary"}
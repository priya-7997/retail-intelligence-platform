"""
Dashboard API for Retail Intelligence Platform
Provides dashboard data and visualization endpoints
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import logging

from config import settings, format_currency

router = APIRouter()
logger = logging.getLogger(__name__)

@router.get("/overview/{file_id}")
async def get_dashboard_overview(file_id: str) -> JSONResponse:
    """
    Get complete dashboard overview for a processed file
    
    Args:
        file_id: Processed file identifier
        
    Returns:
        Dashboard data with KPIs, charts, and insights
    """
    try:
        # Validate file exists
        from api.upload import processing_results
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_result = processing_results[file_id]
        if file_result.get("status") != "completed":
            raise HTTPException(status_code=400, detail="File processing not completed")
        
        # Load processed data
        processed_path = file_result["results"]["processed_file_path"]
        df = pd.read_csv(processed_path)
        column_analysis = file_result["results"]["column_analysis"]
        
        # Generate dashboard data
        dashboard_data = {
            "kpis": _generate_kpi_cards(df, column_analysis),
            "charts": _generate_chart_data(df, column_analysis),
            "recent_insights": _get_recent_insights(file_id),
            "alerts": _get_active_alerts(df, column_analysis),
            "forecast_preview": _get_forecast_preview(file_id),
            "data_health": _get_data_health_status(file_result["results"])
        }
        
        def sanitize_floats(obj):
            if isinstance(obj, dict):
                return {k: sanitize_floats(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize_floats(x) for x in obj]
            elif isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            return obj

        dashboard_data = sanitize_floats(dashboard_data)
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "dashboard": dashboard_data,
            "generated_at": datetime.now().isoformat()
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Dashboard overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/kpis/{file_id}")
async def get_kpi_metrics(file_id: str) -> JSONResponse:
    """
    Get key performance indicators for dashboard
    
    Args:
        file_id: Processed file identifier
        
    Returns:
        KPI metrics with comparisons and trends
    """
    try:
        # Load data
        from api.upload import processing_results
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_result = processing_results[file_id]
        processed_path = file_result["results"]["processed_file_path"]
        df = pd.read_csv(processed_path)
        column_analysis = file_result["results"]["column_analysis"]
        
        kpis = _generate_kpi_cards(df, column_analysis)
        
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "kpis": kpis
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"KPI metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/charts/{file_id}")
async def get_chart_data(file_id: str, chart_type: str = "all") -> JSONResponse:
    """
    Get chart data for dashboard visualizations
    
    Args:
        file_id: Processed file identifier
        chart_type: Specific chart type or 'all'
        
    Returns:
        Chart data formatted for frontend visualization
    """
    try:
        # Load data
        from api.upload import processing_results
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_result = processing_results[file_id]
        processed_path = file_result["results"]["processed_file_path"]
        df = pd.read_csv(processed_path)
        column_analysis = file_result["results"]["column_analysis"]
        
        if chart_type == "all":
            charts = _generate_chart_data(df, column_analysis)
        else:
            charts = {chart_type: _generate_specific_chart(df, column_analysis, chart_type)}
        
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "charts": charts,
            "chart_type": chart_type
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chart data error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/alerts/{file_id}")
async def get_active_alerts(file_id: str) -> JSONResponse:
    """
    Get active alerts and notifications
    
    Args:
        file_id: Processed file identifier
        
    Returns:
        Active alerts with severity levels
    """
    try:
        # Load data
        from api.upload import processing_results
        if file_id not in processing_results:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_result = processing_results[file_id]
        processed_path = file_result["results"]["processed_file_path"]
        df = pd.read_csv(processed_path)
        column_analysis = file_result["results"]["column_analysis"]
        
        alerts = _get_active_alerts(df, column_analysis)
        
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "alerts": alerts
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/forecast-preview/{file_id}")
async def get_forecast_preview(file_id: str) -> JSONResponse:
    """
    Get forecast preview for dashboard
    
    Args:
        file_id: Processed file identifier
        
    Returns:
        Forecast preview with key projections
    """
    try:
        forecast_data = _get_forecast_preview(file_id)
        
        return JSONResponse(content={
            "success": True,
            "file_id": file_id,
            "forecast_preview": forecast_data
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forecast preview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

def _generate_kpi_cards(df: pd.DataFrame, column_analysis: Dict) -> List[Dict]:
    """Generate KPI cards for dashboard"""
    kpis = []
    
    try:
        sales_col = column_analysis.get('sales_column')
        date_col = column_analysis.get('date_column')
        
        if sales_col:
            sales_data = df[sales_col].dropna()
            
            # Total Revenue KPI
            total_revenue = sales_data.sum()
            kpis.append({
                "id": "total_revenue",
                "title": "Total Revenue",
                "value": format_currency(total_revenue),
                "raw_value": float(total_revenue),
                "icon": "₹",
                "color": "blue",
                "trend": _calculate_revenue_trend(df, sales_col, date_col),
                "description": "Total sales revenue"
            })
            
            # Average Transaction KPI
            avg_transaction = sales_data.mean()
            kpis.append({
                "id": "avg_transaction",
                "title": "Average Transaction",
                "value": format_currency(avg_transaction),
                "raw_value": float(avg_transaction),
                "icon": "📊",
                "color": "green",
                "description": "Average transaction value"
            })
            
            # Transaction Count KPI
            transaction_count = len(df)
            kpis.append({
                "id": "transaction_count",
                "title": "Total Transactions",
                "value": f"{transaction_count:,}",
                "raw_value": transaction_count,
                "icon": "🔢",
                "color": "purple",
                "description": "Number of transactions"
            })
            
            # Peak Sales Day KPI
            if date_col:
                df_temp = df.copy()
                df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                daily_sales = df_temp.groupby(df_temp[date_col].dt.date)[sales_col].sum()
                peak_sales = daily_sales.max()
                peak_date = daily_sales.idxmax()
                
                kpis.append({
                    "id": "peak_sales",
                    "title": "Peak Sales Day",
                    "value": format_currency(peak_sales),
                    "raw_value": float(peak_sales),
                    "icon": "⭐",
                    "color": "orange",
                    "description": f"Best day: {peak_date}",
                    "subtitle": str(peak_date)
                })
        
        # Data Quality Score KPI
        from api.upload import processing_results
        for file_id, result in processing_results.items():
            if result.get("results", {}).get("data") is not None:
                quality_score = result["results"].get("quality_score", 0)
                kpis.append({
                    "id": "data_quality",
                    "title": "Data Quality",
                    "value": f"{quality_score:.0f}%",
                    "raw_value": quality_score,
                    "icon": "✅",
                    "color": "green" if quality_score >= 80 else "yellow" if quality_score >= 60 else "red",
                    "description": "Data completeness and accuracy"
                })
                break
        
    except Exception as e:
        logger.error(f"KPI generation error: {e}")
        kpis.append({
            "id": "error",
            "title": "Error",
            "value": "N/A",
            "description": "Unable to calculate KPIs"
        })
    
    return kpis

def _generate_chart_data(df: pd.DataFrame, column_analysis: Dict) -> Dict[str, Any]:
    """Generate chart data for dashboard"""
    charts = {}
    
    try:
        sales_col = column_analysis.get('sales_column')
        date_col = column_analysis.get('date_column')
        product_col = column_analysis.get('product_column')
        
        # Sales Trend Chart
        if sales_col and date_col:
            charts["sales_trend"] = _generate_sales_trend_chart(df, sales_col, date_col)
        
        # Product Performance Chart
        if product_col and sales_col:
            charts["product_performance"] = _generate_product_performance_chart(df, product_col, sales_col)
        
        # Weekly Pattern Chart
        if date_col and sales_col:
            charts["weekly_pattern"] = _generate_weekly_pattern_chart(df, date_col, sales_col)
        
        # Revenue Distribution Chart
        if sales_col:
            charts["revenue_distribution"] = _generate_revenue_distribution_chart(df, sales_col)
        
        # Monthly Comparison Chart
        if date_col and sales_col:
            charts["monthly_comparison"] = _generate_monthly_comparison_chart(df, date_col, sales_col)
    
    except Exception as e:
        logger.error(f"Chart generation error: {e}")
        charts["error"] = {"message": "Unable to generate charts"}
    
    return charts

def _generate_sales_trend_chart(df: pd.DataFrame, sales_col: str, date_col: str) -> Dict:
    """Generate sales trend chart data"""
    try:
        df_chart = df.copy()
        df_chart[date_col] = pd.to_datetime(df_chart[date_col])
        
        # Daily aggregation
        daily_sales = df_chart.groupby(df_chart[date_col].dt.date)[sales_col].sum().reset_index()
        daily_sales.columns = ['date', 'sales']
        daily_sales['date'] = daily_sales['date'].astype(str)
        
        # Calculate moving average
        daily_sales['moving_avg'] = daily_sales['sales'].rolling(window=7, center=True).mean()
        
        return {
            "type": "line",
            "title": "Sales Trend",
            "data": daily_sales.to_dict('records'),
            "x_axis": "date",
            "y_axis": "sales",
            "series": [
                {
                    "name": "Daily Sales",
                    "data": daily_sales[['date', 'sales']].to_dict('records'),
                    "color": "#3b82f6"
                },
                {
                    "name": "7-Day Moving Average",
                    "data": daily_sales[['date', 'moving_avg']].dropna().to_dict('records'),
                    "color": "#ef4444"
                }
            ]
        }
    except Exception as e:
        logger.error(f"Sales trend chart error: {e}")
        return {"error": "Unable to generate sales trend chart"}

def _generate_product_performance_chart(df: pd.DataFrame, product_col: str, sales_col: str) -> Dict:
    """Generate product performance chart data"""
    try:
        product_sales = df.groupby(product_col)[sales_col].agg(['sum', 'count']).reset_index()
        product_sales.columns = ['product', 'total_sales', 'transaction_count']
        
        # Top 10 products by sales
        top_products = product_sales.nlargest(10, 'total_sales')
        
        return {
            "type": "bar",
            "title": "Top Products by Revenue",
            "data": top_products.to_dict('records'),
            "x_axis": "product",
            "y_axis": "total_sales",
            "color": "#10b981"
        }
    except Exception as e:
        logger.error(f"Product performance chart error: {e}")
        return {"error": "Unable to generate product performance chart"}

def _generate_weekly_pattern_chart(df: pd.DataFrame, date_col: str, sales_col: str) -> Dict:
    """Generate weekly pattern chart data"""
    try:
        df_chart = df.copy()
        df_chart[date_col] = pd.to_datetime(df_chart[date_col])
        df_chart['day_name'] = df_chart[date_col].dt.day_name()
        
        # Order days properly
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        weekly_sales = df_chart.groupby('day_name')[sales_col].mean().reindex(day_order).reset_index()
        weekly_sales.columns = ['day', 'avg_sales']
        
        return {
            "type": "bar",
            "title": "Average Sales by Day of Week",
            "data": weekly_sales.to_dict('records'),
            "x_axis": "day",
            "y_axis": "avg_sales",
            "color": "#8b5cf6"
        }
    except Exception as e:
        logger.error(f"Weekly pattern chart error: {e}")
        return {"error": "Unable to generate weekly pattern chart"}

def _generate_revenue_distribution_chart(df: pd.DataFrame, sales_col: str) -> Dict:
    """Generate revenue distribution chart data"""
    try:
        sales_data = df[sales_col].dropna()
        
        # Create bins for distribution
        bins = np.histogram_bin_edges(sales_data, bins=10)
        hist, _ = np.histogram(sales_data, bins=bins)
        
        # Format bin labels
        bin_labels = []
        for i in range(len(bins)-1):
            label = f"₹{bins[i]:.0f}-₹{bins[i+1]:.0f}"
            bin_labels.append(label)
        
        distribution_data = []
        for i, (label, count) in enumerate(zip(bin_labels, hist)):
            distribution_data.append({
                "range": label,
                "count": int(count),
                "percentage": float(count / len(sales_data) * 100)
            })
        
        return {
            "type": "histogram",
            "title": "Transaction Value Distribution",
            "data": distribution_data,
            "x_axis": "range",
            "y_axis": "count",
            "color": "#f59e0b"
        }
    except Exception as e:
        logger.error(f"Revenue distribution chart error: {e}")
        return {"error": "Unable to generate revenue distribution chart"}

def _generate_monthly_comparison_chart(df: pd.DataFrame, date_col: str, sales_col: str) -> Dict:
    """Generate monthly comparison chart data"""
    try:
        df_chart = df.copy()
        df_chart[date_col] = pd.to_datetime(df_chart[date_col])
        df_chart['month_year'] = df_chart[date_col].dt.to_period('M')
        
        monthly_sales = df_chart.groupby('month_year')[sales_col].sum().reset_index()
        monthly_sales['month_year'] = monthly_sales['month_year'].astype(str)
        monthly_sales.columns = ['month', 'total_sales']
        
        # Calculate month-over-month growth
        monthly_sales['growth'] = monthly_sales['total_sales'].pct_change() * 100
        
        return {
            "type": "line",
            "title": "Monthly Sales Comparison",
            "data": monthly_sales.to_dict('records'),
            "x_axis": "month",
            "y_axis": "total_sales",
            "color": "#ef4444"
        }
    except Exception as e:
        logger.error(f"Monthly comparison chart error: {e}")
        return {"error": "Unable to generate monthly comparison chart"}

def _get_recent_insights(file_id: str) -> List[Dict]:
    """Get recent insights for dashboard"""
    try:
        from api.insights import insights_cache
        cache_key = f"{file_id}_insights"
        
        if cache_key in insights_cache:
            insights = insights_cache[cache_key]["insights"]
            
            # Get top insights from different categories
            recent_insights = []
            
            # Executive summary
            exec_summary = insights.get("executive_summary", [])
            if exec_summary:
                recent_insights.append({
                    "type": "executive",
                    "title": "Business Overview",
                    "content": exec_summary[0],
                    "priority": "high"
                })
            
            # Key trend
            trends = insights.get("trend_analysis", [])
            if trends:
                recent_insights.append({
                    "type": "trend",
                    "title": "Key Trend",
                    "content": trends[0],
                    "priority": "medium"
                })
            
            # Top opportunity
            opportunities = insights.get("revenue_opportunities", [])
            if opportunities:
                recent_insights.append({
                    "type": "opportunity",
                    "title": "Revenue Opportunity",
                    "content": opportunities[0],
                    "priority": "medium"
                })
            
            return recent_insights[:3]  # Limit to 3 insights
        
        return []
    except Exception as e:
        logger.error(f"Recent insights error: {e}")
        return []

def _get_active_alerts(df: pd.DataFrame, column_analysis: Dict) -> List[Dict]:
    """Get active alerts for dashboard"""
    alerts = []
    
    try:
        sales_col = column_analysis.get('sales_column')
        date_col = column_analysis.get('date_column')
        
        if sales_col:
            sales_data = df[sales_col].dropna()
            
            # Zero sales alert
            zero_sales = len(sales_data[sales_data == 0])
            if zero_sales > 0:
                alerts.append({
                    "id": "zero_sales",
                    "type": "warning",
                    "title": "Zero Sales Days",
                    "message": f"{zero_sales} days with zero sales detected",
                    "severity": "medium",
                    "action": "Review operational processes"
                })
            
            # High variability alert
            cv = sales_data.std() / sales_data.mean() if sales_data.mean() > 0 else 0
            if cv > 0.8:
                alerts.append({
                    "id": "high_variability",
                    "type": "info",
                    "title": "High Sales Variability",
                    "message": f"Sales show high variability (CV: {cv:.2f})",
                    "severity": "low",
                    "action": "Consider demand smoothing strategies"
                })
            
            # Declining trend alert
            if date_col and len(df) >= 14:
                df_trend = df.copy()
                df_trend[date_col] = pd.to_datetime(df_trend[date_col])
                df_trend = df_trend.sort_values(date_col)
                
                recent_avg = df_trend[sales_col].tail(7).mean()
                previous_avg = df_trend[sales_col].tail(14).head(7).mean()
                
                if previous_avg > 0:
                    decline_rate = ((previous_avg - recent_avg) / previous_avg) * 100
                    if decline_rate > 15:
                        alerts.append({
                            "id": "declining_trend",
                            "type": "error",
                            "title": "Declining Sales Trend",
                            "message": f"Sales declined {decline_rate:.1f}% in recent period",
                            "severity": "high",
                            "action": "Immediate attention required"
                        })
    
    except Exception as e:
        logger.error(f"Alerts generation error: {e}")
        alerts.append({
            "id": "error",
            "type": "error",
            "title": "Alert Generation Error",
            "message": "Unable to generate alerts",
            "severity": "low"
        })
    
    return alerts

def _get_forecast_preview(file_id: str) -> Dict:
    """Get forecast preview for dashboard"""
    try:
        from api.forecast import forecast_cache
        
        # Look for recent forecast
        for cache_key, forecast_data in forecast_cache.items():
            if file_id in cache_key:
                forecast = forecast_data.get("forecast", [])
                if forecast:
                    # Get next 7 days
                    next_week = forecast[:7]
                    total_forecast = sum(item["yhat"] for item in next_week)
                    
                    return {
                        "available": True,
                        "next_7_days_total": format_currency(total_forecast),
                        "daily_average": format_currency(total_forecast / 7),
                        "model_used": forecast_data.get("model_used", "Unknown"),
                        "generated_at": forecast_data.get("generated_at"),
                        "preview_data": next_week
                    }
        
        return {
            "available": False,
            "message": "No forecast available. Generate forecast first."
        }
        
    except Exception as e:
        logger.error(f"Forecast preview error: {e}")
        return {"available": False, "error": "Unable to load forecast preview"}

def _get_data_health_status(processed_results: Dict) -> Dict:
    """Get data health status for dashboard"""
    try:
        quality_score = processed_results.get("quality_score", 0)
        validation = processed_results.get("validation_results", {})
        
        # Determine health status
        if quality_score >= 80:
            status = "excellent"
            color = "green"
        elif quality_score >= 60:
            status = "good"
            color = "yellow"
        else:
            status = "poor"
            color = "red"
        
        return {
            "status": status,
            "score": quality_score,
            "color": color,
            "issues": validation.get("issues", []),
            "recommendations": processed_results.get("recommendations", [])[:3]
        }
        
    except Exception as e:
        logger.error(f"Data health status error: {e}")
        return {"status": "unknown", "score": 0, "error": "Unable to assess data health"}

def _calculate_revenue_trend(df: pd.DataFrame, sales_col: str, date_col: str) -> Dict:
    """Calculate revenue trend for KPI"""
    try:
        if not date_col or len(df) < 14:
            return {"trend": "stable", "change": 0}
        
        df_trend = df.copy()
        df_trend[date_col] = pd.to_datetime(df_trend[date_col])
        df_trend = df_trend.sort_values(date_col)
        
        # Compare recent week vs previous week
        recent_week = df_trend[sales_col].tail(7).sum()
        previous_week = df_trend[sales_col].tail(14).head(7).sum()
        
        if previous_week > 0:
            change_pct = ((recent_week - previous_week) / previous_week) * 100
            
            if change_pct > 5:
                trend = "up"
            elif change_pct < -5:
                trend = "down"
            else:
                trend = "stable"
            
            return {"trend": trend, "change": change_pct}
        
        return {"trend": "stable", "change": 0}
        
    except Exception as e:
        logger.error(f"Revenue trend calculation error: {e}")
        return {"trend": "unknown", "change": 0}

def _generate_specific_chart(df: pd.DataFrame, column_analysis: Dict, chart_type: str) -> Dict:
    """Generate specific chart data"""
    sales_col = column_analysis.get('sales_column')
    date_col = column_analysis.get('date_column')
    product_col = column_analysis.get('product_column')
    
    if chart_type == "sales_trend" and sales_col and date_col:
        return _generate_sales_trend_chart(df, sales_col, date_col)
    elif chart_type == "product_performance" and product_col and sales_col:
        return _generate_product_performance_chart(df, product_col, sales_col)
    elif chart_type == "weekly_pattern" and date_col and sales_col:
        return _generate_weekly_pattern_chart(df, date_col, sales_col)
    elif chart_type == "revenue_distribution" and sales_col:
        return _generate_revenue_distribution_chart(df, sales_col)
    elif chart_type == "monthly_comparison" and date_col and sales_col:
        return _generate_monthly_comparison_chart(df, date_col, sales_col)
    else:
        return {"error": f"Chart type '{chart_type}' not available or missing required columns"}
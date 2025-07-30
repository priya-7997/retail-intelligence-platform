"""
Smart Alert System for Retail Intelligence Platform
Generates real-time alerts and notifications based on data patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

from config import settings, format_currency

logger = logging.getLogger(__name__)

class AlertsGenerator:
    """Generate intelligent alerts for retail business"""
    
    def __init__(self):
        self.alert_thresholds = {
            'stock_out_days': 7,  # Alert if stock out predicted in 7 days
            'sales_decline_threshold': 20,  # Alert if sales decline > 20%
            'data_quality_threshold': 70,  # Alert if data quality < 70%
            'high_volatility_cv': 0.8,  # Alert if coefficient of variation > 0.8
            'zero_sales_threshold': 5  # Alert if more than 5 zero sales days
        }
    
    def generate_alerts(self, df: pd.DataFrame, column_analysis: Dict, 
                       processed_data: Dict = None, forecast_data: List = None) -> List[Dict]:
        """Generate comprehensive alerts for the business"""
        alerts = []
        
        try:
            # Data quality alerts
            alerts.extend(self._check_data_quality(processed_data))
            
            # Sales performance alerts
            alerts.extend(self._check_sales_performance(df, column_analysis))
            
            # Operational alerts
            alerts.extend(self._check_operational_issues(df, column_analysis))
            
            # Inventory alerts
            alerts.extend(self._check_inventory_issues(df, column_analysis))
            
            # Forecast-based alerts
            if forecast_data:
                alerts.extend(self._check_forecast_alerts(forecast_data))
            
            # Sort alerts by severity
            alerts.sort(key=lambda x: self._get_severity_weight(x.get('severity', 'low')), reverse=True)
            
            logger.info(f"Generated {len(alerts)} alerts")
            return alerts
            
        except Exception as e:
            logger.error(f"Error generating alerts: {e}")
            return [{
                'id': 'alert_error',
                'type': 'error',
                'severity': 'high',
                'title': 'Alert System Error',
                'message': 'Unable to generate alerts',
                'timestamp': datetime.now().isoformat()
            }]
    
    def _check_data_quality(self, processed_data: Dict) -> List[Dict]:
        """Check for data quality issues"""
        alerts = []
        
        if not processed_data:
            return alerts
        
        quality_score = processed_data.get('quality_score', 100)
        validation_results = processed_data.get('validation_results', {})
        
        # Low data quality alert
        if quality_score < self.alert_thresholds['data_quality_threshold']:
            alerts.append({
                'id': 'data_quality_low',
                'type': 'warning',
                'severity': 'medium',
                'title': 'Low Data Quality',
                'message': f'Data quality score is {quality_score:.0f}%. Consider improving data collection.',
                'action': 'Review data sources and ensure consistent data entry',
                'timestamp': datetime.now().isoformat()
            })
        
        # Missing data alert
        missing_pct = validation_results.get('missing_data_percentage', 0)
        if missing_pct > 20:
            alerts.append({
                'id': 'missing_data_high',
                'type': 'warning',
                'severity': 'medium',
                'title': 'High Missing Data',
                'message': f'{missing_pct:.1f}% of data is missing or incomplete.',
                'action': 'Fill gaps in data collection to improve accuracy',
                'timestamp': datetime.now().isoformat()
            })
        
        # Insufficient data alert
        if not validation_results.get('has_sufficient_data', True):
            alerts.append({
                'id': 'insufficient_data',
                'type': 'info',
                'severity': 'low',
                'title': 'Insufficient Historical Data',
                'message': 'More historical data needed for reliable forecasting.',
                'action': 'Continue collecting data for better predictions',
                'timestamp': datetime.now().isoformat()
            })
        
        return alerts
    
    def _check_sales_performance(self, df: pd.DataFrame, column_analysis: Dict) -> List[Dict]:
        """Check for sales performance issues"""
        alerts = []
        
        sales_col = column_analysis.get('sales_column')
        date_col = column_analysis.get('date_column')
        
        if not sales_col or not date_col:
            return alerts
        
        try:
            df_sorted = df.sort_values(date_col)
            sales_data = df_sorted[sales_col].dropna()
            
            # Recent sales decline
            if len(df_sorted) >= 14:
                recent_avg = df_sorted[sales_col].tail(7).mean()
                previous_avg = df_sorted[sales_col].tail(14).head(7).mean()
                
                if previous_avg > 0:
                    decline_rate = ((previous_avg - recent_avg) / previous_avg) * 100
                    
                    if decline_rate > self.alert_thresholds['sales_decline_threshold']:
                        alerts.append({
                            'id': 'sales_decline',
                            'type': 'error',
                            'severity': 'high',
                            'title': 'Significant Sales Decline',
                            'message': f'Sales dropped {decline_rate:.1f}% in the last week.',
                            'action': 'Investigate causes and implement recovery strategies',
                            'timestamp': datetime.now().isoformat()
                        })
            
            # High sales volatility
            cv = sales_data.std() / sales_data.mean() if sales_data.mean() > 0 else 0
            if cv > self.alert_thresholds['high_volatility_cv']:
                alerts.append({
                    'id': 'high_volatility',
                    'type': 'warning',
                    'severity': 'medium',
                    'title': 'High Sales Volatility',
                    'message': f'Sales show high variability (CV: {cv:.2f}). This makes forecasting difficult.',
                    'action': 'Analyze factors causing volatility and implement stabilization measures',
                    'timestamp': datetime.now().isoformat()
                })
            
            # Zero sales days
            zero_sales_count = len(sales_data[sales_data == 0])
            if zero_sales_count > self.alert_thresholds['zero_sales_threshold']:
                alerts.append({
                    'id': 'zero_sales_days',
                    'type': 'warning',
                    'severity': 'medium',
                    'title': 'Multiple Zero Sales Days',
                    'message': f'{zero_sales_count} days with zero sales detected.',
                    'action': 'Review operational processes and business hours',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error checking sales performance: {e}")
        
        return alerts
    
    def _check_operational_issues(self, df: pd.DataFrame, column_analysis: Dict) -> List[Dict]:
        """Check for operational issues"""
        alerts = []
        
        date_col = column_analysis.get('date_column')
        sales_col = column_analysis.get('sales_column')
        
        if not date_col or not sales_col:
            return alerts
        
        try:
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            
            # Check for data gaps
            date_range = df_temp[date_col].max() - df_temp[date_col].min()
            expected_days = date_range.days + 1
            actual_days = df_temp[date_col].nunique()
            
            if actual_days < expected_days * 0.8:  # More than 20% missing days
                missing_days = expected_days - actual_days
                alerts.append({
                    'id': 'data_gaps',
                    'type': 'warning',
                    'severity': 'medium',
                    'title': 'Data Collection Gaps',
                    'message': f'{missing_days} days of data missing from records.',
                    'action': 'Ensure consistent daily data collection',
                    'timestamp': datetime.now().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error checking operational issues: {e}")
        
        return alerts
    
    def _check_inventory_issues(self, df: pd.DataFrame, column_analysis: Dict) -> List[Dict]:
        """Check for inventory-related issues"""
        alerts = []
        
        product_col = column_analysis.get('product_column')
        sales_col = column_analysis.get('sales_column')
        
        if not product_col or not sales_col:
            return alerts
        
        try:
            # Product concentration risk
            product_revenue = df.groupby(product_col)[sales_col].sum()
            total_revenue = product_revenue.sum()
            
            if total_revenue > 0:
                top_product_share = (product_revenue.max() / total_revenue) * 100
                
                if top_product_share > 50:
                    top_product = product_revenue.idxmax()
                    alerts.append({
                        'id': 'revenue_concentration',
                        'type': 'warning',
                        'severity': 'medium',
                        'title': 'Revenue Concentration Risk',
                        'message': f'{top_product} generates {top_product_share:.1f}% of total revenue.',
                        'action': 'Diversify product portfolio to reduce dependency',
                        'timestamp': datetime.now().isoformat()
                    })
            
        except Exception as e:
            logger.error(f"Error checking inventory issues: {e}")
        
        return alerts
    
    def _check_forecast_alerts(self, forecast_data: List[Dict]) -> List[Dict]:
        """Check forecast data for potential issues"""
        alerts = []
        
        if not forecast_data:
            return alerts
        
        try:
            # Extract forecast values
            forecast_values = [item.get('yhat', 0) for item in forecast_data]
            
            # Declining forecast trend
            if len(forecast_values) >= 7:
                early_avg = np.mean(forecast_values[:7])
                later_avg = np.mean(forecast_values[-7:])
                
                if early_avg > 0:
                    trend_change = ((later_avg - early_avg) / early_avg) * 100
                    
                    if trend_change < -15:  # More than 15% decline forecasted
                        alerts.append({
                            'id': 'forecast_decline',
                            'type': 'warning',
                            'severity': 'high',
                            'title': 'Declining Sales Forecast',
                            'message': f'Forecast shows {abs(trend_change):.1f}% decline in sales trend.',
                            'action': 'Prepare contingency plans and promotional strategies',
                            'timestamp': datetime.now().isoformat()
                        })
            
        except Exception as e:
            logger.error(f"Error checking forecast alerts: {e}")
        
        return alerts
    
    def _get_severity_weight(self, severity: str) -> int:
        """Get numeric weight for alert severity"""
        weights = {
            'high': 3,
            'medium': 2,
            'low': 1
        }
        return weights.get(severity, 1)
    
    def get_alert_summary(self, alerts: List[Dict]) -> Dict[str, Any]:
        """Generate summary of alerts"""
        if not alerts:
            return {
                'total_alerts': 0,
                'by_severity': {'high': 0, 'medium': 0, 'low': 0},
                'by_type': {},
                'most_recent': None
            }
        
        # Count by severity
        by_severity = {'high': 0, 'medium': 0, 'low': 0}
        for alert in alerts:
            severity = alert.get('severity', 'low')
            by_severity[severity] += 1
        
        # Count by type
        by_type = {}
        for alert in alerts:
            alert_type = alert.get('type', 'unknown')
            by_type[alert_type] = by_type.get(alert_type, 0) + 1
        
        return {
            'total_alerts': len(alerts),
            'by_severity': by_severity,
            'by_type': by_type,
            'critical_count': by_severity['high'],
            'needs_attention': by_severity['high'] + by_severity['medium']
        } 

"""
Business Insights Generator for Retail Intelligence Platform
Converts data patterns into actionable business recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from collections import defaultdict

from config import settings, format_currency, format_indian_number

logger = logging.getLogger(__name__)

class InsightsGenerator:
    """Generate actionable business insights from retail data and forecasts"""
    
    def __init__(self):
        self.insight_templates = {
            'sales_trend': {
                'positive': "📈 Sales are trending upward with {growth_rate}% growth over {period}",
                'negative': "📉 Sales are declining by {decline_rate}% over {period} - immediate attention needed",
                'stable': "📊 Sales remain stable with minimal variation over {period}"
            },
            'seasonality': {
                'weekly': "📅 {best_day} is your best performing day with {performance}% higher sales than average",
                'monthly': "🗓️ {best_month} shows strongest performance with {lift}% above average monthly sales",
                'seasonal': "🌟 Clear seasonal pattern detected - plan inventory for {peak_season}"
            },
            'inventory': {
                'stockout_warning': "⚠️ {product} projected to stock out in {days} days at current sales velocity",
                'overstock_alert': "📦 {product} has {weeks} weeks of excess inventory - consider promotions",
                'reorder_point': "🔄 Optimal reorder point for {product} is {quantity} units"
            },
            'revenue_opportunities': {
                'high_margin': "💰 {product} has highest margin at {margin}% - focus on promoting this item",
                'bundle_opportunity': "🎁 Customers buying {product1} also buy {product2} {frequency}% of the time",
                'price_optimization': "💸 {product} price elasticity suggests {recommendation}"
            }
        }
    
    def generate_comprehensive_insights(self, 
                                      processed_data: Dict, 
                                      forecast_results: Dict = None,
                                      column_analysis: Dict = None) -> Dict[str, Any]:
        """Generate complete business insights from processed data and forecasts"""
        try:
            df = processed_data['data']
            logger.info(f"[DEBUG] DataFrame shape: {df.shape}, columns: {list(df.columns)}")
            logger.info(f"[DEBUG] column_analysis: {column_analysis}")
            logger.info(f"[DEBUG] forecast_results: {forecast_results}")
            insights = {
                'executive_summary': [],
                'sales_insights': [],
                'trend_analysis': [],
                'seasonality_insights': [],
                'inventory_alerts': [],
                'revenue_opportunities': [],
                'action_items': [],
                'kpi_dashboard': {},
                'risk_alerts': []
            }

            # Generate different types of insights
            logger.info("[DEBUG] Generating executive summary...")
            insights['executive_summary'] = self._generate_executive_summary(df, processed_data, column_analysis)
            logger.info(f"[DEBUG] Executive summary: {insights['executive_summary']}")

            logger.info("[DEBUG] Analyzing sales patterns...")
            insights['sales_insights'] = self._analyze_sales_patterns(df, column_analysis)
            logger.info(f"[DEBUG] Sales insights: {insights['sales_insights']}")

            logger.info("[DEBUG] Analyzing trends...")
            insights['trend_analysis'] = self._analyze_trends(df, column_analysis)
            logger.info(f"[DEBUG] Trend analysis: {insights['trend_analysis']}")

            logger.info("[DEBUG] Analyzing seasonality...")
            insights['seasonality_insights'] = self._analyze_seasonality(df, column_analysis)
            logger.info(f"[DEBUG] Seasonality insights: {insights['seasonality_insights']}")

            logger.info("[DEBUG] Generating KPI dashboard...")
            insights['kpi_dashboard'] = self._generate_kpi_dashboard(df, column_analysis)
            logger.info(f"[DEBUG] KPI dashboard: {insights['kpi_dashboard']}")

            # Product-specific insights if product data available
            if column_analysis and column_analysis.get('product_column'):
                logger.info("[DEBUG] Generating inventory alerts...")
                insights['inventory_alerts'] = self._generate_inventory_insights(df, column_analysis)
                logger.info(f"[DEBUG] Inventory alerts: {insights['inventory_alerts']}")
                logger.info("[DEBUG] Identifying revenue opportunities...")
                insights['revenue_opportunities'] = self._identify_revenue_opportunities(df, column_analysis)
                logger.info(f"[DEBUG] Revenue opportunities: {insights['revenue_opportunities']}")

            # Forecast-based insights
            if forecast_results:
                logger.info("[DEBUG] Analyzing forecast insights...")
                forecast_insights = self._analyze_forecast_insights(forecast_results)
                insights['forecast_insights'] = forecast_insights
                logger.info(f"[DEBUG] Forecast insights: {forecast_insights}")

            # Generate action items
            logger.info("[DEBUG] Generating action items...")
            insights['action_items'] = self._generate_action_items(insights)
            logger.info(f"[DEBUG] Action items: {insights['action_items']}")

            logger.info("[DEBUG] Identifying risks...")
            insights['risk_alerts'] = self._identify_risks(df, column_analysis, forecast_results)
            logger.info(f"[DEBUG] Risk alerts: {insights['risk_alerts']}")

            logger.info(f"Generated {sum(len(v) if isinstance(v, list) else 1 for v in insights.values())} insights")

            return {
                'success': True,
                'insights': insights,
                'generated_at': datetime.now().isoformat(),
                'data_period': self._get_data_period(df, column_analysis)
            }

        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return {
                'success': False,
                'error': str(e),
                'insights': {}
            }
    
    def _generate_executive_summary(self, df: pd.DataFrame, processed_data: Dict, column_analysis: Dict) -> List[str]:
        """Generate high-level executive summary"""
        summary = []
        
        try:
            sales_col = column_analysis.get('sales_column')
            date_col = column_analysis.get('date_column')
            
            if not sales_col or not date_col:
                return ["Unable to generate executive summary - missing required columns"]
            
            # Total performance metrics
            total_sales = df[sales_col].sum()
            avg_daily_sales = df[sales_col].mean()
            data_quality = processed_data.get('quality_score', 0)
            
            summary.append(f"📊 **Business Overview**: Total sales of {format_currency(total_sales)} across {len(df)} transactions")
            summary.append(f"💰 **Daily Average**: {format_currency(avg_daily_sales)} per day with {data_quality:.0f}% data quality score")
            
            # Growth analysis
            if len(df) >= 14:  # At least 2 weeks
                recent_sales = df.tail(7)[sales_col].mean()
                earlier_sales = df.head(7)[sales_col].mean()
                
                if earlier_sales > 0:
                    growth_rate = ((recent_sales - earlier_sales) / earlier_sales) * 100
                    if growth_rate > 10:
                        summary.append(f"🚀 **Strong Growth**: Recent sales up {growth_rate:.1f}% - business momentum is positive")
                    elif growth_rate < -10:
                        summary.append(f"⚠️ **Declining Trend**: Sales down {abs(growth_rate):.1f}% - requires immediate attention")
                    else:
                        summary.append(f"📈 **Stable Performance**: Sales growth at {growth_rate:.1f}% - maintaining steady trajectory")
            
            # Forecasting readiness
            validation = processed_data.get('validation_results', {})
            if validation.get('has_sufficient_data') and validation.get('seasonal_potential'):
                summary.append("✅ **Forecast Ready**: Data quality excellent for predictive analytics and demand planning")
            else:
                summary.append("⏳ **Data Building**: Collect more historical data for enhanced forecasting capabilities")
            
        except Exception as e:
            logger.error(f"Error in executive summary: {e}")
            summary.append("❌ Unable to generate complete executive summary")
        
        return summary
    
    def _analyze_sales_patterns(self, df: pd.DataFrame, column_analysis: Dict) -> List[str]:
        """Analyze sales patterns and performance"""
        patterns = []
        
        try:
            sales_col = column_analysis.get('sales_column')
            date_col = column_analysis.get('date_column')
            
            if not sales_col or not date_col:
                return ["Sales pattern analysis requires date and sales columns"]
            
            # Statistical analysis
            sales_data = df[sales_col].dropna()
            if len(sales_data) == 0:
                return ["No valid sales data found for analysis"]
                
            mean_sales = sales_data.mean()
            median_sales = sales_data.median()
            std_sales = sales_data.std()
            
            # Variability analysis
            cv = std_sales / mean_sales if mean_sales > 0 else 0
            if cv > 0.5:
                patterns.append(f"📊 **High Variability**: Sales vary significantly (CV: {cv:.2f}) - consider demand smoothing strategies")
            elif cv < 0.2:
                patterns.append(f"📊 **Consistent Sales**: Low variability (CV: {cv:.2f}) - predictable demand pattern")
            
            # Distribution analysis
            if median_sales < mean_sales * 0.8:
                patterns.append("📈 **Right-skewed Distribution**: Few high-value transactions drive overall performance")
            elif median_sales > mean_sales * 1.2:
                patterns.append("📉 **Left-skewed Distribution**: Majority of transactions above average value")
            
            # Peak performance identification
            top_10_pct = sales_data.quantile(0.9)
            peak_days = len(sales_data[sales_data >= top_10_pct])
            patterns.append(f"⭐ **Peak Performance**: {peak_days} days ({peak_days/len(df)*100:.1f}%) generated top 10% sales")
            
            # Zero sales analysis
            zero_sales_days = len(sales_data[sales_data == 0])
            if zero_sales_days > 0:
                patterns.append(f"🚫 **No-Sales Days**: {zero_sales_days} days with zero sales - investigate operational issues")
            
        except Exception as e:
            logger.error(f"Error in sales pattern analysis: {e}")
            patterns.append("❌ Unable to complete sales pattern analysis")
        
        return patterns
    
    def _analyze_trends(self, df: pd.DataFrame, column_analysis: Dict) -> List[str]:
        """Analyze sales trends over time"""
        trends = []
        
        try:
            sales_col = column_analysis.get('sales_column')
            date_col = column_analysis.get('date_column')
            
            if not sales_col or not date_col:
                return ["Trend analysis requires date and sales columns"]
            
            df_sorted = df.sort_values(date_col)
            
            # Moving average trends
            if len(df_sorted) >= 7:
                df_sorted = df_sorted.copy()  # Avoid SettingWithCopyWarning
                df_sorted['ma_7'] = df_sorted[sales_col].rolling(window=7, center=True).mean()
                
                # Recent trend (last 7 days vs previous 7 days)
                if len(df_sorted) >= 14:
                    recent_avg = df_sorted[sales_col].tail(7).mean()
                    previous_avg = df_sorted[sales_col].tail(14).head(7).mean()
                    
                    if previous_avg > 0:
                        trend_change = ((recent_avg - previous_avg) / previous_avg) * 100
                        
                        if trend_change > 15:
                            trends.append(f"🚀 **Strong Upward Trend**: Sales up {trend_change:.1f}% in recent week - capitalize on momentum")
                        elif trend_change > 5:
                            trends.append(f"📈 **Positive Trend**: Sales growing {trend_change:.1f}% - continue current strategies")
                        elif trend_change < -15:
                            trends.append(f"📉 **Concerning Decline**: Sales down {abs(trend_change):.1f}% - urgent intervention needed")
                        elif trend_change < -5:
                            trends.append(f"⚠️ **Declining Trend**: Sales down {abs(trend_change):.1f}% - monitor closely")
                        else:
                            trends.append(f"📊 **Stable Trend**: Sales relatively stable ({trend_change:+.1f}%)")
            
            # Long-term trend analysis
            if len(df_sorted) >= 30:
                # Linear regression for overall trend
                x = np.arange(len(df_sorted))
                y = df_sorted[sales_col].values
                
                # Remove NaN values for regression
                mask = ~np.isnan(y)
                if mask.sum() > 10:
                    x_clean, y_clean = x[mask], y[mask]
                    slope, intercept = np.polyfit(x_clean, y_clean, 1)
                    
                    # Convert slope to percentage change per day
                    if y_clean.mean() > 0:
                        daily_change_pct = (slope / y_clean.mean()) * 100
                        monthly_change_pct = daily_change_pct * 30
                        
                        if monthly_change_pct > 10:
                            trends.append(f"📈 **Long-term Growth**: {monthly_change_pct:.1f}% monthly growth trend established")
                        elif monthly_change_pct < -10:
                            trends.append(f"📉 **Long-term Decline**: {abs(monthly_change_pct):.1f}% monthly decline trend - strategic review needed")
            
            # Volatility analysis
            if len(df_sorted) >= 7:
                rolling_std = df_sorted[sales_col].rolling(window=7).std()
                recent_volatility = rolling_std.tail(7).mean()
                overall_volatility = df_sorted[sales_col].std()
                
                if recent_volatility > overall_volatility * 1.5:
                    trends.append("🌊 **Increased Volatility**: Recent sales more unpredictable - investigate external factors")
                elif recent_volatility < overall_volatility * 0.5:
                    trends.append("🎯 **Stabilizing Pattern**: Sales becoming more predictable - good for planning")
            
        except Exception as e:
            logger.error(f"Error in trend analysis: {e}")
            trends.append("❌ Unable to complete trend analysis")
        
        return trends
    
    def _analyze_seasonality(self, df: pd.DataFrame, column_analysis: Dict) -> List[str]:
        """Analyze seasonal patterns in sales"""
        seasonality = []
        
        try:
            sales_col = column_analysis.get('sales_column')
            date_col = column_analysis.get('date_column')
            
            if not sales_col or not date_col:
                return ["Seasonality analysis requires date and sales columns"]
            
            df_temp = df.copy()
            df_temp[date_col] = pd.to_datetime(df_temp[date_col])
            
            # Day of week analysis
            if len(df_temp) >= 14:
                df_temp['day_of_week'] = df_temp[date_col].dt.day_name()
                daily_performance = df_temp.groupby('day_of_week')[sales_col].agg(['mean', 'count'])
                
                # Filter days with sufficient data
                daily_performance = daily_performance[daily_performance['count'] >= 2]
                
                if not daily_performance.empty:
                    best_day = daily_performance['mean'].idxmax()
                    worst_day = daily_performance['mean'].idxmin()
                    best_performance = daily_performance['mean'].max()
                    avg_performance = daily_performance['mean'].mean()
                    
                    if avg_performance > 0:
                        lift_pct = ((best_performance - avg_performance) / avg_performance) * 100
                        seasonality.append(f"📅 **Best Day**: {best_day} performs {lift_pct:.1f}% above average - optimize staffing and inventory")
                    
                    # Weekend vs weekday analysis
                    df_temp['is_weekend'] = df_temp[date_col].dt.dayofweek.isin([5, 6])
                    weekend_avg = df_temp[df_temp['is_weekend']][sales_col].mean()
                    weekday_avg = df_temp[~df_temp['is_weekend']][sales_col].mean()
                    
                    if weekday_avg > 0:
                        weekend_lift = ((weekend_avg - weekday_avg) / weekday_avg) * 100
                        if weekend_lift > 20:
                            seasonality.append(f"🎉 **Weekend Boost**: Weekend sales {weekend_lift:.1f}% higher - focus weekend promotions")
                        elif weekend_lift < -20:
                            seasonality.append(f"💼 **Weekday Focus**: Weekday sales {abs(weekend_lift):.1f}% stronger - target business customers")
            
            # Monthly analysis
            if len(df_temp) >= 60:
                df_temp['month'] = df_temp[date_col].dt.month_name()
                monthly_performance = df_temp.groupby('month')[sales_col].agg(['mean', 'count'])
                monthly_performance = monthly_performance[monthly_performance['count'] >= 3]
                
                if not monthly_performance.empty:
                    best_month = monthly_performance['mean'].idxmax()
                    worst_month = monthly_performance['mean'].idxmin()
                    
                    seasonality.append(f"🗓️ **Seasonal Peak**: {best_month} is strongest month - plan inventory and marketing campaigns")
                    
                    # Quarter analysis
                    quarters = {
                        'Q1': ['January', 'February', 'March'],
                        'Q2': ['April', 'May', 'June'],
                        'Q3': ['July', 'August', 'September'],
                        'Q4': ['October', 'November', 'December']
                    }
                    
                    quarterly_performance = {}
                    for quarter, months in quarters.items():
                        quarter_data = monthly_performance[monthly_performance.index.isin(months)]
                        if not quarter_data.empty:
                            quarterly_performance[quarter] = quarter_data['mean'].mean()
                    
                    if quarterly_performance:
                        best_quarter = max(quarterly_performance, key=quarterly_performance.get)
                        seasonality.append(f"📊 **Quarterly Trend**: {best_quarter} shows strongest performance - align annual planning")
            
            # Special events and holidays (Indian context)
            df_temp['day'] = df_temp[date_col].dt.day
            df_temp['month_num'] = df_temp[date_col].dt.month
            
            # Festival seasons (Diwali, Dussehra period - Oct/Nov)
            festival_season = df_temp[df_temp['month_num'].isin([10, 11])]
            if len(festival_season) >= 5:
                festival_avg = festival_season[sales_col].mean()
                overall_avg = df_temp[sales_col].mean()
                
                if overall_avg > 0:
                    festival_lift = ((festival_avg - overall_avg) / overall_avg) * 100
                    if festival_lift > 15:
                        seasonality.append(f"🪔 **Festival Season**: Oct-Nov sales {festival_lift:.1f}% higher - maximize Diwali opportunities")
            
            # Month-end effect
            month_end_data = df_temp[df_temp['day'] >= 28]
            if len(month_end_data) >= 5:
                month_end_avg = month_end_data[sales_col].mean()
                overall_avg = df_temp[sales_col].mean()
                
                if overall_avg > 0:
                    month_end_lift = ((month_end_avg - overall_avg) / overall_avg) * 100
                    if month_end_lift > 10:
                        seasonality.append(f"📅 **Month-end Effect**: {month_end_lift:.1f}% sales boost - salary-driven purchases")
            
        except Exception as e:
            logger.error(f"Error in seasonality analysis: {e}")
            seasonality.append("❌ Unable to complete seasonality analysis")
        
        return seasonality
    
    def _generate_kpi_dashboard(self, df: pd.DataFrame, column_analysis: Dict) -> Dict[str, Any]:
        """Generate key performance indicators"""
        kpis = {}
        
        try:
            sales_col = column_analysis.get('sales_column')
            date_col = column_analysis.get('date_column')
            
            if sales_col:
                sales_data = df[sales_col].dropna()
                
                if len(sales_data) > 0:
                    kpis['financial_metrics'] = {
                        'total_revenue': {
                            'value': format_currency(sales_data.sum()),
                            'raw_value': float(sales_data.sum())
                        },
                        'average_transaction': {
                            'value': format_currency(sales_data.mean()),
                            'raw_value': float(sales_data.mean())
                        },
                        'median_transaction': {
                            'value': format_currency(sales_data.median()),
                            'raw_value': float(sales_data.median())
                        },
                        'highest_sale': {
                            'value': format_currency(sales_data.max()),
                            'raw_value': float(sales_data.max())
                        }
                    }
                    
                    # Performance metrics
                    kpis['performance_metrics'] = {
                        'total_transactions': len(df),
                        'revenue_per_day': format_currency(sales_data.sum() / len(df)) if len(df) > 0 else "₹0",
                        'coefficient_of_variation': f"{(sales_data.std() / sales_data.mean() * 100):.1f}%" if sales_data.mean() > 0 else "N/A"
                    }
            
            if date_col:
                try:
                    date_data = pd.to_datetime(df[date_col])
                    date_range = date_data.max() - date_data.min()
                    
                    kpis['time_metrics'] = {
                        'data_period_days': date_range.days,
                        'start_date': date_data.min().strftime('%Y-%m-%d'),
                        'end_date': date_data.max().strftime('%Y-%m-%d'),
                        'data_density': f"{len(df) / max(1, date_range.days):.1f} transactions/day"
                    }
                except:
                    kpis['time_metrics'] = {'error': 'Unable to parse date data'}
            
            # Product metrics if available
            product_col = column_analysis.get('product_column')
            if product_col and sales_col and product_col in df.columns:
                product_performance = df.groupby(product_col)[sales_col].agg(['sum', 'count', 'mean'])
                
                kpis['product_metrics'] = {
                    'unique_products': len(product_performance),
                    'top_revenue_product': product_performance['sum'].idxmax() if not product_performance.empty else 'N/A',
                    'top_volume_product': product_performance['count'].idxmax() if not product_performance.empty else 'N/A',
                    'highest_avg_value': product_performance['mean'].idxmax() if not product_performance.empty else 'N/A'
                }
        
        except Exception as e:
            logger.error(f"Error generating KPIs: {e}")
            kpis['error'] = "Unable to generate complete KPI dashboard"
        
        return kpis
    
    def _generate_inventory_insights(self, df: pd.DataFrame, column_analysis: Dict) -> List[str]:
        """Generate inventory-related insights"""
        insights = []
        
        try:
            product_col = column_analysis.get('product_column')
            sales_col = column_analysis.get('sales_column')
            quantity_col = column_analysis.get('quantity_column')
            date_col = column_analysis.get('date_column')
            
            if not product_col or not sales_col or product_col not in df.columns:
                return ["Product and sales data required for inventory insights"]
            
            # Product velocity analysis
            agg_dict = {sales_col: ['sum', 'count', 'mean']}
            if quantity_col and quantity_col in df.columns:
                agg_dict[quantity_col] = 'sum'
            else:
                agg_dict[sales_col + '_count'] = 'count'
            
            product_velocity = df.groupby(product_col).agg(agg_dict).round(2)
            
            # Flatten column names
            if quantity_col and quantity_col in df.columns:
                product_velocity.columns = ['total_revenue', 'transaction_count', 'avg_transaction', 'total_quantity']
            else:
                product_velocity.columns = ['total_revenue', 'transaction_count', 'avg_transaction', 'total_count']
            
            # Top performers
            if not product_velocity.empty:
                top_revenue_products = product_velocity.nlargest(5, 'total_revenue')
                total_revenue_sum = product_velocity['total_revenue'].sum()
                if total_revenue_sum > 0:
                    top_3_share = (top_revenue_products['total_revenue'][:3].sum() / total_revenue_sum * 100)
                    insights.append(f"💰 **Revenue Leaders**: {', '.join(top_revenue_products.index[:3])} drive {top_3_share:.1f}% of revenue")
                
                # Fast movers
                if quantity_col and quantity_col in df.columns and date_col:
                    try:
                        df_temp = df.copy()
                        df_temp[date_col] = pd.to_datetime(df_temp[date_col])
                        recent_period = df_temp[df_temp[date_col] >= (df_temp[date_col].max() - pd.Timedelta(days=7))]
                        if not recent_period.empty:
                            recent_velocity = recent_period.groupby(product_col)[quantity_col].sum()
                            fast_movers = recent_velocity.nlargest(3)
                            
                            for product, qty in fast_movers.items():
                                daily_velocity = qty / 7
                                insights.append(f"🚀 **Fast Mover**: {product} - {daily_velocity:.1f} units/day velocity")
                                break  # Limit to one example
                    except:
                        pass  # Skip if date parsing fails
                
                # Slow movers identification
                slow_movers = product_velocity.nsmallest(5, 'transaction_count')
                if len(slow_movers) > 0:
                    insights.append(f"🐌 **Slow Movers**: Review {', '.join(slow_movers.index[:3])} - low transaction frequency")
                
                # ABC Analysis
                product_velocity_sorted = product_velocity.sort_values('total_revenue', ascending=False)
                cumulative_revenue = product_velocity_sorted['total_revenue'].cumsum()
                total_revenue = product_velocity_sorted['total_revenue'].sum()
                
                if total_revenue > 0:
                    # A products (80% of revenue)
                    a_products = cumulative_revenue[cumulative_revenue <= total_revenue * 0.8]
                    a_percentage = (len(a_products) / len(product_velocity)) * 100
                    insights.append(f"📊 **ABC Analysis**: {len(a_products)} products ({a_percentage:.1f}%) generate 80% of revenue")
            
        except Exception as e:
            logger.error(f"Error in inventory insights: {e}")
            insights.append("❌ Unable to generate complete inventory insights")
        
        return insights
    
    def _identify_revenue_opportunities(self, df: pd.DataFrame, column_analysis: Dict) -> List[str]:
        """Identify revenue optimization opportunities"""
        opportunities = []
        
        try:
            product_col = column_analysis.get('product_column')
            sales_col = column_analysis.get('sales_column')
            quantity_col = column_analysis.get('quantity_column')
            
            if not product_col or not sales_col or product_col not in df.columns:
                return ["Product and sales data required for revenue opportunity analysis"]
            
            # Price analysis if quantity is available
            if quantity_col and quantity_col in df.columns:
                df_price = df[(df[quantity_col] > 0) & (df[quantity_col].notna())].copy()
                if not df_price.empty:
                    df_price['unit_price'] = df_price[sales_col] / df_price[quantity_col]
                    
                    price_analysis = df_price.groupby(product_col)['unit_price'].agg(['mean', 'std', 'count'])
                    
                    # High-value products
                    high_value = price_analysis.nlargest(3, 'mean')
                    if not high_value.empty:
                        opportunities.append(f"💎 **Premium Products**: Focus on {', '.join(high_value.index)} - highest unit values")
                    
                    # Price inconsistency opportunities
                    inconsistent_pricing = price_analysis[price_analysis['std'] > price_analysis['mean'] * 0.2]
                    if not inconsistent_pricing.empty:
                        opportunities.append(f"💰 **Price Optimization**: Standardize pricing for {', '.join(inconsistent_pricing.index[:3])}")
            
            # Cross-selling opportunities (simplified market basket analysis)
            if len(df) >= 50:  # Need sufficient data
                # Customer-based analysis if customer column exists
                customer_col = column_analysis.get('customer_column')
                if customer_col and customer_col in df.columns:
                    customer_products = df.groupby(customer_col)[product_col].apply(lambda x: set(x))
                    
                    # Find common product combinations
                    product_combinations = defaultdict(int)
                    for products in customer_products:
                        if len(products) > 1:
                            for p1 in products:
                                for p2 in products:
                                    if p1 != p2:
                                        combo = tuple(sorted([p1, p2]))
                                        product_combinations[combo] += 1
                    
                    if product_combinations:
                        top_combo = max(product_combinations, key=product_combinations.get)
                        frequency = product_combinations[top_combo]
                        opportunities.append(f"🎁 **Bundle Opportunity**: {top_combo[0]} + {top_combo[1]} bought together {frequency} times")
            
            # Seasonal opportunities
            date_col = column_analysis.get('date_column')
            if date_col and date_col in df.columns:
                try:
                    df_seasonal = df.copy()
                    df_seasonal[date_col] = pd.to_datetime(df_seasonal[date_col])
                    df_seasonal['month'] = df_seasonal[date_col].dt.month
                    
                    monthly_product_sales = df_seasonal.groupby(['month', product_col])[sales_col].sum().unstack(fill_value=0)
                    
                    # Find products with strong seasonal patterns
                    for product in monthly_product_sales.columns:
                        product_sales = monthly_product_sales[product]
                        if product_sales.sum() > 0:
                            peak_month = product_sales.idxmax()
                            peak_value = product_sales.max()
                            avg_value = product_sales.mean()
                            
                            if peak_value > avg_value * 2:  # Strong seasonality
                                month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                                             7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
                                opportunities.append(f"🌟 **Seasonal Focus**: {product} peaks in {month_names.get(peak_month, peak_month)} - plan campaigns")
                                break  # Limit to top opportunity
                except:
                    pass  # Skip if date parsing fails
            
            # Market share opportunities
            product_revenue = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False)
            total_revenue = product_revenue.sum()
            
            # Products with growth potential (good volume, moderate revenue share)
            if len(product_revenue) > 10 and total_revenue > 0:  # Only if we have enough products
                for product in product_revenue.index[5:10]:  # Look at mid-tier products
                    current_share = (product_revenue[product] / total_revenue) * 100
                    if current_share > 2 and current_share < 10:  # Sweet spot for growth
                        opportunities.append(f"📈 **Growth Potential**: {product} has {current_share:.1f}% market share - growth opportunity")
                        break
            
        except Exception as e:
            logger.error(f"Error identifying revenue opportunities: {e}")
            opportunities.append("❌ Unable to identify complete revenue opportunities")
        
        return opportunities
    
    def _analyze_forecast_insights(self, forecast_results: Dict) -> List[str]:
        """Analyze forecast results for business insights"""
        insights = []
        
        try:
            if not forecast_results.get('success'):
                return ["Forecast analysis unavailable"]
            
            forecast_data = forecast_results.get('forecast', [])
            if not forecast_data:
                return ["No forecast data available"]
            
            # Convert to DataFrame for analysis
            forecast_df = pd.DataFrame(forecast_data)
            
            # Ensure required columns exist
            if 'ds' not in forecast_df.columns or 'yhat' not in forecast_df.columns:
                return ["Invalid forecast data structure"]
                
            forecast_df['ds'] = pd.to_datetime(forecast_df['ds'])
            
            # Growth projections
            future_sales = forecast_df['yhat'].values
            if len(future_sales) >= 7:
                if future_sales[0] > 0:  # Avoid division by zero
                    weekly_growth = ((future_sales[6] - future_sales[0]) / future_sales[0]) * 100
                    if weekly_growth > 10:
                        insights.append(f"🚀 **Growth Forecast**: {weekly_growth:.1f}% growth expected next week")
                    elif weekly_growth < -10:
                        insights.append(f"⚠️ **Decline Forecast**: {abs(weekly_growth):.1f}% decline projected - take action")
            
            # Revenue projections
            if len(future_sales) >= 30:
                monthly_revenue = sum(future_sales[:30])
                insights.append(f"💰 **Revenue Projection**: {format_currency(monthly_revenue)} forecasted for next 30 days")
            
            # Confidence intervals analysis
            if 'yhat_lower' in forecast_df.columns and 'yhat_upper' in forecast_df.columns:
                # Calculate relative uncertainty
                forecast_df_clean = forecast_df[forecast_df['yhat'] > 0]  # Avoid division by zero
                if not forecast_df_clean.empty:
                    avg_uncertainty = ((forecast_df_clean['yhat_upper'] - forecast_df_clean['yhat_lower']) / forecast_df_clean['yhat']).mean()
                    if avg_uncertainty > 0.5:
                        insights.append("🎯 **High Uncertainty**: Wide prediction intervals - monitor closely and adjust plans")
                    else:
                        insights.append("✅ **Reliable Forecast**: Narrow confidence intervals indicate high prediction accuracy")
            
            # Trend analysis in forecast
            if len(future_sales) >= 14:
                early_period = np.mean(future_sales[:7])
                later_period = np.mean(future_sales[7:14])
                
                if later_period > early_period * 1.05:
                    insights.append("📈 **Accelerating Growth**: Forecast shows increasing momentum")
                elif later_period < early_period * 0.95:
                    insights.append("📉 **Deceleration Alert**: Forecast shows slowing growth trend")
            
        except Exception as e:
            logger.error(f"Error in forecast insights: {e}")
            insights.append("❌ Unable to analyze forecast data")
        
        return insights
    
    def _generate_action_items(self, insights: Dict) -> List[str]:
        """Generate specific action items based on insights"""
        actions = []
        
        try:
            # Priority actions based on risk alerts
            risk_alerts = insights.get('risk_alerts', [])
            if risk_alerts:
                actions.append("🚨 **Immediate**: Address risk alerts - review declining trends and operational issues")
            
            # Inventory actions
            inventory_alerts = insights.get('inventory_alerts', [])
            if any("slow mover" in alert.lower() for alert in inventory_alerts):
                actions.append("📦 **This Week**: Review slow-moving inventory - consider promotions or bundle deals")
            
            # Revenue optimization actions
            revenue_opportunities = insights.get('revenue_opportunities', [])
            if any("bundle" in opp.lower() for opp in revenue_opportunities):
                actions.append("🎁 **Marketing**: Implement cross-selling campaigns for frequently bought together items")
            
            # Seasonal actions
            seasonality_insights = insights.get('seasonality_insights', [])
            if any("weekend" in insight.lower() for insight in seasonality_insights):
                actions.append("📅 **Operations**: Optimize weekend staffing and inventory based on performance patterns")
            
            # Forecasting actions
            forecast_insights = insights.get('forecast_insights', [])
            if any("decline" in insight.lower() for insight in forecast_insights):
                actions.append("📊 **Planning**: Develop contingency plans for projected sales decline")
            
            # Always include data quality action
            actions.append("📈 **Ongoing**: Continue data collection for improved forecasting accuracy")
            
        except Exception as e:
            logger.error(f"Error generating action items: {e}")
            actions.append("❌ Review insights manually to determine action items")
        
        return actions
    
    def _identify_risks(self, df: pd.DataFrame, column_analysis: Dict, forecast_results: Dict = None) -> List[str]:
        """Identify business risks from data patterns"""
        risks = []
        
        try:
            sales_col = column_analysis.get('sales_column')
            date_col = column_analysis.get('date_column')
            
            if sales_col and sales_col in df.columns:
                sales_data = df[sales_col].dropna()
                
                if len(sales_data) > 0:
                    # Revenue concentration risk
                    product_col = column_analysis.get('product_column')
                    if product_col and product_col in df.columns:
                        product_revenue = df.groupby(product_col)[sales_col].sum()
                        total_revenue = product_revenue.sum()
                        if total_revenue > 0:
                            top_product_share = (product_revenue.max() / total_revenue) * 100
                            
                            if top_product_share > 50:
                                risks.append(f"⚠️ **Concentration Risk**: Single product drives {top_product_share:.1f}% of revenue")
                    
                    # Volatility risk
                    if sales_data.mean() > 0:
                        cv = sales_data.std() / sales_data.mean()
                        if cv > 1.0:
                            risks.append("🌊 **Volatility Risk**: Highly unpredictable sales patterns detected")
                    
                    # Declining trend risk
                    if len(df) >= 14 and date_col and date_col in df.columns:
                        try:
                            df_sorted = df.sort_values(date_col)
                            recent_avg = df_sorted[sales_col].tail(7).mean()
                            previous_avg = df_sorted[sales_col].tail(14).head(7).mean()
                            
                            if previous_avg > 0:
                                decline_rate = ((previous_avg - recent_avg) / previous_avg) * 100
                                if decline_rate > 20:
                                    risks.append(f"📉 **Revenue Risk**: {decline_rate:.1f}% sales decline in recent period")
                        except:
                            pass  # Skip if date sorting fails
                    
                    # Zero sales risk
                    zero_sales = len(sales_data[sales_data == 0])
                    if zero_sales > len(df) * 0.1:  # More than 10% zero sales days
                        risks.append(f"🚫 **Operational Risk**: {zero_sales} days with zero sales detected")
            
            # Forecast-based risks
            if forecast_results and forecast_results.get('success'):
                forecast_data = forecast_results.get('forecast', [])
                if forecast_data:
                    try:
                        future_values = [item['yhat'] for item in forecast_data if 'yhat' in item]
                        if len(future_values) >= 7:
                            if any(val < 0 for val in future_values[:7]):
                                risks.append("📊 **Forecast Risk**: Negative sales projections in forecast")
                    except:
                        pass  # Skip if forecast data is malformed
            
        except Exception as e:
            logger.error(f"Error identifying risks: {e}")
            risks.append("❌ Unable to complete risk analysis")
        
        return risks
    
    def _get_data_period(self, df: pd.DataFrame, column_analysis: Dict) -> Dict[str, str]:
        """Get the time period covered by the data"""
        try:
            date_col = column_analysis.get('date_column')
            if date_col and date_col in df.columns:
                dates = pd.to_datetime(df[date_col])
                return {
                    'start_date': dates.min().strftime('%Y-%m-%d'),
                    'end_date': dates.max().strftime('%Y-%m-%d'),
                    'duration_days': (dates.max() - dates.min()).days
                }
        except Exception as e:
            logger.error(f"Error getting data period: {e}")
        
        return {
            'start_date': 'Unknown',
            'end_date': 'Unknown',
            'duration_days': 0
        }
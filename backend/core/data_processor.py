"""
Smart Data Processor for Retail Intelligence Platform
Automatically detects and processes retail CSV data with Indian market optimization
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import re

from config import settings, format_currency

logger = logging.getLogger(__name__)

class RetailDataProcessor:
    """Intelligent retail data processor with automatic column detection"""
    
    def __init__(self):
        self.date_columns = ['date', 'order_date', 'sale_date', 'transaction_date', 'created_at', 'timestamp']
        self.sales_columns = ['sales', 'revenue', 'amount', 'total', 'value', 'price', 'cost']
        self.quantity_columns = ['quantity', 'qty', 'units', 'count', 'volume']
        self.product_columns = ['product', 'item', 'sku', 'product_name', 'item_name', 'category']
        self.customer_columns = ['customer', 'customer_id', 'user_id', 'buyer']
        
    def process_csv(self, file_path: str) -> Dict[str, Any]:
        """
        Main processing function - analyzes and cleans retail CSV data
        
        Args:
            file_path: Path to the CSV file
            
        Returns:
            Dictionary containing processed data and metadata
        """
        try:
            # Read CSV with multiple encodings fallback
            df = self._read_csv_smart(file_path)
            
            # Basic data validation
            if df.empty:
                raise ValueError("CSV file is empty")
                
            logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # Analyze column structure
            column_analysis = self._analyze_columns(df)
            
            # Clean and process data
            df_cleaned = self._clean_data(df, column_analysis)
            
            # Validate for forecasting requirements
            validation_results = self._validate_for_forecasting(df_cleaned, column_analysis)
            
            # Generate data quality score
            quality_score = self._calculate_quality_score(df_cleaned, validation_results)
            
            # Create summary statistics
            summary_stats = self._generate_summary_stats(df_cleaned, column_analysis)
            
            return {
                'success': True,
                'data': df_cleaned,
                'original_shape': df.shape,
                'processed_shape': df_cleaned.shape,
                'column_analysis': column_analysis,
                'validation_results': validation_results,
                'quality_score': quality_score,
                'summary_stats': summary_stats,
                'recommendations': self._generate_recommendations(validation_results, quality_score)
            }
            
        except Exception as e:
            logger.error(f"Error processing CSV: {e}")
            return {
                'success': False,
                'error': str(e),
                'data': None
            }
    
    def _read_csv_smart(self, file_path: str) -> pd.DataFrame:
        """Smart CSV reading with encoding detection and error handling"""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                logger.info(f"Successfully read CSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                logger.warning(f"Failed to read CSV with {encoding}: {e}")
                continue
        
        raise ValueError("Unable to read CSV file with any supported encoding")
    
    def _analyze_columns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze and categorize columns automatically"""
        analysis = {
            'date_column': None,
            'sales_column': None,
            'quantity_column': None,
            'product_column': None,
            'customer_column': None,
            'numeric_columns': [],
            'categorical_columns': [],
            'detected_patterns': {}
        }
        
        for col in df.columns:
            col_lower = col.lower().strip()
            col_data = df[col].dropna()
            
            # Detect date columns
            if any(date_word in col_lower for date_word in self.date_columns):
                analysis['date_column'] = col
                analysis['detected_patterns'][col] = 'date'
            elif self._is_date_column(col_data):
                analysis['date_column'] = col
                analysis['detected_patterns'][col] = 'date_inferred'
            
            # Detect sales/revenue columns
            elif any(sales_word in col_lower for sales_word in self.sales_columns):
                if analysis['sales_column'] is None:  # Take first match
                    analysis['sales_column'] = col
                    analysis['detected_patterns'][col] = 'sales'
            
            # Detect quantity columns
            elif any(qty_word in col_lower for qty_word in self.quantity_columns):
                if analysis['quantity_column'] is None:
                    analysis['quantity_column'] = col
                    analysis['detected_patterns'][col] = 'quantity'
            
            # Detect product columns
            elif any(prod_word in col_lower for prod_word in self.product_columns):
                if analysis['product_column'] is None:
                    analysis['product_column'] = col
                    analysis['detected_patterns'][col] = 'product'
            
            # Detect customer columns
            elif any(cust_word in col_lower for cust_word in self.customer_columns):
                if analysis['customer_column'] is None:
                    analysis['customer_column'] = col
                    analysis['detected_patterns'][col] = 'customer'
            
            # Categorize as numeric or categorical
            if pd.api.types.is_numeric_dtype(col_data):
                analysis['numeric_columns'].append(col)
            else:
                analysis['categorical_columns'].append(col)
        
        # If no sales column detected, use the first numeric column
        if analysis['sales_column'] is None and analysis['numeric_columns']:
            analysis['sales_column'] = analysis['numeric_columns'][0]
            analysis['detected_patterns'][analysis['numeric_columns'][0]] = 'sales_inferred'
        
        return analysis
    
    def _is_date_column(self, series: pd.Series) -> bool:
        """Check if a series contains date-like data"""
        if len(series) == 0:
            return False
            
        # Sample a few values to test
        sample_size = min(10, len(series))
        sample = series.head(sample_size)
        
        date_count = 0
        for value in sample:
            try:
                pd.to_datetime(str(value))
                date_count += 1
            except:
                continue
        
        return date_count / sample_size > 0.8  # 80% of samples should be dates
    
    def _clean_data(self, df: pd.DataFrame, column_analysis: Dict) -> pd.DataFrame:
        """Clean and standardize the data"""
        df_cleaned = df.copy()
        
        # Clean date column
        if column_analysis['date_column']:
            df_cleaned[column_analysis['date_column']] = pd.to_datetime(
                df_cleaned[column_analysis['date_column']], 
                errors='coerce'
            )
            # Remove rows with invalid dates
            df_cleaned = df_cleaned.dropna(subset=[column_analysis['date_column']])
        
        # Clean sales column
        if column_analysis['sales_column']:
            sales_col = column_analysis['sales_column']
            # Remove currency symbols and convert to numeric
            if df_cleaned[sales_col].dtype == 'object':
                df_cleaned[sales_col] = df_cleaned[sales_col].astype(str).str.replace(r'[₹$,]', '', regex=True)
            df_cleaned[sales_col] = pd.to_numeric(df_cleaned[sales_col], errors='coerce')
            # Remove negative sales and outliers
            df_cleaned = df_cleaned[df_cleaned[sales_col] >= 0]
        
        # Clean quantity column
        if column_analysis['quantity_column']:
            qty_col = column_analysis['quantity_column']
            df_cleaned[qty_col] = pd.to_numeric(df_cleaned[qty_col], errors='coerce')
            df_cleaned = df_cleaned[df_cleaned[qty_col] > 0]  # Remove zero/negative quantities
        
        # Remove duplicate rows
        df_cleaned = df_cleaned.drop_duplicates()
        
        # Sort by date if available
        if column_analysis['date_column']:
            df_cleaned = df_cleaned.sort_values(column_analysis['date_column'])
        
        logger.info(f"Data cleaned: {len(df)} → {len(df_cleaned)} rows")
        return df_cleaned
    
    def _validate_for_forecasting(self, df: pd.DataFrame, column_analysis: Dict) -> Dict[str, Any]:
        """Validate data quality for forecasting"""
        validation = {
            'has_date_column': column_analysis['date_column'] is not None,
            'has_sales_column': column_analysis['sales_column'] is not None,
            'has_sufficient_data': len(df) >= 30,  # Minimum 30 data points
            'date_range_days': 0,
            'missing_data_percentage': 0,
            'seasonal_potential': False,
            'trend_potential': False,
            'issues': []
        }
        
        if validation['has_date_column'] and validation['has_sales_column']:
            date_col = column_analysis['date_column']
            sales_col = column_analysis['sales_column']
            
            # Calculate date range
            date_range = df[date_col].max() - df[date_col].min()
            validation['date_range_days'] = date_range.days
            
            # Check for sufficient historical data
            if validation['date_range_days'] < 90:
                validation['issues'].append("Less than 3 months of historical data")
            
            # Calculate missing data percentage
            total_possible_days = validation['date_range_days']
            actual_data_points = len(df)
            if total_possible_days > 0:
                validation['missing_data_percentage'] = max(0, (total_possible_days - actual_data_points) / total_possible_days * 100)
            
            # Check for seasonal patterns (weekly/monthly cycles)
            if validation['date_range_days'] >= 60:  # At least 2 months
                validation['seasonal_potential'] = True
            
            # Check for trend potential
            if len(df) >= 14:  # At least 2 weeks of data
                sales_trend = df[sales_col].rolling(window=7).mean()
                if not sales_trend.dropna().empty:
                    validation['trend_potential'] = True
        
        # Add specific issues
        if not validation['has_date_column']:
            validation['issues'].append("No date column detected")
        if not validation['has_sales_column']:
            validation['issues'].append("No sales/revenue column detected")
        if not validation['has_sufficient_data']:
            validation['issues'].append(f"Insufficient data points: {len(df)} (minimum 30 required)")
        
        return validation
    
    def _calculate_quality_score(self, df: pd.DataFrame, validation: Dict) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 0
        
        # Base requirements (50 points)
        if validation['has_date_column']:
            score += 20
        if validation['has_sales_column']:
            score += 20
        if validation['has_sufficient_data']:
            score += 10
        
        # Data completeness (20 points)
        missing_percentage = validation.get('missing_data_percentage', 100)
        completeness_score = max(0, (100 - missing_percentage) / 100 * 20)
        score += completeness_score
        
        # Historical data range (15 points)
        date_range_days = validation.get('date_range_days', 0)
        if date_range_days >= 365:  # 1+ years
            score += 15
        elif date_range_days >= 180:  # 6+ months
            score += 10
        elif date_range_days >= 90:  # 3+ months
            score += 5
        
        # Forecasting potential (15 points)
        if validation.get('seasonal_potential'):
            score += 7
        if validation.get('trend_potential'):
            score += 8
        
        return min(100, score)
    
    def _generate_summary_stats(self, df: pd.DataFrame, column_analysis: Dict) -> Dict[str, Any]:
        """Generate comprehensive summary statistics"""
        stats = {
            'total_records': len(df),
            'date_range': {},
            'sales_summary': {},
            'top_products': [],
            'time_patterns': {}
        }
        
        # Date range analysis
        if column_analysis['date_column']:
            date_col = column_analysis['date_column']
            stats['date_range'] = {
                'start_date': df[date_col].min().strftime('%Y-%m-%d'),
                'end_date': df[date_col].max().strftime('%Y-%m-%d'),
                'total_days': (df[date_col].max() - df[date_col].min()).days,
                'data_points': len(df)
            }
        
        # Sales summary
        if column_analysis['sales_column']:
            sales_col = column_analysis['sales_column']
            sales_data = df[sales_col].dropna()
            
            stats['sales_summary'] = {
                'total_sales': format_currency(sales_data.sum()),
                'average_transaction': format_currency(sales_data.mean()),
                'median_transaction': format_currency(sales_data.median()),
                'max_transaction': format_currency(sales_data.max()),
                'min_transaction': format_currency(sales_data.min()),
                'std_deviation': format_currency(sales_data.std())
            }
        
        # Top products analysis
        if column_analysis['product_column']:
            product_col = column_analysis['product_column']
            sales_col = column_analysis['sales_column']
            
            if sales_col:
                top_products = df.groupby(product_col)[sales_col].agg(['sum', 'count']).reset_index()
                top_products = top_products.sort_values('sum', ascending=False).head(10)
                
                stats['top_products'] = [
                    {
                        'product': row[product_col],
                        'total_sales': format_currency(row['sum']),
                        'transaction_count': int(row['count'])
                    }
                    for _, row in top_products.iterrows()
                ]
        
        # Time patterns
        if column_analysis['date_column'] and column_analysis['sales_column']:
            date_col = column_analysis['date_column']
            sales_col = column_analysis['sales_column']
            
            df['day_of_week'] = df[date_col].dt.day_name()
            df['month'] = df[date_col].dt.month_name()
            
            # Weekly patterns
            weekly_sales = df.groupby('day_of_week')[sales_col].mean()
            best_day = weekly_sales.idxmax()
            worst_day = weekly_sales.idxmin()
            
            # Monthly patterns
            monthly_sales = df.groupby('month')[sales_col].mean()
            best_month = monthly_sales.idxmax()
            worst_month = monthly_sales.idxmin()
            
            stats['time_patterns'] = {
                'best_day_of_week': best_day,
                'worst_day_of_week': worst_day,
                'best_month': best_month,
                'worst_month': worst_month,
                'weekend_vs_weekday': self._analyze_weekend_patterns(df, date_col, sales_col)
            }
        
        return stats
    
    def _analyze_weekend_patterns(self, df: pd.DataFrame, date_col: str, sales_col: str) -> Dict:
        """Analyze weekend vs weekday sales patterns"""
        df['is_weekend'] = df[date_col].dt.dayofweek.isin([5, 6])  # Saturday, Sunday
        
        weekend_avg = df[df['is_weekend']][sales_col].mean()
        weekday_avg = df[~df['is_weekend']][sales_col].mean()
        
        if weekday_avg > 0:
            weekend_lift = ((weekend_avg - weekday_avg) / weekday_avg) * 100
        else:
            weekend_lift = 0
        
        return {
            'weekend_average': format_currency(weekend_avg),
            'weekday_average': format_currency(weekday_avg),
            'weekend_lift_percentage': round(weekend_lift, 2)
        }
    
    def _generate_recommendations(self, validation: Dict, quality_score: float) -> List[str]:
        """Generate actionable recommendations for data improvement"""
        recommendations = []
        
        if quality_score < 60:
            recommendations.append("⚠️ Data quality is below recommended threshold. Consider data cleaning.")
        
        if validation.get('date_range_days', 0) < 90:
            recommendations.append("📅 Collect more historical data (minimum 3 months) for better forecasting accuracy.")
        
        if validation.get('missing_data_percentage', 0) > 20:
            recommendations.append("📊 High percentage of missing data detected. Ensure consistent data collection.")
        
        if not validation.get('seasonal_potential'):
            recommendations.append("🔄 Limited seasonal analysis possible. Collect data over multiple seasons.")
        
        if len(validation.get('issues', [])) > 0:
            recommendations.append("🔧 Address data structure issues before proceeding with forecasting.")
        
        # Positive recommendations
        if quality_score >= 80:
            recommendations.append("✅ Excellent data quality! Ready for advanced forecasting.")
        elif quality_score >= 60:
            recommendations.append("✅ Good data quality. Suitable for reliable forecasting.")
        
        if validation.get('seasonal_potential') and validation.get('trend_potential'):
            recommendations.append("🎯 Data shows both trend and seasonal patterns - perfect for Prophet model.")
        
        return recommendations
    
    def aggregate_for_forecasting(self, df: pd.DataFrame, column_analysis: Dict, 
                                frequency: str = 'daily') -> pd.DataFrame:
        """Aggregate data for time series forecasting"""
        if not column_analysis['date_column'] or not column_analysis['sales_column']:
            raise ValueError("Date and sales columns required for forecasting aggregation")
        
        date_col = column_analysis['date_column']
        sales_col = column_analysis['sales_column']
        
        # Set date as index
        df_agg = df.copy()
        df_agg = df_agg.set_index(date_col)
        
        # Aggregate based on frequency
        freq_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M'
        }
        
        agg_freq = freq_map.get(frequency, 'D')
        
        # Aggregate sales data
        aggregated = df_agg.groupby(pd.Grouper(freq=agg_freq)).agg({
            sales_col: 'sum'
        }).reset_index()
        
        # Fill missing dates with zero sales
        date_range = pd.date_range(
            start=aggregated[date_col].min(),
            end=aggregated[date_col].max(),
            freq=agg_freq
        )
        
        full_range = pd.DataFrame({date_col: date_range})
        aggregated = full_range.merge(aggregated, on=date_col, how='left')
        aggregated[sales_col] = aggregated[sales_col].fillna(0)
        
        # Rename columns for standard format
        aggregated = aggregated.rename(columns={
            date_col: 'ds',
            sales_col: 'y'
        })
        
        logger.info(f"Aggregated data to {frequency} frequency: {len(aggregated)} data points")
        return aggregated

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

def generate_sample_retail_data():
    """Generate comprehensive sample retail data"""
    
    # Configuration
    start_date = datetime.now() - timedelta(days=180)  # 6 months of data
    end_date = datetime.now()
    
    # Indian retail products
    products = [
        'Basmati Rice', 'Wheat Flour', 'Sugar', 'Tea', 'Coffee',
        'Cooking Oil', 'Spices Mix', 'Pulses', 'Milk', 'Yogurt',
        'Biscuits', 'Bread', 'Eggs', 'Chicken', 'Fish',
        'Vegetables', 'Fruits', 'Onions', 'Potatoes', 'Tomatoes'
    ]
    
    # Base prices in INR
    base_prices = {
        'Basmati Rice': 80, 'Wheat Flour': 40, 'Sugar': 45, 'Tea': 120, 'Coffee': 200,
        'Cooking Oil': 150, 'Spices Mix': 80, 'Pulses': 100, 'Milk': 50, 'Yogurt': 60,
        'Biscuits': 30, 'Bread': 25, 'Eggs': 5, 'Chicken': 180, 'Fish': 220,
        'Vegetables': 40, 'Fruits': 80, 'Onions': 30, 'Potatoes': 25, 'Tomatoes': 40
    }
    
    # Customer IDs
    customers = [f'CUST{i:04d}' for i in range(1, 301)]  # 300 customers
    
    data = []
    
    # Generate daily data
    current_date = start_date
    while current_date <= end_date:
        # Number of transactions per day (more on weekends and festivals)
        base_transactions = 20
        
        # Weekend boost
        if current_date.weekday() >= 5:  # Saturday, Sunday
            base_transactions = int(base_transactions * 1.4)
        
        # Festival season boost (October-November)
        if current_date.month in [10, 11]:
            base_transactions = int(base_transactions * 1.6)
        
        # Month-end boost (salary days)
        if current_date.day >= 28:
            base_transactions = int(base_transactions * 1.3)
        
        num_transactions = np.random.poisson(base_transactions)
        
        for _ in range(num_transactions):
            product = random.choice(products)
            customer = random.choice(customers)
            
            # Price calculation with market variations
            base_price = base_prices[product]
            
            # Random price variation (±15%)
            price_variation = np.random.uniform(0.85, 1.15)
            unit_price = base_price * price_variation
            
            # Quantity (realistic for each product type)
            if product in ['Basmati Rice', 'Wheat Flour', 'Sugar', 'Cooking Oil']:
                quantity = np.random.choice([1, 2, 5], p=[0.6, 0.3, 0.1])
            elif product in ['Spices Mix', 'Tea', 'Coffee']:
                quantity = np.random.choice([1, 2], p=[0.8, 0.2])
            else:
                quantity = np.random.choice([1, 2, 3, 5], p=[0.5, 0.3, 0.15, 0.05])
            
            total_sales = unit_price * quantity
            
            # Add seasonal variations
            if current_date.month in [10, 11]:  # Festival season
                total_sales *= 1.2
            elif current_date.month in [6, 7, 8]:  # Monsoon season
                total_sales *= 0.9
            
            data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'sales': round(total_sales, 2),
                'product': product,
                'quantity': quantity,
                'customer': customer,
                'unit_price': round(unit_price, 2)
            })
        
        current_date += timedelta(days=1)
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Add some realistic patterns
    # 1. Customer loyalty (some customers buy more frequently)
    loyal_customers = random.sample(customers, 50)  # 50 loyal customers
    
    # 2. Product seasonality
    # Add more vegetable sales in winter, more fruits in summer
    
    # Save to CSV
    output_file = 'data/samples/indian_retail_sample.csv'
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    # Generate summary
    summary = {
        'total_records': len(df),
        'date_range': f"{df['date'].min()} to {df['date'].max()}",
        'total_sales': df['sales'].sum(),
        'unique_products': df['product'].nunique(),
        'unique_customers': df['customer'].nunique(),
        'avg_transaction': df['sales'].mean(),
        'max_transaction': df['sales'].max()
    }
    
    print("✅ Sample data generated successfully!")
    print(f"📄 File: {output_file}")
    print(f"📊 Records: {summary['total_records']:,}")
    print(f"📅 Period: {summary['date_range']}")
    print(f"💰 Total Sales: ₹{summary['total_sales']:,.2f}")
    print(f"🛍️ Products: {summary['unique_products']}")
    print(f"👥 Customers: {summary['unique_customers']}")
    print(f"💳 Avg Transaction: ₹{summary['avg_transaction']:.2f}")
    
    return df, summary

if __name__ == "__main__":
    generate_sample_retail_data()
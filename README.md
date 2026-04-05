# Sales & Demand Forecasting

Predicts daily sales for retail stores using historical sales data and machine learning. 

## Features
- Handles missing and incorrect data  
- Creates time-based features: Month, Weekday, Season  
- Lag & rolling features to capture past behavior  
- Forecasts next 30 days of sales  

## Tools
- Python, Pandas, NumPy  
- Scikit-learn (Random Forest Regression)  
- Matplotlib, Seaborn  

## Model Evaluation
- RMSE ≈ 26 units  
- R² ≈ 0.94  
- Key features: `Sales_lag_1`, `Rolling_mean_3`, `Promo_Impact`  

## Business Insights
- Daily sales fluctuate → demand is unstable  
- Discounts & promotions influence sales  
- Recommendations: maintain safety stock, use promotions strategically

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic house price dataset
def generate_house_data(n_samples=500):
    np.random.seed(42)
    
    # Generate features
    size = np.random.normal(1500, 500, n_samples)  # Square feet
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples)
    age = np.random.randint(0, 50, n_samples)  # Years old
    location_score = np.random.uniform(1, 10, n_samples)  # 1-10 score
    distance_to_city = np.random.uniform(1, 30, n_samples)  # Miles
    
    # Generate price with some noise
    price = (
        100 * size +  # Base price per sq ft
        50000 * bedrooms +  # Bedroom value
        30000 * bathrooms +  # Bathroom value
        -2000 * age +  # Depreciation
        15000 * location_score +  # Location premium
        -1000 * distance_to_city +  # Distance penalty
        np.random.normal(0, 50000, n_samples)  # Noise
    )
    
    # Ensure minimum price
    price = np.maximum(price, 50000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'size_sqft': size,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'age_years': age,
        'location_score': location_score,
        'distance_to_city_miles': distance_to_city,
        'price': price
    })
    
    return data

# Create and save dataset
def create_and_save_model():
    print("Generating house price dataset...")
    df = generate_house_data(500)
    
    # Save dataset
    df.to_csv('house_data.csv', index=False)
    print(f"Dataset saved with {len(df)} records")
    
    # Prepare features and target
    X = df.drop('price', axis=1)
    y = df['price']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Performance:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R² Score: {r2:.4f}")
    
    # Save model and scaler
    joblib.dump(model, 'house_price_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print("\nModel saved as 'house_price_model.pkl'")
    print("Scaler saved as 'scaler.pkl'")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', ascending=False)
    
    print("\nFeature Importance:")
    print(feature_importance)
    
    return df, model, scaler, feature_importance

if __name__ == "__main__":
    df, model, scaler, feature_importance = create_and_save_model()
    
    # Sample prediction
    sample_house = pd.DataFrame([{
        'size_sqft': 1800,
        'bedrooms': 3,
        'bathrooms': 2,
        'age_years': 10,
        'location_score': 7.5,
        'distance_to_city_miles': 5
    }])
    
    sample_scaled = scaler.transform(sample_house)
    predicted_price = model.predict(sample_scaled)[0]
    
    print(f"\nSample Prediction:")
    print(f"Features: {sample_house.iloc[0].to_dict()}")
    print(f"Predicted Price: ${predicted_price:,.2f}")
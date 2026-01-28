import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="🏠 House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 House Price Prediction AI")
st.markdown("Predict house prices using machine learning with Linear Regression")

# Load model and data
@st.cache_resource
def load_model():
    try:
        model = joblib.load('house_price_model.pkl')
        scaler = joblib.load('scaler.pkl')
        df = pd.read_csv('house_data.csv')
        return model, scaler, df
    except:
        st.error("Please run 'house_price_model.py' first to generate the model!")
        return None, None, None

model, scaler, df = load_model()

if model is not None:
    # Create sidebar for input
    st.sidebar.header("🏡 House Specifications")
    
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        size = st.number_input(
            "Size (sq ft)",
            min_value=500,
            max_value=5000,
            value=1800,
            step=100
        )
        
        bedrooms = st.selectbox(
            "Bedrooms",
            options=[1, 2, 3, 4, 5],
            index=2
        )
        
        bathrooms = st.selectbox(
            "Bathrooms",
            options=[1, 1.5, 2, 2.5, 3, 3.5, 4],
            index=2
        )
    
    with col2:
        age = st.slider(
            "Age (years)",
            min_value=0,
            max_value=50,
            value=10,
            step=1
        )
        
        location = st.slider(
            "Location Score (1-10)",
            min_value=1.0,
            max_value=10.0,
            value=7.5,
            step=0.5
        )
        
        distance = st.slider(
            "Distance to City (miles)",
            min_value=1.0,
            max_value=30.0,
            value=5.0,
            step=0.5
        )
    
    # Predict button
    predict_btn = st.sidebar.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Prediction", "📈 Data Insights", "📋 Dataset", "ℹ️ About"])
    
    with tab1:
        st.header("Price Prediction")
        
        if predict_btn:
            # Prepare input
            input_data = pd.DataFrame([{
                'size_sqft': size,
                'bedrooms': bedrooms,
                'bathrooms': bathrooms,
                'age_years': age,
                'location_score': location,
                'distance_to_city_miles': distance
            }])
            
            # Scale and predict
            input_scaled = scaler.transform(input_data)
            predicted_price = model.predict(input_scaled)[0]
            
            # Display result
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Price", f"${predicted_price:,.2f}")
            
            with col2:
                price_per_sqft = predicted_price / size
                st.metric("Price per sq ft", f"${price_per_sqft:,.2f}")
            
            with col3:
                avg_price = df['price'].mean()
                diff_percent = ((predicted_price - avg_price) / avg_price) * 100
                st.metric("Vs Average", f"{diff_percent:+.1f}%")
            
            # House details card
            st.subheader("📝 House Details")
            details_col1, details_col2 = st.columns(2)
            
            with details_col1:
                st.write(f"**Size:** {size:,} sq ft")
                st.write(f"**Bedrooms:** {bedrooms}")
                st.write(f"**Bathrooms:** {bathrooms}")
            
            with details_col2:
                st.write(f"**Age:** {age} years")
                st.write(f"**Location Score:** {location}/10")
                st.write(f"**Distance to City:** {distance} miles")
            
            # Feature importance visualization
            st.subheader("📈 Feature Impact on Price")
            
            feature_names = ['size_sqft', 'bedrooms', 'bathrooms', 'age_years', 
                           'location_score', 'distance_to_city_miles']
            coefficients = model.coef_
            
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.barh(feature_names, coefficients)
            ax.set_xlabel('Impact on Price (Coefficient)')
            ax.set_title('How Each Feature Affects House Price')
            ax.axvline(x=0, color='gray', linestyle='--')
            
            # Color bars
            for i, bar in enumerate(bars):
                if coefficients[i] >= 0:
                    bar.set_color('green')
                else:
                    bar.set_color('red')
            
            st.pyplot(fig)
            
        else:
            st.info("👈 Enter house specifications in the sidebar and click 'Predict Price'")
    
    with tab2:
        st.header("Data Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Price distribution
            fig1 = px.histogram(df, x='price', nbins=50,
                              title='House Price Distribution',
                              labels={'price': 'Price ($)'})
            fig1.update_layout(showlegend=False)
            st.plotly_chart(fig1, use_container_width=True)
        
        with col2:
            # Size vs Price scatter
            fig2 = px.scatter(df, x='size_sqft', y='price',
                            color='bedrooms',
                            title='Size vs Price (Colored by Bedrooms)',
                            labels={'size_sqft': 'Size (sq ft)', 'price': 'Price ($)'})
            st.plotly_chart(fig2, use_container_width=True)
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        corr_matrix = df.corr()
        fig3 = px.imshow(corr_matrix, 
                        text_auto=True,
                        aspect="auto",
                        title="Correlation Matrix")
        st.plotly_chart(fig3, use_container_width=True)
    
    with tab3:
        st.header("House Dataset")
        
        # Show data
        st.dataframe(df.head(100), use_container_width=True)
        
        # Statistics
        st.subheader("Dataset Statistics")
        st.dataframe(df.describe(), use_container_width=True)
        
        # Download option
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Dataset (CSV)",
            data=csv,
            file_name="house_price_dataset.csv",
            mime="text/csv"
        )
    
    with tab4:
        st.header("About This Model")
        
        st.markdown("""
        ### 🏠 House Price Prediction Model
        
        This machine learning model predicts house prices based on various features:
        
        **📊 Features Used:**
        - **Size (sq ft)**: Total living area
        - **Bedrooms**: Number of bedrooms
        - **Bathrooms**: Number of bathrooms
        - **Age (years)**: Age of the property
        - **Location Score**: Quality of location (1-10)
        - **Distance to City**: Distance from city center in miles
        
        **🤖 Model Details:**
        - **Algorithm**: Linear Regression
        - **Dataset**: 500 synthetic house records
        - **Training**: 80% of data
        - **Testing**: 20% of data
        - **Evaluation**: Root Mean Square Error (RMSE) and R² Score
        
        **📈 How It Works:**
        1. Input features are standardized
        2. Model applies learned coefficients
        3. Prediction is made based on feature relationships
        4. Results include price estimate and feature impacts
        
        **⚠️ Note**: This is a demonstration model using synthetic data. 
        For real-world applications, use actual market data and more sophisticated models.
        """)
        
        # Model performance metrics
        if 'df' in locals():
            X = df.drop('price', axis=1)
            y = df['price']
            X_scaled = scaler.transform(X)
            y_pred = model.predict(X_scaled)
            
            from sklearn.metrics import mean_squared_error, r2_score
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            r2 = r2_score(y, y_pred)
            
            st.subheader("Model Performance")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("RMSE (Error)", f"${rmse:,.2f}")
            with col2:
                st.metric("R² Score (Accuracy)", f"{r2:.4f}")
        
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **💡 Tips:**
        - Larger houses = Higher prices
        - Better locations = Higher prices
        - Older houses = Lower prices
        - Closer to city = Higher prices
        """
    )

else:
    st.error("""
    ## Setup Required!
    
    1. First run the model creation script:
    ```bash
    python house_price_model.py
    ```
    
    2. This will generate:
       - `house_data.csv` - Dataset
       - `house_price_model.pkl` - Trained model
       - `scaler.pkl` - Feature scaler
    
    3. Then restart this app
    """)
import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def train_model(input_file, model_file):
    print("🤖 Starting model training...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Please run dataset_prepare.py first!")
        return

    # Load updated dataset
    print(f"📥 Loading updated dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Define features and target
    X = df[['Acetone', 'Temperature', 'Humidity']]
    y = df['Glucose']
    
    # Train-test split (80/20)
    print("🔀 Splitting dataset (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # ------------------ Linear Regression Model (Baseline) ------------------
    print("\n📉 Training Linear Regression model (baseline)...")
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    lr_preds = lr_model.predict(X_test)
    lr_mae = mean_absolute_error(y_test, lr_preds)
    print(f"📊 Linear Regression MAE: {lr_mae:.2f}")
    
    from sklearn.ensemble import GradientBoostingRegressor
    # ------------------ Gradient Boosting Regressor (Optimized Model) ------------------
    print("\n🚀 Training Gradient Boosting Regressor (highly optimized)...")
    rf_model = GradientBoostingRegressor(n_estimators=150, learning_rate=0.1, max_depth=4, random_state=42)
    rf_model.fit(X_train, y_train)
    rf_preds = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, rf_preds)
    rf_r2 = r2_score(y_test, rf_preds)
    print(f"📊 Model MAE: {rf_mae:.2f}")
    print(f"📈 Model Accuracy (R² Score): {rf_r2:.4f}")
    
    # ------------------ Save the best model ------------------
    print(f"\n💾 Saving Random Forest model to {model_file}...")
    joblib.dump(rf_model, model_file)
    
    # ------------------ Feature Correlation Visualization ------------------
    print("\n📊 Generating visualizations...")
    
    # Plot feature correlations
    plt.figure(figsize=(10, 6))
    correlation_matrix = df[['Glucose', 'Acetone', 'Temperature', 'Humidity']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='RdYlBu', fmt=".2f")
    plt.title('Feature Correlation Matrix: Glucose prediction features')
    plt.savefig('reports/correlation_matrix.png')
    print("🖼️ Correlation matrix plot saved as reports/correlation_matrix.png")
    
    # Plot Actual vs Predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, rf_preds, alpha=0.6, color='blue', label='Predictions')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Glucose')
    plt.ylabel('Predicted Glucose')
    plt.title('Actual vs Predicted Glucose Levels (Random Forest Regressor)')
    plt.legend()
    plt.savefig('reports/actual_vs_predicted.png')
    print("🖼️ Actual vs Predicted plot saved as reports/actual_vs_predicted.png")
    
    # Show prediction samples
    print("\n📊 Sample Predictions (Actual vs Predicted):")
    samples = pd.DataFrame({'Actual': y_test, 'Predicted': rf_preds.round(2)})
    print(samples.head(10))

if __name__ == "__main__":
    # Ensure folders exist
    for d in ['models', 'reports']:
        if not os.path.exists(d):
            os.makedirs(d)
    train_model('data/updated_dataset.csv', 'models/gluco_model.pkl')

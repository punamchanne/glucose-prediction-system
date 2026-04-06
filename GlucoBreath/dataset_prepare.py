import pandas as pd
import numpy as np
import os

def prepare_dataset(input_file, output_file):
    print("🚀 Starting dataset preparation...")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found!")
        return

    # Load dataset
    print(f"📥 Loading dataset from {input_file}...")
    df = pd.read_csv(input_file)
    
    # Display initial preview
    print("\n📊 Initial Dataset Preview:")
    print(df.head())

    # Add new synthetic columns
    print("\n✨ Adding synthetic IoT-like features (Acetone, Temperature, Humidity)...")
    
    # Acetone (ppm) = (Glucose / 15) * random multiplier
    # This makes input values (ppm) much smaller than output (mg/dL)
    df['Acetone'] = (df['Glucose'] / 15) * np.random.uniform(0.85, 1.15, size=len(df))
    
    # Temperature: Random values between 25°C and 37°C
    df['Temperature'] = np.random.uniform(25, 37, size=len(df))
    
    # Humidity: Random values between 40% and 80%
    df['Humidity'] = np.random.uniform(40, 80, size=len(df))

    # Round values for realism
    df['Acetone'] = df['Acetone'].round(2)
    df['Temperature'] = df['Temperature'].round(1)
    df['Humidity'] = df['Humidity'].round(1)

    # Save updated dataset
    print(f"💾 Saving updated dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    
    print("\n✅ Dataset preparation complete!")
    print("\n📊 Updated Dataset Preview (First 5 rows):")
    print(df[['Glucose', 'Acetone', 'Temperature', 'Humidity']].head())

if __name__ == "__main__":
    # Create data directory if not exists
    if not os.path.exists('data'):
        os.makedirs('data')
    prepare_dataset('data/diabetes.csv', 'data/updated_dataset.csv')

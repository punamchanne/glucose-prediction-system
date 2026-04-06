import joblib
import pandas as pd
import os
import sys

def predict_glucose(acetone, temperature, humidity, model_file='gluco_model.pkl'):
    # Check if model file exists
    if not os.path.exists(model_file):
        print(f"❌ Error: {model_file} not found. Please run model_train.py first!")
        return None

    # Load model
    print(f"📥 Loading trained model: {model_file}")
    model = joblib.load(model_file)
    
    # Prepare input data
    input_data = pd.DataFrame([[acetone, temperature, humidity]], 
                              columns=['Acetone', 'Temperature', 'Humidity'])
    
    # Predict glucose
    prediction = model.predict(input_data)[0]
    
    return prediction

if __name__ == "__main__":
    if len(sys.argv) == 4:
        acetone = float(sys.argv[1])
        temperature = float(sys.argv[2])
        humidity = float(sys.argv[3])
    else:
        print("💡 Tip: You can pass input as command line arguments: python predict.py <acetone> <temperature> <humidity>")
        print("📝 Enter input values manually:")
        try:
            acetone = float(input("🔹 Enter Acetone level (e.g. 150): "))
            temperature = float(input("🌡️ Enter Temperature (°C - range 25-37): "))
            humidity = float(input("💧 Enter Humidity (% - range 40-80): "))
        except ValueError:
            print("❌ Invalid input! Please enter numbers only.")
            sys.exit(1)

    print(f"\n🔍 Predicting glucose for: Acetone={acetone}, Temp={temperature}°C, Humidity={humidity}%")
    glucose = predict_glucose(acetone, temperature, humidity)
    
    if glucose is not None:
        print(f"📈 Predicted Blood Glucose Level: {glucose:.2f} mg/dL")
        print("\n✅ Prediction complete!")

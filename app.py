from flask import Flask, request, render_template, jsonify
import xgboost as xgb
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# --- Load mô hình Machine Learning ---
GD_model, encoder = joblib.load("GDmodel_enc.rpk")

# --- Mapping cho dữ liệu dạng chữ ---
gender_map = {"Male": 0, "Female": 1}
academic_map = {"Undergraduate": 0, "Graduated": 1, "Highschool": 2}
relationship_map = {"Single": 0, "In a relationship": 1, "Complicated": 3}
platform_map = {"Youtube": 0, "Facebook": 1, "TikTok": 2, "Instagram": 3, "Other": 4}

@app.route('/', method=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_templates('index.html')
    try:
        # --- Lấy dữ liệu từ form ---
        data = {
            "Gender": gender_map[request.form["Gender"]],
            "Academic_Level": academic_map[request.form["Academic_Level"]],
            "Sleep_Hours_Per_Night": float(request.form["Sleep_Hours_Per_Night"]),
            "Relationship_Status": relationship_map[request.form["Relationship_Status"]],
            "Mental_Health_Score": float(request.form["Mental_Health_Score"]),
            "Most_Used_Platform": platform_map[request.form["Most_Used_Platform"]],
            "Avg_Daily_Usage_Hours": float(request.form["Avg_Daily_Usage_Hours"])
        }

        # --- Chuyển dữ liệu thành DataFrame cho mô hình ---
        X = pd.DataFrame([data])

        # --- Dự đoán ---
        X_encoded = encoder.transform(X)
        
        prediction = GD_model.predict(X_encoded)[0]

        # --- Xếp loại mức độ ---
        if prediction < 4:
            level = "Thấp"
        elif prediction < 7:
            level = "Trung bình"
        else:
            level = "Cao"

        return render_template('index.html', result={
            "score": round(prediction, 2),
            "level": level
    except Exception as e: 
    return render_template('iindex.html', error=str(e))
        })

if __name__ == '__main__':
    app.run(debug=False)

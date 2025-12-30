from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model
model = joblib.load("xgboost_pipeline.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    probability = None

    if request.method == "POST":
        # Lấy dữ liệu từ form
        age = int(request.form["age"])
        bmi = float(request.form["bmi"])
        sleep = float(request.form["sleep"])
        smoking = request.form["smoking"]
        exercise = request.form["exercise"]
        alcohol = request.form["alcohol"]
        sugar_intake = request.form["sugar_intake"]


        # Tạo DataFrame (PHẢI ĐÚNG TÊN CỘT)
        df = pd.DataFrame([{
            "age": age,
            "bmi": bmi,
            "sleep": sleep,
            "smoking": smoking,
            "exercise": exercise,
            "alcohol": alcohol,
            "sugar_intake": sugar_intake
            
        }])

        # Dự đoán
        prediction = model.predict(df)[0]
        probability = model.predict_proba(df)[0][1]

    return render_template(
        "index.html",
        prediction=prediction,
        probability=probability
    )

if __name__ == "__main__":
    app.run(debug=True)

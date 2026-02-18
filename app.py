import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, session

app = Flask(__name__)
app.secret_key = "loan_predictor_secret_key"

model = joblib.load("Notebook/Knn_loan_model.pkl")

FEATURE_COLUMNS = [
    "Gender", "Married", "Dependents", "Education", "Self_Employed",
    "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
    "Loan_Amount_Term", "Credit_History", "Property_Area", "Total_Income",
]


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            applicant_income = int(request.form.get("ApplicantIncome"))
            coapplicant_income = float(request.form.get("CoapplicantIncome"))

            form_data = {
                "Gender": request.form.get("Gender"),
                "Married": request.form.get("Married"),
                "Dependents": request.form.get("Dependents"),
                "Education": request.form.get("Education"),
                "Self_Employed": request.form.get("Self_Employed"),
                "ApplicantIncome": applicant_income,
                "CoapplicantIncome": coapplicant_income,
                "LoanAmount": float(request.form.get("LoanAmount")),
                "Loan_Amount_Term": float(request.form.get("Loan_Amount_Term")),
                "Credit_History": float(request.form.get("Credit_History")),
                "Property_Area": request.form.get("Property_Area"),
                "Total_Income": applicant_income + coapplicant_income,
            }

            df = pd.DataFrame([form_data], columns=FEATURE_COLUMNS)
            result = model.predict(df)[0]
            session["prediction"] = "Approved" if result == 1 else "Not Approved"
        except Exception as e:
            session["error"] = f"Prediction failed: {e}"

        return redirect(url_for("index"))

    prediction = session.pop("prediction", None)
    error = session.pop("error", None)
    return render_template("index.html", prediction=prediction, error=error, form_data={})


if __name__ == "__main__":
    app.run(debug=True)

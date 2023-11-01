from flask import Flask, render_template, request
import joblib as jb
import json

app = Flask(__name__)

mdl = jb.load("mdl.pkl.z")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    title = request.form.get("title", "")
    if title:
        prediction = mdl.predict_proba([title])[0][1]
    else:
        prediction = None

    return render_template("result.html", title=title, prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
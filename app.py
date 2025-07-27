from flask import Flask, request, render_template
from predict.predict_url import predict_url

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_result = None
    entered_url = None

    if request.method == "POST":
        entered_url = request.form.get("url")

        if entered_url:
            prediction_result = predict_url(entered_url)

    return render_template("index.html", prediction=prediction_result, url=entered_url)

if __name__ == "__main__":
    app.run(debug=True)

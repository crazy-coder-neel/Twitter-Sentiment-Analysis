import numpy as np
from flask import Flask,request,jsonify,render_template
import pickle

app=Flask(__name__)

model=pickle.load(open("X_sentiment_analysis.pkl","rb"))

@app.route("/")
def Home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():
    data = request.get_json()
    tweet = data['tweet']
    vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    transformed_tweet = vectorizer.transform([tweet])

    prediction = model.predict(transformed_tweet)[0]

    if prediction == 1:
        emoji = '&#x1F60A'
    elif prediction == 0:
        emoji = '&#x1F610'
    elif prediction == -1:
        emoji = '&#x1F622'

    return jsonify({'emoji': emoji})

if __name__ == '__main__':
    app.run(debug=True)
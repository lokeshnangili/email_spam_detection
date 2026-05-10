import pickle
import string
import nltk
from nltk.corpus import stopwords

# Ensure stopwords are downloaded (should be done once)
try:
    stopwords.words('english')
except LookupError:
    nltk.download('stopwords')

# Load the model and vectorizer
with open('spam_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf = pickle.load(vectorizer_file)

def clean_text(text):
    text = text.lower()
    no_punc = [char for char in text if char not in string.punctuation]
    no_punc = ''.join(no_punc)
    return ' '.join(
        word for word in no_punc.split()
        if word not in stopwords.words('english')
    )

def predict_spam(message):
    cleaned_message = clean_text(message)
    message_vector = tfidf.transform([cleaned_message])
    prediction = model.predict(message_vector)
    
    # MultinomialNB outputs 'ham' or 'spam'
    return prediction[0]

if __name__ == '__main__':
    test_message_ham = "Hey, let's catch up later today."
    test_message_spam = "Congratulations! You've won a free iPhone. Click here to claim."

    print(f"Message: '{test_message_ham}' -> Prediction: {predict_spam(test_message_ham)}")
    print(f"Message: '{test_message_spam}' -> Prediction: {predict_spam(test_message_spam)}")

    # Example of how you might use it in a web framework (e.g., Flask)
    # from flask import Flask, request, jsonify
    # app = Flask(__name__)

    # @app.route('/predict', methods=['POST'])
    # def predict():
    #     data = request.get_json(force=True)
    #     message = data['message']
    #     prediction = predict_spam(message)
    #     return jsonify(prediction=prediction)

    # app.run(debug=True)

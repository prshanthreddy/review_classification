import joblib
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import gradio as gr
import nltk
nltk.download('punkt')
nltk.download('stopwords')

count_vectorizer = joblib.load("count_vectorizer.joblib")
best_logistic_model = joblib.load("logistic_regression_model.joblib")
best_knn_model = joblib.load("knn_model.joblib")

knn_test_accuracy =0.635
logistic_test_accuracy = 0.735

def preprocess_text(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word.isalnum()]
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words("english"))
    tokens = [word for word in tokens if word not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    preprocessed_text = " ".join(tokens)
    transformed_text = count_vectorizer.transform([preprocessed_text])
    return transformed_text

def predict_sentiment(text, model_name):
    preprocessed_text = preprocess_text(text)
    if model_name == "Logistic Regression":
        prediction = best_logistic_model.predict(preprocessed_text)[0]
        accuracy = logistic_test_accuracy

    elif model_name == "K-nearest Neighbors":
        prediction = best_knn_model.predict(preprocessed_text)[0]
        accuracy = knn_test_accuracy
    else:
        return "Invalid model selection", None
    if prediction==1: 
      sentiment="Positive"
      pos_accuracy=accuracy
      neg_accuracy=1-pos_accuracy
    else:
      sentiment="Negative"
      neg_accuracy=accuracy
      pos_accuracy=1-neg_accuracy
    accuracy_dict = {"Positive": pos_accuracy, "Negative": neg_accuracy}
    
    return accuracy_dict
examples = [["Point your finger at any item on the menu, order it and you won't be disappointed","K-nearest Neighbors"], ["Similarly, the delivery man did not say a word of apology when our food was 45 minutes late.","Logistic Regression"],["I vomited in the bathroom mid lunch.","K-nearest Neighbors"]]


interface = gr.Interface(
    fn=predict_sentiment,
    inputs=[gr.Textbox(lines=7, label="Input Text"),
            gr.Dropdown(["Logistic Regression", "K-nearest Neighbors"], label="Select Model")],
    outputs=gr.Label(dict),
    title="Sentiment Prediction",
    description="Select a model and enter a text to predict the sentiment (positive or negative) and show the accuracy score.",
    examples=examples
)
interface.launch()

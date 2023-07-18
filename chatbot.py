from flask import Flask, request, jsonify, render_template
import openai
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os
import matplotlib.pyplot as plt

app = Flask(__name__, static_folder='static')

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a caring therapist"}]
user_responses = []

# Load the emotion classification model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def CustomChatGPT(user_input):
    user_responses.append(user_input)
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    reply = CustomChatGPT(user_input)
    return jsonify({"reply": reply})

@app.route("/result", methods=["GET", "POST"])
def result():
    if request.method == "POST":
        text = request.form.get("text")
        user_responses.append(text)
        tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
        tokenizer.fit_on_texts(user_responses)
        sequences = tokenizer.texts_to_sequences(user_responses)
        padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
        tf.config.run_functions_eagerly(True)
        predictions = model.predict(padded_sequences)[-1]  # Only consider the last response
        emotion_probabilities = predictions.tolist()

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(emotion_probabilities, labels=emotion_labels, autopct='%1.1f%%')
        plt.title("Emotion Distribution")
        plt.savefig("static/pie_chart.png")

        return render_template("result.html", emotion_labels=emotion_labels, emotion_probabilities=emotion_probabilities)
    return render_template("result.html")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5005))
    app.run(host="0.0.0.0", port=port, debug=True)





'''
from flask import Flask, request, jsonify, render_template
import openai
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__, static_folder='', static_url_path='')

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a caring therapist"}]

# Load the emotion classification model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    reply = CustomChatGPT(user_input)
    return jsonify({"reply": reply})

@app.route("/result")
def analyze_emotion(text):
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    tf.config.run_functions_eagerly(True)
    predictions = model.predict(padded_sequences)
    return predictions[0]

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5003))
    app.run(host="0.0.0.0", port=port, debug=True)
'''
'''
from flask import Flask, request, jsonify, render_template
import openai
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__, static_folder='', static_url_path='')

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a therapist"}]

# Load the emotion classification model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

def analyze_emotion(text):
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    predictions = model.predict(padded_sequences)
    return predictions[0]

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    reply = CustomChatGPT(user_input)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
'''
'''
from flask import Flask, request, jsonify, render_template
import openai
import tensorflow as tf
import keras
import matplotlib.pyplot as plt
import os

app = Flask(__name__, static_folder='', static_url_path='')

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a therapist"}]

# Load the emotion classification model
model = tf.keras.models.load_model("emotion_model.h5")

# Emotion labels
emotion_labels = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

def analyze_emotion(text):
    tokenizer = Tokenizer(num_words=10000, oov_token='<OOV>')
    tokenizer.fit_on_texts(text)
    sequences = tokenizer.texts_to_sequences(text)
    padded_sequences = pad_sequences(sequences, maxlen=100, padding='post', truncating='post')
    predictions = model.predict(padded_sequences)
    return predictions[0]

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    reply = CustomChatGPT(user_input)
    return jsonify({"reply": reply})

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get("text")
    predictions = analyze_emotion([text])
    emotion_probabilities = predictions.tolist()
    return render_template("chart.html", emotion_labels=emotion_labels, emotion_probabilities=emotion_probabilities)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
'''

'''
#chatbo1.py
from flask import Flask, request, jsonify, render_template
import openai
import os

app = Flask(__name__ ,static_folder='', static_url_path='')

openai.api_key = "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"

messages = [{"role": "system", "content": "You are a therapist"}]

def CustomChatGPT(user_input):
    messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": ChatGPT_reply})
    return ChatGPT_reply

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/api/chat", methods=["POST"])
def chat():
    user_input = request.json.get("user_input")
    reply = CustomChatGPT(user_input)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
'''
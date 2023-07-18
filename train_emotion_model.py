import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Load the data into a DataFrame
data = pd.read_csv('tweet_emotions.csv', header=None, names=['id', 'label', 'text'])

# Preprocess the data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data['text'])
sequences = tokenizer.texts_to_sequences(data['text'])
X = pad_sequences(sequences)
y = data['label']

# Encode the labels
label_to_index = {label: i for i, label in enumerate(set(y))}
index_to_label = {i: label for label, i in label_to_index.items()}
y = y.map(label_to_index)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model architecture
model = keras.Sequential([
    keras.layers.Embedding(len(tokenizer.word_index) + 1, 100, input_length=X.shape[1]),
    keras.layers.Bidirectional(keras.layers.GRU(64)),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(len(label_to_index), activation='softmax')
])

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=32)

model.save("emotion_model.h5")
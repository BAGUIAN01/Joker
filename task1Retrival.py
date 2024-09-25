# %%
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
import joblib
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import requests
# %%
# LOAD JSON FILES

corpus_file_path = "./task1_data/joker_2024_task1_corpus.json"
queries_file_path = "./task1_data/joker_2024_task1_queries_train.json"
qrels_file_path = "./task1_data/joker_2024_task1_qrels_train.json"
queries_test_file_path = "./task1_data/joker_2024_task1_queries_test.json"

with open(corpus_file_path, 'r') as f:
    corpus = json.load(f)

with open(queries_file_path, 'r') as f:
    queries = json.load(f)

with open(qrels_file_path, 'r') as f:
    qrels = json.load(f)

with open(queries_test_file_path, 'r') as f:
    queries_test = json.load(f)


# %%

# JSON TO DATAFRAME
doc_df = pd.DataFrame(corpus)
query_df = pd.DataFrame(queries)
doc_df = doc_df.dropna(subset=['text'])
doc_qrel_df = pd.DataFrame(qrels)
queries_test_df = pd.DataFrame(queries_test)
# queries_test_df = pd.merge(queries_test_df, doc_df, on="qid")
# %%

# GET JOKES
loaded_pipeline = joblib.load('joker_classifier.pkl')
doc_df["label"] = loaded_pipeline.predict(doc_df["text"])
jokes_data = doc_df[doc_df["label"] == 1]
# %%
# FIND SIMILAIRE TEXT BY GIVING A QUERY
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(jokes_data['text'])
query_tfidf = vectorizer.transform(queries_test_df['query'])
cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix)
results = []
for idx, query in enumerate(queries_test_df['qid']):
    sim_scores = list(enumerate(cosine_similarities[idx]))

    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    for rank, (doc_idx, score) in enumerate(sim_scores):
        print(doc_idx)
        result = {
            "run_id": "the_jokester_1_task_1_TFIDF_LogisticRegression",
            "manual": 0,
            "qid": query,
            "docid": jokes_data.iloc[doc_idx]['docid'],
            "rank": rank + 1,
            "score": score  # not sure
        }
        results.append(result)
# %%
results_df = pd.DataFrame(results)
data = pd.merge(results_df, doc_df, on="docid")
# %%
loaded_pipeline = joblib.load('joker_classifier.pkl')
data["label"] = loaded_pipeline.predict(data["text"])
joks_data = data[data["label"] == 1]
# %%

# TRAIN MODEL TO CLASSIIFY IF EXTACTED TEXT IS A JOK OR NOT

merged_data = pd.merge(doc_df, doc_qrel_df, on="docid")
# %%
X = merged_data['text']
Y = merged_data['qrel']
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42)
# %%
pipeline = make_pipeline(
    TfidfVectorizer(),
    LogisticRegression(max_iter=1000)
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))
# %%
test_text = "The musical score to Topsyturveydom does not survive, but amateur productions in recent decades have used newly composed scores or performed the work as a non-musical play."
pipeline.predict([test_text])
# %%
joblib.dump(pipeline, 'joker_classifier.pkl')

# %%
loaded_pipeline = joblib.load('joker_classifier.pkl')
# %%
loaded_pipeline.predict([test_text])
# %%
data["label"] = pipeline.predict(data["text"])
# %%
Counter(data["label"].values)
data[data["label"] == 1]

# %%
# CLASSIFICATION OF JOKES WITH LSTM

lstm_data = pd.merge(doc_df, doc_qrel_df, on="docid")
lstm_data = lstm_data[lstm_data["qrel"] == 1]
lstm_data = pd.merge(lstm_data, query_df, on="qid")

# %%

X = lstm_data['text'].values
y = lstm_data['query'].values
# %%
# Encodage des labels
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
# %%
# Prétraitement du texte
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(X)
X = tokenizer.texts_to_sequences(X)
X = pad_sequences(X, maxlen=100)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
# %%
# Création du modèle
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length=100))
model.add(SpatialDropout1D(0.2))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

# Compilation du modèle
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
# %%
# Entraînement du modèle
history = model.fit(X_train, y_train, epochs=60, batch_size=64,
                    validation_data=(X_test, y_test), verbose=2)

# Évaluation du modèle
score, acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {acc}')
# %%
model.save("classifier_kind_of_jokes.h5")
# %%
# TEST MODEL

test_data = jokes_data.iloc[2]["text"]
# %%
tokenizer.fit_on_texts(test_data)
test_data = tokenizer.texts_to_sequences(test_data)
test_data = pad_sequences(test_data, maxlen=100)

# %%
predictions = model.predict(test_data)
predicted_category = label_encoder.inverse_transform(
    [predictions.argmax(axis=-1)[0]])
# %%
loaded_model = tf.keras.models.load_model('classifier_kind_of_jokes.h5')
# %%


def transform(test_data):
    tokenizer.fit_on_texts(test_data)
    test_data = tokenizer.texts_to_sequences(test_data)
    test_data = pad_sequences(test_data, maxlen=100)

    predictions = model.predict(test_data)
    predicted_category = label_encoder.inverse_transform(
        [predictions.argmax(axis=-1)[0]])
    qrel_text = predicted_category[0]
    return qrel_text


# %%
jokes_data["query_text"] = transform(jokes_data["text"])
# %%

# %%
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from transformers import BertTokenizer, BertForSequenceClassification
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
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import Embedding, SpatialDropout1D, LSTM, Bidirectional, Dense, BatchNormalization
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report,accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.ensemble import VotingClassifier
# %%
#LOAD JSON FILES

corpus_train_file_path = "./Task 2 - classification/joker-2024-task2-classification-train-input.json"
corpus_test_file_path ="./Task 2 - classification/joker-2024-task2-classification-test.json"
qrels_file_path = "./Task 2 - classification/joker-2024-task2-classification-train-qrels.json"


with open(corpus_train_file_path, 'r') as f:
    corpus_train = json.load(f)

with open(corpus_test_file_path, 'r') as f:
    corpus_test = json.load(f)

with open(qrels_file_path, 'r') as f:
    qrels = json.load(f)

# %%
# JSON TO DATAFRAME
doc_train_df = pd.DataFrame(corpus_train)
doc_test_df = pd.DataFrame(corpus_test)
doc_qrel_df = pd.DataFrame(qrels)


# %%
merged_data = pd.merge(doc_train_df, doc_qrel_df, on="id")
# %%
# %%
X = merged_data['text'].values
Y = merged_data['class'].values

#%%
X_train, X_test, y_train, y_test = train_test_split(merged_data['text'], merged_data['class'], test_size=0.2, random_state=42)

# %%
# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
# %%
#SVM Model

svm_model = SVC(C=1000, gamma=0.001, kernel='rbf', probability=True)
tree_model = DecisionTreeClassifier(criterion='gini',max_depth= 30,
                                    min_samples_leaf= 1,
                                    min_samples_split=5,
                                    splitter='random')
# bosting_model = GradientBoostingClassifier(learning_rate=0.05, max_depth=5, n_estimators=400)
#%%
svm_pipeline = make_pipeline(
    TfidfVectorizer(),
    svm_model
)

svm_pipeline.fit(X_train, y_train)
y_pred = svm_pipeline.predict(X_test)
print(classification_report(y_test, y_pred, target_names=['IR', 'SC', 'EX', 'AID', 'SD', 'WS']))
joblib.dump(svm_pipeline, './models/svm_model.joblib')
#%%
# param_grid = {
#     'C': [0.1, 1, 10, 100,101, 1000],
#     'gamma': np.arange(0.001,1,0.1),
#     'kernel': ['rbf','linear', "poly"]
# }

# grid_search = GridSearchCV(svm_model, param_grid, refit=True, verbose=3)
# grid_search.fit(X_train_tfidf, y_train)
# best_params = grid_search.best_params_
# best_svm_model = grid_search.best_estimator_
# y_pred = best_svm_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=['IR', 'SC', 'EX', 'AID', 'SD', 'WS']))
#%%
# SAVE MODEL
joblib.dump(best_svm_model, './models/svm_model.joblib')
#%%
pipelines = {
    'Gradient Boosting': make_pipeline(TfidfVectorizer(), GradientBoostingClassifier()),
    'Random Forest': make_pipeline(TfidfVectorizer(), RandomForestClassifier()),
    'Naive Bayes': make_pipeline(TfidfVectorizer(), MultinomialNB())
}

param_grids = {
    'Gradient Boosting': {
        'gradientboostingclassifier__n_estimators': [100, 200],
        'gradientboostingclassifier__learning_rate': [0.05, 0.1, 0.2],
        'gradientboostingclassifier__max_depth': [3, 5, 7]
    },
    'Random Forest': {
        'randomforestclassifier__n_estimators': [100, 200],
        'randomforestclassifier__max_features': ['auto', 'sqrt', 'log2'],
        'randomforestclassifier__max_depth': [10, 20, 30],
        'randomforestclassifier__criterion': ['gini', 'entropy']
    },
    'Naive Bayes': {
        'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],
        'tfidfvectorizer__use_idf': [True, False],
        'multinomialnb__alpha': [0.1, 1.0, 10.0]
    }
}

# Initialiser et exécuter GridSearchCV pour chaque modèle
for name, pipeline in pipelines.items():
    grid_search = GridSearchCV(pipeline, param_grids[name], cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    print(f"Results for {name}:")
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score: ", grid_search.best_score_)
    print("-" * 30)
#%%
# TREE
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(tree_model, param_grid, cv=5, refit=True, verbose=3)
grid_search.fit(X_train_tfidf, y_train)
best_params = grid_search.best_params_
best_dt_model = grid_search.best_estimator_
y_pred = best_dt_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=['IR', 'SC', 'EX', 'AID', 'SD', 'WS']))
#%%
# SAVE tree MODEL
joblib.dump(best_dt_model, './models/tree_model.joblib')

#%%
# Bosting
param_grid = {
    'n_estimators': [100, 200, 300 , 400, 500],
    'learning_rate': [0.05, 0.1, 0.2, 0.3, 0.4, 0.5],
    'max_depth': [3, 5, 7, 10, 15, 20],
}
# learning_rate=0.05, max_depth=5, n_estimators=400;,
grid_search = GridSearchCV(bosting_model, param_grid, cv=5, refit=True, verbose=3)
grid_search.fit(X_train_tfidf, y_train)
best_params = grid_search.best_params_
best_bt_model = grid_search.best_estimator_
y_pred = best_bt_model.predict(X_test_tfidf)
print(classification_report(y_test, y_pred, target_names=['IR', 'SC', 'EX', 'AID', 'SD', 'WS']))
#%%
# SAVE tree MODEL
joblib.dump(best_bt_model, './models/bosting_model.joblib')
#%%
# Stacking
svm_model = joblib.load('./models/svm_model.joblib')
base_models = [
    ('tree', DecisionTreeClassifier(criterion='gini',max_depth= 30,
                                    min_samples_leaf= 1,
                                    min_samples_split=5,
                                    splitter='random')),
    ('rf', RandomForestClassifier(criterion='gini',max_depth= 30,
                                  max_features= 'sqrt',
                                  n_estimators= 100)),
    ('gb', GradientBoostingClassifier(learning_rate=0.05, max_depth=5, n_estimators=400))
]

meta_model = LogisticRegression(multi_class='multinomial', solver='lbfgs')


pipeline = make_pipeline(
    TfidfVectorizer(),
    StackingClassifier(estimators=base_models, final_estimator=meta_model, cv=5)
)

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['IR', 'SC', 'EX', 'AID', 'SD', 'WS']))
joblib.dump(pipeline, './models/stacking_rf_gb_nb.joblib')
# %%

#%%
staking_model = joblib.load('./models/stacking_rf_gb_nb.joblib')
y_pred = staking_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=staking_model.classes_)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title('Matrice de Confusion')
plt.show()
#%%
svm_model = joblib.load('./models/svm_model.joblib')
# vectorizer = TfidfVectorizer(max_features=5000)
# X_test_tfidf = vectorizer.transform(X_test)
y_pred = svm_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_model.classes_)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
plt.title('Matrice de Confusion')
plt.show()
#%%
# Voting Classifier
ensemble_model = VotingClassifier(estimators=[('stacking1', staking_model), ('svm1', svm_model)], voting='soft',weights=[2, 1])
ensemble_model.fit(X_train, y_train)
#%%

y_pred = ensemble_model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=ensemble_model.classes_)
plt.figure(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=plt.gca())
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=['IR', 'SC', 'EX', 'AID', 'SD', 'WS']))
joblib.dump(ensemble_model, './models/voting_st1_st2_svm.joblib')
#%%
def predict(text):
    # Convert text to TF-IDF features
    text_tfidf = vectorizer.transform([text])
    # Predict the class
    prediction = model.predict(text_tfidf)
    return prediction[0]

# Example usage
text_sample = doc_test_df.iloc[2]["text"]
prediction = predict(text_sample)
print(f'Predicted class: {prediction}')

# %%
# TEST du model
voting_model = joblib.load('./models/voting_st1_st2_svm.joblib')


# y_pred = voting_model.predict([doc_test_df["text"][0]])

results = []
for index in range(doc_test_df.shape[0]):
    y_pred = voting_model.predict([doc_test_df["text"][i]])
    result = {
            "run_id": "the_jokester_1_task_2_Stacking_Voting",
            "manual": 0,
            "id": str(doc_test_df.iloc[i]['id']),
            "class": y_pred[0],
        }
    results.append(result)


# %%
json.dump(results, "./the_jokester_1_task_2_Stacking_Voting.json")
# %%
with open("./the_jokester_1_task_2_Stacking_Voting.json", "w") as json_file:
    json.dump(results, json_file)
# %%

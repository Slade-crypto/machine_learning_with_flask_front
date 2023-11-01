import pandas as pd
import numpy as np
import matplotlib  as matlib
import joblib as jb
import requests as r
import urllib.parse
import json

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

p = pd.read_csv("clickbait_titles.csv", index_col=0, parse_dates=['timestamp'])
n = pd.read_csv("non_clickbait_titles.csv", index_col=0, parse_dates=['timestamp'])
p['y'] = 1
n['y'] = 0

data = pd.concat([p,n], axis=0, ignore_index=True).sort_values("timestamp")

matlib.use('TkAgg')

data = data[data['timestamp'] >= "2017-06-01"]
data.groupby(data['timestamp'].dt.date)['y'].mean().plot(figsize=(10,5))

data['title_proc'] = data['title'].str.lower().str.replace(r'[^\w\s]+', '')

data.shape

Xtrain, ytrain = data['title_proc'].iloc[:data.shape[0] // 2], data['y'].iloc[:data.shape[0] // 2]
Xtest, ytest = data['title_proc'].iloc[data.shape[0] // 2:], data['y'].iloc[data.shape[0] // 2:]

base = np.ones(Xtest.shape[0]) * ytrain.mean()
log_loss(ytest, base)

mdl = make_pipeline(TfidfVectorizer(min_df=1, ngram_range=(1,1)), RandomForestClassifier(n_estimators=1000, n_jobs=6, random_state=0))

mdl.fit(Xtrain, ytrain)
p = mdl.predict_proba(Xtest)[:, 1]
log_loss(ytest, p)

mdl = make_pipeline(TfidfVectorizer(min_df=2, ngram_range=(1,1)), LogisticRegression(C=20.))

mdl.fit(Xtrain, ytrain)
p = mdl.predict_proba(Xtest)[:, 1]
log_loss(ytest, p)

jb.dump(mdl, "mdl.pkl.z")

encoded = urllib.parse.quote("10 coisas fofas e super baratas para alegrar a vida")
# encoded = urllib.parse.quote("Você não vai acreditar nestes segredos devastadores sobre enriquecer como os famosos")
# encoded = urllib.parse.quote("Homem é preso por descobrir segredo que ensina a falar inglês fluente em uma hora")
res = r.get("http://localhost:8000/?titulo={}".format(encoded))
res.text
json.loads(res.text)
# data.head()
# matlib.pyplot.show()
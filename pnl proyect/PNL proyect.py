import pandas as pd
import re
import json
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.stem.lancaster import LancasterStemmer 
from conect import connection_sqlite3

def PNL(data):
    lts = LancasterStemmer()
    data = [x.split() for x in data]
    data = [[lts.stem(i.lower()) for i in x]for x in data]
    data = [' '.join(x) for x in data] 
    return data

def filtro(df):  
    df = df.dropna(subset=['text'])
    df = df.reset_index()

    
    data = df.copy() 

    data = preparacion_datos(data)
    
    with open("feelings.json",encoding='utf-8') as file:
        fe = json.load(file)

    data2 = pd.DataFrame(columns=df.columns)

    positivos = PNL(fe['feelings'][0]['Positive'])
    negativos = PNL(fe['feelings'][1]['Negative'])

    data['text'] = PNL(data['text'])

    for i,x in enumerate(data['text']):

        if data['airline_sentiment_confidence'][i] > 0.75:
            data2 = pd.concat([data2, data.loc[i].to_frame().T], ignore_index=True)           

        else:
            pos = r'\b(' + '|'.join(positivos) + r')\b'
            neg = r'\b(' + '|'.join(negativos) + r')\b'

            if len(re.findall(pos, x.lower())) > len(re.findall(neg, x.lower())) and data['airline_sentiment'][i]=='positive':
                data2 = pd.concat([data2, data.loc[i].to_frame().T], ignore_index=True)

            elif len(re.findall(pos, x.lower())) < len(re.findall(neg, x.lower())) and data['airline_sentiment'][i]=='negative':
                data2 = pd.concat([data2, data.loc[i].to_frame().T], ignore_index=True)

            elif len(re.findall(pos, x.lower())) == len(re.findall(neg, x.lower())) and data['airline_sentiment'][i]=='neutral':
                data2 = pd.concat([data2, data.loc[i].to_frame().T], ignore_index=True)

    return data2.drop(columns=['index','level_0'])

def preparacion_datos(data):
    data['text'] = data['text'].apply(lambda x: x.lower()) 
    data['text'] = data['text'].str.replace('[^\w\s]','') 
    data = data.dropna(subset=['text'])
    data = data.reset_index()
    return data


def precision(data,val_output):
   
    data = preparacion_datos(data)

    val_features = vectorizer.transform(data['text'])
    accuracy = clf.score(val_features, val_output)
    return print("Exactitud del modelo:", accuracy)

base, col_names = connection_sqlite3()
df = pd.DataFrame(base,columns=col_names)

df = filtro(df)

train_data = df.sample(frac=0.75, random_state=40)
val_data = df.drop(train_data.index)


train_output =train_data['airline_sentiment']
val_output = val_data['airline_sentiment']

vectorizer = CountVectorizer()
vectorizer.fit(train_data['text'])
train_features = vectorizer.transform(train_data['text'])

clf = LogisticRegression(penalty='elasticnet', solver='saga', l1_ratio=0.5, C=0.6,
            random_state=42, tol=0.0001,max_iter=500)


clf.fit(train_features, train_output)
precision(val_data,val_output)

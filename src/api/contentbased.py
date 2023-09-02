from src.api.preprocessing import load_data
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from fastapi import FastAPI, Request

import pandas as pd
import joblib

app = FastAPI()

df = load_data()

df = df.drop_duplicates(subset=['Place_Name'])

def load_vectorizer():
    try:
        vectorizer = joblib.load('models/vectorizer.pkl')
        return vectorizer
    except Exception as e:
        response = {
            "status": 204,
            "message": str(e)
        }
        return response

def find_similarity(vectorizer, df):
    try:
        tfidf_matrix = vectorizer.transform(df['City'] + ' ' + df['Category'])
        cosine_sim = cosine_similarity(tfidf_matrix)
        cosine_sim_df = pd.DataFrame(cosine_sim, index=df['Place_Name'], columns=df['Place_Name'])
        return cosine_sim_df
    except Exception as e:
        response = {
            "status": 204,
            "message": str(e)
        }
        return response

vectorizer = load_vectorizer()

cosine_sim_df = find_similarity(vectorizer, df)

@app.post("/contentbased")
async def place_recommendations(data: dict = None):
    if data is None:
        response = {
            "status": 400,
            "message": "Invalid request data"
        }
        return response

    try:
        place_name = data.get('place_name')
        total = data.get('total', 10)

        k = int(total)

        user_place_sim = cosine_sim_df[place_name]
        closest_indices = user_place_sim.to_numpy().argsort()[::-1][1:k+1]
        recommendations = df.iloc[closest_indices][['Place_Name', 'Description', 'Category', 'City', 'Price']].to_dict(orient='records')
        response = {
            "status": 200,
            "user_input": place_name,
            "recommendations": recommendations
        }
    except Exception as e:
        response = {
            "status": 500,  # Internal Server Error
            "message": str(e)
        }
    return response


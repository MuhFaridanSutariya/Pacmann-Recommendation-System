from fastapi import FastAPI, Request
from src.api.architectures import RecommenderNet
from src.api.preprocessing import load_data, get_place_encodings
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import os
import joblib
import numpy as np
import pandas as pd

app = FastAPI()

User_Id = 10000
Place_Id = 10000
Age = 10000
Location_Encoded = 10000
embedding_size = 100


def load_model():
    try:
        model = RecommenderNet(User_Id, Place_Id, Age, Location_Encoded, embedding_size)
        dummy_input = tf.constant([[0, 0, 0, 0]], dtype=tf.int32)
        _ = model(dummy_input)
        model.load_weights('models/recommender_model_weights.h5')
        return model  # Return the loaded model, not a response dictionary
    except Exception as e:
        response = {
            "status": 204,
            "message": str(e)
        }
        return response

def load_encoder():
    try:
        ohe_location = joblib.load('models/encoder.pkl')
        return ohe_location
    except Exception as e:
        response = {
            "status": 204,
            "message": str(e)
        }
        return response

def get_top_recommendations(model, user_id, age_encoded, location_encoded, place_to_place_encoded, place_encoded_to_place, user_to_user_encoded, place_df, num_recommendations=10):
    place_visited_by_user = place_df[place_df['User_Id'] == user_id]['Place_Id']
    place_not_visited = place_df[~place_df['Place_Id'].isin(place_visited_by_user)]['Place_Id']

    # Convert to a list of not visited place IDs
    place_not_visited = list(
        set(place_not_visited)
        .intersection(set(place_to_place_encoded.keys()))
    )

    # Encode the not visited places and user
    place_not_visited = [[place_to_place_encoded.get(x)] for x in place_not_visited]
    user_encoder = user_to_user_encoded.get(user_id)
    user_place_array = np.hstack(
        ([[user_encoder, age_encoded, location_encoded]] * len(place_not_visited), place_not_visited)
    )

    ratings = model.predict(user_place_array).flatten()

    top_ratings_indices = ratings.argsort()[-num_recommendations:][::-1]
    recommended_place_ids = [
        place_encoded_to_place.get(place_not_visited[x][0]) for x in top_ratings_indices
    ]

    top_place_user = (
        place_df[place_df['Place_Id'].isin(place_visited_by_user)]
        .sort_values(by='Place_Ratings', ascending=False)
        .head(5)
        .Place_Id.values
    )

    top_places = place_df[place_df['Place_Id'].isin(top_place_user)].drop_duplicates(subset=['Place_Id'])
    top_recommendations = place_df[place_df['Place_Id'].isin(recommended_place_ids)].drop_duplicates(subset=['Place_Id'])

    user_recommendations = {
        "top_places_user": [
            f"{row.Place_Name} : {row.City}" for row in top_places.itertuples()
        ],
        "recommended_places": [
            f"{row.Place_Name} : {row.City}" for row in top_recommendations.itertuples()
        ]
    }

    return user_recommendations


@app.post("/collaborative")
async def recommendation(data: Request):
    data = await data.json()
    user = data['user_id']

    user = int(user)

    model = load_model()
    encoder = load_encoder()
    df = load_data()
    place_to_place_encoded, place_encoded_to_place, user_to_user_encoded = get_place_encodings(df)

    place_df = df.copy()

    user_data = df[df['User_Id'] == user]
    if not user_data.empty:
        age = int(user_data.iloc[0]['Age'])
        Location = user_data.iloc[0]['Location']

        location_encoded = encoder.transform([Location])

        location_encoded = location_encoded[0]
    else:
        response = {
            "status": 204,
            "message": "User not found"
        }
        return response

    try:
        recommendations = get_top_recommendations(model, user, age, location_encoded, place_to_place_encoded, place_encoded_to_place, user_to_user_encoded, place_df)
        
        response = {
            "status": 200,
            "input": [user, age, Location],
            "recommendations": recommendations
        }
    except Exception as e:
        response = {
            "status": 204,
            "message": str(e)
        }
    return response
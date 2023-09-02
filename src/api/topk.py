from src.api.preprocessing import load_data
from fastapi import FastAPI, Request

import pandas as pd

app = FastAPI()

@app.post("/topk")
async def topk_recommendations(data: dict = None):
    if data is None:
        response = {
            "status": 400,
            "message": "Invalid request data"
        }
        return response

    try:
        k = data.get('k', 10)  # Default 10

        k = int(k)

        df = load_data()
        df = df.drop_duplicates(subset=['Place_Id'])

        top_k = df[['Place_Name', 'Description', 'Category', 'City', 'Price', 'Place_Ratings']]

        top_k = top_k.sort_values('Place_Ratings', ascending=False)
        top_k = top_k.head(k)

        recommendations = top_k.to_dict(orient='records')

        response = {
            "status": 200,
            "message": "Top K recommendations retrieved successfully",
            "data": recommendations
        }
    except Exception as e:
        response = {
            "status": 500,  # Internal Server Error
            "message": str(e)
        }
    return response



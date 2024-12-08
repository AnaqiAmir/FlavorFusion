from fastapi import FastAPI
import pandas as pd
from recommendation_model import get_recommendations

app = FastAPI()

@app.get('/recommend')
def recommend(recipe_name: str):
    recommendations = get_recommendations(recipe_name)
    return recommendations.to_dict(orient='records')
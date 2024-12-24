from fastapi import FastAPI
from pydantic import BaseModel
import spacy
import re
import pandas as pd
import uvicorn
import numpy as np
import tensorflow as tf
from typing import Optional
from fastapi.responses import JSONResponse
from fastapi.exceptions import HTTPException




nlp = spacy.load("en_core_web_sm")

# Load the tours data
try:
    tours_df = pd.read_csv("world_tours.csv")
    tours_df = tours_df.fillna("unknown").replace([np.inf, -np.inf], "unknown")
    tours_df["price"] = pd.to_numeric(tours_df["price"], errors="coerce").fillna(0)
    tours_df["average_rating"] = pd.to_numeric(tours_df["average_rating"], errors="coerce").fillna(0)

    print(tours_df.head(5))
except FileNotFoundError:
    raise Exception("CSV file not found. Ensure 'clean_vietnam_tours.csv' is in the working directory.")

# Initialize FastAPI
app = FastAPI()

class UserInput(BaseModel):
    text: Optional[str] = None 
    user_id: int = None
    num_tours: int = 5
    
try:
    ncf_model = tf.keras.models.load_model("ncf_model.keras")  # Update with your model's path
    print("NCF Model loaded successfully.")
except Exception as e:
    raise Exception(f"Error loading NCF model: {e}")

@app.post("/extract_entities")
def extract_entities(text: str):
    print("prompt", text)
    doc = nlp(text)
    entities = {"GPE": [], "DATE": [], "CARDINAL": []}
    for ent in doc.ents:
        if ent.label_ in entities:
            entities[ent.label_].append(ent.text)
    return entities

def extract_price_range(cardinal):
    match = re.search(r'(\d+)\s*to\s*(\d+)', cardinal)
    if match: 
        min_price = int(match.group(1))
        max_price = int(match.group(2))
        return min_price, max_price
    return None, None
    

def recommend_tours(user_input_text: str, tours_df: pd.DataFrame):
    entities = extract_entities(user_input_text)
    print("Extracted Entities", entities)
    recommendations = tours_df.copy()

    if entities['GPE']:
        locations = '|'.join(map(re.escape, entities['GPE']))
        print(locations)
        recommendations = recommendations[
            recommendations['description'].str.contains(locations, case=False, na=False)
        ]

    if entities['DATE']:
        date_parts = re.findall(r'\d+', entities['DATE'][0])
        if date_parts:
            duration_regrex = '|'.join(map(re.escape, date_parts))
            recommendations = recommendations[
                recommendations["duration"].str.contains(duration_regrex, case=False, na=False)
            ]

    if entities['CARDINAL']:
        try:
            min_price, max_price = extract_price_range(entities["CARDINAL"][0])
            print("price",min_price, max_price)
            if min_price and max_price: 
                recommendations["price"] = pd.to_numeric(recommendations["price"], errors='coerce')
                
                recommendations = recommendations[
                    (recommendations["price"] >= min_price) &
                    (recommendations["price"] <= max_price)
                ]
        except ValueError:
            pass
    print(recommendations)
    return recommendations.to_dict(orient='records')


@app.post("/recommend")
async def recommend(user_input: UserInput):
    try :
        recommendations = recommend_tours(user_input.text, tours_df)
        if not recommendations:
            return {"message": "No recommendations found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

    return recommendations

@app.post("/similarTours")
async def similarTours(user_input: UserInput):
    try :
        
        recommendations = tours_df.copy()
        
        user_id = user_input.user_id
        num_tours = user_input.num_tours

        # Prepare input data
        user_array = np.full(num_tours, user_id)
        tour_array = np.arange(num_tours)

        # Make predictions
        predictions = ncf_model.predict([user_array, tour_array])
        top_indices = np.argsort(-predictions.flatten())[:5]

        filtered_recommendations = recommendations.iloc[top_indices]

        # Convert to a list of dictionaries for JSON serialization
        recommended_tours = filtered_recommendations.to_dict(orient="records")

        return JSONResponse(content=recommended_tours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
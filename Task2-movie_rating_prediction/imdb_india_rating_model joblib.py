import joblib
import pandas as pd

# Path where you downloaded the file
model_path = r"C:\Users\HP\Downloads\imdb_india_rating_model.joblib"

# 1. Load the model
model = joblib.load(model_path)

# 2. Build a sample input row with the same columns as X during training
sample = pd.DataFrame([{
    "year": 2024,
    "duration": 130,
    "votes": 5000,
    "director_mean_rating": 7.5,
    "director_movie_count": 10,
    "actor1_mean_rating": 7.2,
    "actor1_movie_count": 20,
    "genre_Action": 1,
    "genre_Drama": 1,
    # set all other genre_* your model expects to 0
    "genre_Comedy": 0,
    "genre_Thriller": 0,
    # ...
    "director_reduced": "Other",
    "actor1_reduced": "Other",
}])

# 3. Predict rating
pred = model.predict(sample)[0]
print("Predicted rating:", round(pred, 2))
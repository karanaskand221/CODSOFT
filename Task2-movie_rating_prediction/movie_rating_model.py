import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib

RANDOM_STATE = 42

# 1. LOAD DATA  ---------------------------------------------------
df = pd.read_csv("IMDb Movies India.csv", encoding="latin1")  # or encoding="cp1252"
print("Raw shape:", df.shape)
print(df.head())

# 2. CLEANING  ----------------------------------------------------

# Standardize column names
df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
print(df.columns)

required = ["name", "year", "duration", "genre",
            "rating", "votes", "director",
            "actor_1", "actor_2", "actor_3"]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing expected columns: {missing}")

# Drop rows with missing rating
df = df.dropna(subset=["rating"])

# --- CLEAN year, duration, votes first ---
df["year"] = df["year"].astype(str).str.extract(r"(\d{4})")[0]
df["year"] = pd.to_numeric(df["year"], errors="coerce")

df["duration"] = df["duration"].astype(str).str.extract(r"(\d+)")[0]
df["duration"] = pd.to_numeric(df["duration"], errors="coerce")

df["votes"] = df["votes"].astype(str).str.replace(",", "", regex=False)
df["votes"] = pd.to_numeric(df["votes"], errors="coerce")

df = df.dropna(subset=["year", "duration", "votes"])

# Filter to realistic values
df = df[(df["year"] >= 1940) & (df["year"] <= 2030)]
df = df[df["duration"].between(40, 240)]
print("After cleaning year/duration/rating:", df.shape)

# Fill simple missing values for categorical columns
for col in ["genre", "director", "actor_1", "actor_2", "actor_3"]:
    df[col] = df[col].fillna("Unknown")

print("After cleaning:", df.shape)


# 3. FEATURE ENGINEERING  -----------------------------------------

# 3.1 expand multi-genre into dummy columns
def expand_genres(frame, col="genre", prefix="genre_"):
    genres_series = frame[col].fillna("").apply(
        lambda x: [g.strip() for g in str(x).split(",") if g.strip()]
    )
    all_genres = sorted({g for sub in genres_series for g in sub})
    genre_cols = []
    for g in all_genres:
        new_col = f"{prefix}{g}"
        frame[new_col] = genres_series.apply(lambda arr, g=g: int(g in arr))
        genre_cols.append(new_col)
    return frame, genre_cols

df, genre_cols = expand_genres(df)

# 3.2 director and main‑actor stats (mean rating & count)
director_stats = (
    df.groupby("director")["rating"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "director_mean_rating",
                       "count": "director_movie_count"})
)
df = df.merge(director_stats, left_on="director", right_index=True, how="left")

actor1_stats = (
    df.groupby("actor_1")["rating"]
      .agg(["mean", "count"])
      .rename(columns={"mean": "actor1_mean_rating",
                       "count": "actor1_movie_count"})
)
df = df.merge(actor1_stats, left_on="actor_1", right_index=True, how="left")

for col in ["director_mean_rating", "director_movie_count",
            "actor1_mean_rating", "actor1_movie_count"]:
    df[col] = df[col].fillna(df[col].median())

# 4. TRAIN-TEST SPLIT  --------------------------------------------

target = "rating"

numeric_features = [
    "year", "duration", "votes",
    "director_mean_rating", "director_movie_count",
    "actor1_mean_rating", "actor1_movie_count"
] + genre_cols

# reduce cardinality of director / actor for one‑hot
TOP_N = 30
top_directors = df["director"].value_counts().nlargest(TOP_N).index
top_actor1   = df["actor_1"].value_counts().nlargest(TOP_N).index

df["director_reduced"] = df["director"].where(df["director"].isin(top_directors), "Other")
df["actor1_reduced"]   = df["actor_1"].where(df["actor_1"].isin(top_actor1), "Other")

categorical_features = ["director_reduced", "actor1_reduced"]

X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
print("Train:", X_train.shape, "Test:", X_test.shape)

# 4.1 preprocessing
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# 5. MODEL TRAINING  ----------------------------------------------

def eval_pipe(name, model):
    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("model", model)
    ])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mse  = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y_test, preds)
    r2   = r2_score(y_test, preds)

    print(f"\n{name}")
    print("-" * len(name))
    print(f"RMSE: {rmse:.3f}")
    print(f"MAE : {mae:.3f}")
    print(f"R²  : {r2:.3f}")
    return pipe


# 5.1 Linear Regression (baseline)
lr_pipe = eval_pipe("Linear Regression", LinearRegression())

# 5.2 Ridge Regression with simple tuning
ridge = Ridge(random_state=RANDOM_STATE)
ridge_params = {"model__alpha": [0.1, 1.0, 10.0, 50.0]}

ridge_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", ridge)
])

ridge_grid = GridSearchCV(
    ridge_pipe,
    param_grid=ridge_params,
    scoring="neg_root_mean_squared_error",
    cv=3,
    n_jobs=-1
)
ridge_grid.fit(X_train, y_train)
print("\nBest Ridge params:", ridge_grid.best_params_)

ridge_best = ridge_grid.best_estimator_
ridge_preds = ridge_best.predict(X_test)

ridge_mse  = mean_squared_error(y_test, ridge_preds)  # no 'squared' arg
ridge_rmse = np.sqrt(ridge_mse)
ridge_mae  = mean_absolute_error(y_test, ridge_preds)
ridge_r2   = r2_score(y_test, ridge_preds)

print(f"Ridge (best) – RMSE: {ridge_rmse:.3f}, MAE: {ridge_mae:.3f}, R²: {ridge_r2:.3f}")


# 5.3 Random Forest (final model as in CODSOFT project)
rf_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=None,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
rf_pipe = eval_pipe("Random Forest", rf_model)

# 5.4 Gradient Boosting (optional)
gbr_model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=3,
    random_state=RANDOM_STATE
)
gbr_pipe = eval_pipe("Gradient Boosting", gbr_model)

# 6. FEATURE IMPORTANCE  ------------------------------------------

rf = rf_pipe.named_steps["model"]
ohe = rf_pipe.named_steps["preprocess"].named_transformers_["cat"]["onehot"]

num_features = numeric_features
cat_features = list(ohe.get_feature_names_out(categorical_features))
all_features = num_features + cat_features

importances = rf.feature_importances_
fi = pd.DataFrame({"feature": all_features,
                   "importance": importances}) \
       .sort_values("importance", ascending=False)

print("\nTop 20 important features:")
print(fi.head(20))

# 7. SAVE FINAL MODEL  --------------------------------------------

final_model = rf_pipe   # choose best performing pipeline
joblib.dump(final_model, "imdb_india_rating_model.joblib")
print("Model saved as imdb_india_rating_model.joblib")

# 8. SIMPLE TEST PREDICTION WHEN RUN FROM TERMINAL  ----------------
if __name__ == "__main__":
    # use one sample from test set
    sample = X_test.iloc[[0]]
    true_rating = y_test.iloc[0]
    pred_rating = final_model.predict(sample)[0]

    print("\nSample true rating :", true_rating)
    print("Sample pred rating :", round(pred_rating, 2))
    print("Sample features    :")
    print(sample.to_dict(orient="records")[0])      
    
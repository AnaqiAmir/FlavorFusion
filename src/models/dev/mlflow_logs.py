import os
import time

import dagshub
import mlflow
import pandas as pd

import faiss_rec

# Initialize dagshub mlflow
# dagshub.init(repo_owner="anaqi.amirrazif", repo_name="FlavorFusion", mlflow=True)

# # Set up mlflow credentials
# os.environ["MLFLOW_TRACKING_USERNAME"] = "[insert username here]"
# os.environ["MLFLOW_TRACKING_PASSWORD"] = "[insert password here]"
# os.environ["MLFLOW_TRACKING_URI"] = (
#     "https://dagshub.com/anaqi.amirrazif/FlavorFusion.mlflow"
# )

# Get df
simple_recipes = pd.read_csv("data/simple_recipes.csv")

# Set inputs
queries = {
    "query1": {
        "user_ingredients": ["chicken", "rice", "salt", "pepper", "honey", "garlic"],
        "allergens": ["pork"],
        "protein": (20, 60),
        "top_n": 50,
    },
    "query2": {
        "user_ingredients": [
            "tortilla",
            "beans",
            "beef",
            "sour cream",
            "salsa",
            "tomatoes",
            "lettuce",
        ],
        "allergens": ["cashew"],
        "protein": (10, 80),
        "top_n": 50,
    },
}

output_similarity = []
encoding_times = []
building_times = []
searching_times = []

for query in queries:
    # IVF recs
    model = faiss_rec.faiss_model(simple_recipes.head(10000), index="IVF")
    ivf_encode_time = model.get_encode_time()
    ivf_build_time = model.get_build_time()

    start = time.perf_counter()
    ivf_recs = set(
        model.recommend_recipes(
            user_ingredients=queries[query]["user_ingredients"],
            allergens=queries[query]["allergens"],
            protein=queries[query]["protein"],
            top_n=queries[query]["top_n"],
        )
    )
    end = time.perf_counter()
    ivf_search_time = end - start

    # FlatL2 recs
    model = faiss_rec.faiss_model(simple_recipes.head(10000), index="FlatL2")
    flat_encode_time = model.get_encode_time()
    flat_build_time = model.get_build_time()

    start = time.perf_counter()
    flat_recs = set(
        model.recommend_recipes(
            user_ingredients=queries[query]["user_ingredients"],
            allergens=queries[query]["allergens"],
            protein=queries[query]["protein"],
            top_n=queries[query]["top_n"],
        )
    )
    end = time.perf_counter()
    flat_search_time = end - start

    # Get Jaccard Similarity
    intersection = ivf_recs.intersection(flat_recs)
    union = ivf_recs.union(flat_recs)
    jacc_sim = float(len(intersection)) / float(len(union))

    # Update metrics
    encoding_times.append((ivf_encode_time, flat_encode_time))
    building_times.append((ivf_build_time, flat_build_time))
    searching_times.append((ivf_search_time, flat_search_time))
    output_similarity.append(jacc_sim)

print(f"The output similarities are: {output_similarity}")
print(
    f"The average output similarity is: {sum(output_similarity)/len(output_similarity)}"
)
print(f"Encoding times (IVF, Flat): {encoding_times}")
print(f"Building times (IVF, Flat): {building_times}")
print(f"Searching times (IVF, Flat): {searching_times}")

# Create a new MLflow Experiment
# mlflow.set_experiment("MLflow Quickstart")

# with mlflow.start_run(run_name="FAISS test run"):
#     mlflow.log_param("nprobe", "50")
#     mlflow.log_metric("MAP", 0.75)

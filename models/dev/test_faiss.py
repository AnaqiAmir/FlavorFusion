from faiss_indexes import FlatIndex

import time


# Create flat index
start = time.perf_counter()
flat_index = FlatIndex(metadata_file_path="recipe_embeddings.json")
end = time.perf_counter()
print(end - start)

# Get recs
start = time.perf_counter()
recs = flat_index.recommend_recipes(
    user_ingredients=["chicken", "honey", "garlic"],
    calories=(250, 750),
)
end = time.perf_counter()

print(recs)
print(end - start)

###################
#### Save index ###
###################
# flat_index.save_index("faiss_index.bin")

####################
### Reload index ###
####################
# flat_index2 = FlatIndex(
#     "recipe_embeddings.json",
#     "faiss_index.bin",
# )

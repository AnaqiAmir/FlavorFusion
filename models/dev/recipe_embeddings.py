import ast
import time

import torch
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


class RecipeEmbeddings:
    def __init__(self, recipes_df: pd.DataFrame):
        # Switch to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create copy of dataframe
        self.recipes_df = recipes_df.copy()

        # Get ingredient names as list (needed for allergen filter later on)
        self.recipes_df["ingredient_list"] = self.recipes_df["ingredient_names"].apply(
            ast.literal_eval
        )

        # Prepare ingredients for embedding
        self.recipes_df["ingredient_names"] = (
            self.recipes_df["ingredient_names"].str.strip().str.lower()
        )
        self.embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2").to(
            self.device
        )

        # Embed ingredients
        self.ingredient_embeddings, self.embedding_time = self._embed_ingredients()

    def _embed_ingredients(self):
        start = time.perf_counter()
        ingredient_embeddings = (
            self.embedding_model.encode(
                self.recipes_df["ingredient_names"].tolist(), convert_to_tensor=True
            )
            .cpu()
            .numpy()
        )
        ingredient_embeddings = np.asarray(ingredient_embeddings.astype("float32"))
        end = time.perf_counter()
        embedding_time = end - start
        return ingredient_embeddings, embedding_time

    def get_recipes_df(self):
        return self.recipes_df

    def get_embedding_model(self):
        return self.embedding_model

    def get_ingredient_embeddings(self):
        return self.ingredient_embeddings

    def get_nutrition_min_max(self):
        nutrition_min_max = {
            "calorie_min": self.recipes_df["calories (#)"].min(),
            "calorie_max": self.recipes_df["calories (#)"].max(),
            "total_fat_min": self.recipes_df["total_fat (g)"].min(),
            "total_fat_max": self.recipes_df["total_fat (g)"].max(),
            "sugar_min": self.recipes_df["sugar (g)"].min(),
            "sugar_max": self.recipes_df["sugar (g)"].max(),
            "sodium_min": self.recipes_df["sodium (mg)"].min(),
            "sodium_max": self.recipes_df["sodium (mg)"].max(),
            "protein_min": self.recipes_df["protein (g)"].min(),
            "protein_max": self.recipes_df["protein (g)"].max(),
            "saturated_fat_min": self.recipes_df["saturated_fat (g)"].min(),
            "saturated_fat_max": self.recipes_df["saturated_fat (g)"].max(),
            "carbs_min": self.recipes_df["carbs (g)"].min(),
            "carbs_max": self.recipes_df["carbs (g)"].max(),
        }
        return nutrition_min_max

    def get_embedding_time(self):
        return self.embedding_time

import ast
import time
import json

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

    def get_embedding_time(self):
        return self.embedding_time

    def save_embeddings(self, file_path: str):
        """
        Save the dataframe, embedding model, ingredient embeddings, and embedding time to a JSON file.
        """
        # Save metadata
        metadata = {
            "recipes_df": (
                self.recipes_df.to_json(orient="split")
                if self.recipes_df is not None
                else None
            ),
            "embedding_model": "paraphrase-MiniLM-L6-v2",  # Save the identifier for reinitialization
            "ingredient_embeddings": self.ingredient_embeddings.tolist(),
            "embedding_time": self.embedding_time,
        }

        # Save metadata into json file
        with open(file_path, "w") as f:
            json.dump(metadata, f)
        print(f"Metadata saved to {file_path}")

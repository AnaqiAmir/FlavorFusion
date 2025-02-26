import time
from typing import Tuple, List
import json

import faiss
import pandas as pd
import numpy as np
from rapidfuzz import fuzz
from sentence_transformers import SentenceTransformer
from io import StringIO


class BaseFaissIndex:
    def __init__(self, metadata_file_path: str, index_file_path: str = None):
        # Load index (if applicable)
            self.index = faiss.read_index(index_file_path) if index_file_path else None

        # Load metadata
        with open(metadata_file_path, "r") as f:
            metadata = json.load(f)

        # Recover the dataframe
        self.recipes_df = (
            pd.read_json(StringIO(metadata["recipes_df"]), orient="split")
            if metadata["recipes_df"]
            else None
        )

        # Recover embedding time
        self.embedding_time = metadata.get("embedding_time", None)

        # Convert ingredient embeddings back to a NumPy array
        self.ingredient_embeddings = np.array(
            metadata["ingredient_embeddings"], dtype=np.float32
        )

        # Reinitialize the embedding model using the saved model name
        model_name = metadata.get("embedding_model", "paraphrase-MiniLM-L6-v2")
        self.embedding_model = SentenceTransformer(model_name)

        # Initialize build time
        self.build_time = 0

    def _nutrition_filter(
        self,
        recipes_df: pd.DataFrame,
        calories: Tuple[float, float],
        total_fat: Tuple[float, float],
        sugar: Tuple[float, float],
        sodium: Tuple[float, float],
        protein: Tuple[float, float],
        saturated_fat: Tuple[float, float],
        carbs: Tuple[float, float],
    ) -> pd.DataFrame:
        """
        Filter recipes based on user-specified nutritional constraints.

        Args:
        - recipes_df (DataFrame): The dataframe containing all relevant recipes.
        - calories (float): A user-specified calorie constraint.
        - total_fat (float): A user-specified total_fat constraint.
        - sugar (float): A user-specified sugar constraint.
        - sodium (float): A user-specified sodium constraint.
        - protein (float): A user-specified protein constraint.
        - saturated_fat (float): A user-specified saturated_fat constraint.
        - carbs (float): A user-specified carbs constraint.

        Returns:
        - DataFrame: A dataframe that only contains recipes within the specified constraint.
        """
        # Get min-max values for each nutrition
        nutrition_ranges = {
            "calories (#)": calories,
            "total_fat (g)": total_fat,
            "sugar (g)": sugar,
            "sodium (mg)": sodium,
            "protein (g)": protein,
            "saturated_fat (g)": saturated_fat,
            "carbs (g)": carbs,
        }

        # Start with all rows unselected
        mask = np.zeros(len(recipes_df), dtype=bool)

        # Apply each provided constraint
        for nutrition, (min_val, max_val) in nutrition_ranges.items():
            mask |= (recipes_df[nutrition] <= min_val) | (
                recipes_df[nutrition] >= max_val
            )

        return recipes_df[~mask]

    def _contains_allergen(
        self, ingredients: List[str], allergens: List[str], threshold: int = 90
    ) -> bool:
        """
        Returns a boolean indicating whether a recipe contains user-specified allergens based on a fuzzy search.

        Args:
        - ingredients (List[str]): A list of all present ingredients in a recipe.
        - allergens (List[str]): A list of user-specified allergens.
        - threshold (int): The similarity score of a single ingredient compared to a list of allergens.

        Returns:
        - bool: The return value. True if recipe contains an allergen, False otherwise.
        """
        if not isinstance(ingredients, list):  # Ensure ingredients is a list
            return False
        else:
            for ingredient in ingredients:
                for allergen in allergens:
                    score = fuzz.WRatio(ingredient, allergen)
                    if score >= threshold:
                        return True  # Remove this recipe
            return False  # Keep this recipe

    def _search_nearest_neighbors(
        self, id_selector: np.array, user_ingredients: List[str], top_n: int
    ):
        """
        Find the nearest top_n recipes based on user_ingredients within the scope of id_selector.

        Args:
        - id_select (np.array): An array of eligible recipes ids.
        - user_ingredients (list): A list of ingredients provided by the user.
        - top_n (int): The number of recommended recipes to return.

        Returns:
        - list: A list of recommended recipes.
        """
        user_vector = (
            self.embedding_model.encode(user_ingredients, convert_to_tensor=True)
            .cpu()
            .numpy()
        )
        _, filtered_indices = self.index.search(
            user_vector,
            k=top_n,
            params=(faiss.SearchParameters(sel=id_selector)),
        )
        return self.recipes_df.iloc[filtered_indices[0]]["name"].tolist()

    def recommend_recipes(
        self,
        user_ingredients: List[str],
        allergens: List[str] = None,
        calories: Tuple[float, float] = (None, None),
        total_fat: Tuple[float, float] = (None, None),
        sugar: Tuple[float, float] = (None, None),
        sodium: Tuple[float, float] = (None, None),
        protein: Tuple[float, float] = (None, None),
        saturated_fat: Tuple[float, float] = (None, None),
        carbs: Tuple[float, float] = (None, None),
        top_n: int = 5,
    ) -> List[str]:
        """
        Gives a list of top_n recommended recipes based on the given user_ingredients.

        Args:
        - user_ingredients (list): A list of ingredients provided by the user.
        - allergens (list): A list of allergens from the user.
                            Recipes that contain ingredients in allergens will not be recommended.
        - calories (float): A user-specified calorie (#) constraint.
        - total_fat (float): A user-specified total_fat (grams) constraint.
        - sugar (float): A user-specified sugar (grams) constraint.
        - sodium (float): A user-specified sodium (miligrams) constraint.
        - protein (float): A user-specified protein (grams) constraint.
        - saturated_fat (float): A user-specified saturated_fat (grams) constraint.
        - carbs (float): A user-specified carbs (grams) constraint.
        - top_n (int): The number of recommended recipes to return.

        Returns:
        - list: A list of recommended recipes.
        """
        # Assign default min/max values for nutrition directly from the dataframe
        nutrition_defaults = {
            "calories": (
                self.recipes_df["calories (#)"].min(),
                self.recipes_df["calories (#)"].max(),
            ),
            "total_fat": (
                self.recipes_df["total_fat (g)"].min(),
                self.recipes_df["total_fat (g)"].max(),
            ),
            "sugar": (
                self.recipes_df["sugar (g)"].min(),
                self.recipes_df["sugar (g)"].max(),
            ),
            "sodium": (
                self.recipes_df["sodium (mg)"].min(),
                self.recipes_df["sodium (mg)"].max(),
            ),
            "protein": (
                self.recipes_df["protein (g)"].min(),
                self.recipes_df["protein (g)"].max(),
            ),
            "saturated_fat": (
                self.recipes_df["saturated_fat (g)"].min(),
                self.recipes_df["saturated_fat (g)"].max(),
            ),
            "carbs": (
                self.recipes_df["carbs (g)"].min(),
                self.recipes_df["carbs (g)"].max(),
            ),
        }

        # Dictionary to store the final nutrition constraints
        nutrition_constraints = {
            "calories": calories,
            "total_fat": total_fat,
            "sugar": sugar,
            "sodium": sodium,
            "protein": protein,
            "saturated_fat": saturated_fat,
            "carbs": carbs,
        }

        # Update constraints with defaults if not provided
        for nutrient, default in nutrition_defaults.items():
            if nutrition_constraints[nutrient] == (None, None):
                nutrition_constraints[nutrient] = default

        # Filter out recipes that are not within the specified nutrition range
        filtered_recipes = self._nutrition_filter(
            self.recipes_df,
            nutrition_constraints["calories"],
            nutrition_constraints["total_fat"],
            nutrition_constraints["sugar"],
            nutrition_constraints["sodium"],
            nutrition_constraints["protein"],
            nutrition_constraints["saturated_fat"],
            nutrition_constraints["carbs"],
        )

        # Filter out recipes that contain allergens
        if allergens is not None:
            filtered_recipes = filtered_recipes[
                ~filtered_recipes["ingredient_list"].apply(
                    lambda x: self._contains_allergen(x, allergens)
                )
            ]

        # Get ids of relevant recipes
        filtered_ids = filtered_recipes.index
        id_selector = faiss.IDSelectorBatch(filtered_ids)

        # SEARCH
        recommended_recipes = self._search_nearest_neighbors(
            id_selector, user_ingredients, top_n
        )

        return recommended_recipes

    def get_build_time(self):
        return self.build_time

    def save_index(self, index_file_path: str):
        """
        Save the FAISS index using FAISS's built-in serialization.
        Optionally, also save additional metadata (including embeddings and model info) to a JSON file.
        """
        if self.index is None:
            raise ValueError("FAISS index is not initialized or built.")

        # Save the FAISS index
        faiss.write_index(self.index, index_file_path)
        print(f"FAISS index saved to {index_file_path}")


class FlatIndex(BaseFaissIndex):
    def __init__(
        self, metadata_file_path: str, index_file_path: str = None, metric: str = "L2"
    ):
        super().__init__(metadata_file_path, index_file_path)
        start = time.perf_counter()
        self.vector_size = self.ingredient_embeddings.shape[1]

        # Validate metric input
        valid_metrics = {"L2", "IP"}
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of {valid_metrics}."
            )

        # Initialize the FAISS index
        self.index = (
            faiss.IndexFlatL2(self.vector_size)
            if metric == "L2"
            else faiss.IndexFlatIP(self.vector_size)
        )

        self.index.add(self.ingredient_embeddings)
        end = time.perf_counter()
        self.build_time = end - start


class IVFFlatIndex(BaseFaissIndex):
    def __init__(
        self,
        metadata_file_path: str,
        num_of_cells: str,
        nprobe: int,
        index_file_path: str = None,
        metric: str = "L2",
    ):
        super().__init__(metadata_file_path, index_file_path)
        self.num_of_cells = num_of_cells  # num of voronoi cells
        self.nprobe = nprobe
        start = time.perf_counter()
        self.vector_size = self.ingredient_embeddings.shape[1]

        # Validate metric input
        valid_metrics = {"L2", "IP"}
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of {valid_metrics}."
            )

        # Initialize the FAISS index
        self.quantizer = (
            faiss.IndexFlatL2(self.vector_size)
            if metric == "L2"
            else faiss.IndexFlatIP(self.vector_size)
        )

        # Build Index
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.vector_size, self.num_of_cells
        )
        self.index.train(self.ingredient_embeddings)
        self.ids = np.array(range(0, self.ingredient_embeddings.shape[0]))
        self.ids = np.asarray(self.ids.astype("int64"))
        self.index.add_with_ids(self.ingredient_embeddings, self.ids)
        end = time.perf_counter()
        self.build_time = end - start

    def _search_nearest_neighbors(
        self, id_selector: np.array, user_ingredients: List[str], top_n: int
    ):
        user_vector = (
            self.embedding_model.encode(user_ingredients, convert_to_tensor=True)
            .cpu()
            .numpy()
        )
        _, filtered_indices = self.index.search(
            user_vector,
            k=top_n,
            params=(faiss.SearchParametersIVF(sel=id_selector, nprobe=self.nprobe)),
        )
        return self.recipes_df.iloc[filtered_indices[0]]["name"].tolist()


class PQIndex(BaseFaissIndex):
    def __init__(
        self,
        metadata_file_path: str,
        index_file_path: str = None,
        m: int = 8,
        nbits: int = 8,
    ):
        super().__init__(metadata_file_path, index_file_path)
        self.m = m
        self.nbits = nbits
        start = time.perf_counter()
        self.vector_size = self.ingredient_embeddings.shape[1]
        self.index = faiss.IndexPQ(self.vector_size, self.m, self.nbits)
        self.index.train(self.ingredient_embeddings)
        self.ids = np.array(range(0, self.ingredient_embeddings.shape[0]))
        self.ids = np.asarray(self.ids.astype("int64"))
        self.index2 = faiss.IndexIDMap(self.index)
        self.index2.add_with_ids(self.ingredient_embeddings, self.ids)
        end = time.perf_counter()
        self.build_time = end - start

    def _search_nearest_neighbors(
        self, id_selector: np.array, user_ingredients: List[str], top_n: int
    ):
        user_vector = (
            self.embedding_model.encode(user_ingredients, convert_to_tensor=True)
            .cpu()
            .numpy()
        )
        _, filtered_indices = self.index.search(
            user_vector,
            k=top_n,
            params=(faiss.SearchParametersPQ(sel=id_selector, nprobe=self.nprobe)),
        )
        return self.recipes_df.iloc[filtered_indices[0]]["name"].tolist()


class IVFPQIndex(BaseFaissIndex):
    def __init__(
        self,
        metadata_file_path: str,
        num_of_cells: str,
        nprobe: int,
        index_file_path: str = None,
        m: int = 8,
        nbits: int = 8,
        metric: str = "L2",
    ):
        super().__init__(metadata_file_path, index_file_path)
        self.num_of_cells = num_of_cells  # num of voronoi cells
        self.nprobe = nprobe
        self.m = m
        self.nbits = nbits
        start = time.perf_counter()
        self.vector_size = self.ingredient_embeddings.shape[1]

        # Validate metric input
        valid_metrics = {"L2", "IP"}
        if metric not in valid_metrics:
            raise ValueError(
                f"Invalid metric '{metric}'. Must be one of {valid_metrics}."
            )

        # Initialize the FAISS index
        self.quantizer = (
            faiss.IndexFlatL2(self.vector_size)
            if metric == "L2"
            else faiss.IndexFlatIP(self.vector_size)
        )

        self.index = faiss.IndexIVFPQ(
            self.quantizer, self.vector_size, self.num_of_cells, self.m, self.nbits
        )
        self.index.train(self.ingredient_embeddings)
        self.ids = np.array(range(0, self.ingredient_embeddings.shape[0]))
        self.ids = np.asarray(self.ids.astype("int64"))
        self.index2 = faiss.IndexIDMap(self.index)
        self.index2.add_with_ids(self.ingredient_embeddings, self.ids)
        end = time.perf_counter()
        self.build_time = end - start

    def _search_nearest_neighbors(
        self, id_selector: np.array, user_ingredients: List[str], top_n: int
    ):
        user_vector = (
            self.embedding_model.encode(user_ingredients, convert_to_tensor=True)
            .cpu()
            .numpy()
        )
        _, filtered_indices = self.index.search(
            user_vector,
            k=top_n,
            params=(faiss.SearchParametersIVF(sel=id_selector, nprobe=self.nprobe)),
        )
        return self.recipes_df.iloc[filtered_indices[0]]["name"].tolist()


class HNSWIndex(BaseFaissIndex):
    def __init__(
        self, metadata_file_path: str, index_file_path: str = None, m: int = 8
    ):
        super().__init__(metadata_file_path, index_file_path)
        self.m = m
        start = time.perf_counter()
        self.vector_size = self.ingredient_embeddings.shape[1]
        self.index = faiss.IndexHNSWFlat(self.vector_size, self.m)
        self.index.add(self.ingredient_embeddings)
        end = time.perf_counter()
        self.build_time = end - start

    def _search_nearest_neighbors(self, id_selector, user_ingredients, top_n):
        user_vector = (
            self.embedding_model.encode(user_ingredients, convert_to_tensor=True)
            .cpu()
            .numpy()
        )
        _, filtered_indices = self.index.search(
            user_vector,
            k=top_n,
            params=(faiss.SearchParametersHNSW(sel=id_selector)),
        )
        return self.recipes_df.iloc[filtered_indices[0]]["name"].tolist()

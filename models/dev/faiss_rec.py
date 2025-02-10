import ast
import time
from typing import Tuple, List

import faiss
import torch
import pandas as pd
import numpy as np
from rapidfuzz import process
from sentence_transformers import SentenceTransformer


class faiss_model:
    """
    A class to build and manage a FAISS index for recipe recommendation based on ingredient similarity.

    This model encodes recipe ingredient lists into vector representations, builds a FAISS index for efficient
    nearest-neighbor search, and provides methods for querying recipes based on input ingredients.

    Attributes:
        recipes_df (pd.DataFrame): The DataFrame containing recipe data. Must include a column `ingredient_names`
                                   where each entry is a list of ingredient names.
        unique_ingredients (list): A sorted list of all unique ingredients across all recipes.
        ingredient_to_idx (dict): A mapping of ingredients to their unique integer indices.
        vector_size (int): The size of the vectors, equal to the number of unique ingredients.
        index (faiss.IndexIVFFlat): The FAISS index used for nearest-neighbor search.
        vectors (np.ndarray): A matrix of encoded ingredient vectors, where each row represents a recipe.

    Methods:
        Methods:
        __init__(recipes_df):
            Initializes the FAISS model, extracts unique ingredients, builds a mapping of ingredients to indices,
            encodes the recipe ingredients into vectors, and builds the FAISS index.

        nutrition_filter(recipes_df, calories, total_fat, sugar, sodium, protein, saturated_fat, carbs):
            Filters recipes based on user-specified nutritional constraints.

        contains_allergen():
            Returns a boolean indicating whether a recipe contains user-specified allergens based on a fuzzy search.

        recommend_recipes(user_ingredients, allergens=[''], calories=None, total_fat=None, sugar=None,
                          sodium=None, protein=None, saturated_fat=None, carbs=None, top_n=5):
            Recommends a list of top_n recipes based on user-provided ingredients, allergen restrictions,
            and nutritional constraints.

    Example:
        # Load recipes DataFrame
        recipes_df = pd.read_csv("recipes.csv")

        # Initialize the FAISS model
        model = faiss_model(recipes_df)

        # Get recommendations based on user input
        user_ingredients = ['chicken', 'garlic', 'onion']
        recommendations = model.recommend_recipes(user_ingredients, calories=(500,2000), top_n=5)
        print(recommendations)
    """

    def __init__(self, recipes_df: pd.DataFrame, index_key: str):
        """
        Initializes the FAISS model with recipe data and builds the FAISS index.

        Args:
            recipes_df (pd.DataFrame): A DataFrame containing recipe information. It must have a column
                                       `ingredient_names`, where each entry is a list of ingredient strings.

        Process:
            1. Extracts and sorts all unique ingredients from the `ingredient_names` column.
            2. Creates a mapping of each unique ingredient to a unique integer index.
            3. Encodes the ingredient lists into vectors, where each vector is a one-hot representation of ingredients.
            4. Initializes a FAISS index with vectors of recipes for efficient nearest-neighbor search.

        Raises:
            ValueError: If the `ingredient_names` column is missing in the input DataFrame or is not iterable.
        """
        # Index key
        self.index_key = index_key

        # Number of voronoi cells to be initalized in IndexIVFFlat
        if self.index_key == "IVF":
            NUM_OF_CELLS = 200

        # Switch to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Get unique ingredients
        self.recipes_df = recipes_df.copy()
        self.unique_ingredients = sorted(
            set(
                ingredient
                for ingredients in recipes_df["ingredient_names"]
                for ingredient in ingredients
            )
        )
        self.ingredient_to_idx = {
            ingredient: idx for idx, ingredient in enumerate(self.unique_ingredients)
        }

        # Get ingredient names as list
        self.recipes_df["ingredient_list"] = self.recipes_df["ingredient_names"].apply(
            ast.literal_eval
        )

        # Find min max values of each nutrition
        self.CALORIE_MIN = self.recipes_df["calories (#)"].min()
        self.CALORIE_MAX = self.recipes_df["calories (#)"].max()

        self.TOTAL_FAT_MIN = self.recipes_df["total_fat (g)"].min()
        self.TOTAL_FAT_MAX = self.recipes_df["total_fat (g)"].max()

        self.SUGAR_MIN = self.recipes_df["sugar (g)"].min()
        self.SUGAR_MAX = self.recipes_df["sugar (g)"].max()

        self.SODIUM_MIN = self.recipes_df["sodium (mg)"].min()
        self.SODIUM_MAX = self.recipes_df["sodium (mg)"].max()

        self.PROTEIN_MIN = self.recipes_df["protein (g)"].min()
        self.PROTEIN_MAX = self.recipes_df["protein (g)"].max()

        self.SATURATED_FAT_MIN = self.recipes_df["saturated_fat (g)"].min()
        self.SATURATED_FAT_MAX = self.recipes_df["saturated_fat (g)"].max()

        self.CARBS_MIN = self.recipes_df["carbs (g)"].min()
        self.CARBS_MAX = self.recipes_df["carbs (g)"].max()

        # Prepare ingredients for encoding
        self.recipes_df["ingredient_names"] = (
            self.recipes_df["ingredient_names"].str.strip().str.lower()
        )
        self.ingredients = self.recipes_df["ingredient_names"].tolist()
        self.model = SentenceTransformer("paraphrase-MiniLM-L6-v2").to(self.device)

        # Encode ingredients
        start = time.perf_counter()
        self.ingredient_embeddings = (
            self.model.encode(self.ingredients, convert_to_tensor=True).cpu().numpy()
        )
        self.ingredient_embeddings = np.asarray(
            self.ingredient_embeddings.astype("float32")
        )
        end = time.perf_counter()
        self.encode_time = end - start

        # Build FAISS IndexIVFFlat
        if self.index_key == "IVF":
            start = time.perf_counter()
            self.vector_size = self.ingredient_embeddings.shape[1]  # embedding dim
            self.quantizer = faiss.IndexFlatL2(self.vector_size)
            self.num_of_cells = NUM_OF_CELLS  # num of voronoi cells
            self.index = faiss.IndexIVFFlat(
                self.quantizer, self.vector_size, self.num_of_cells
            )
            self.index.train(self.ingredient_embeddings)
            self.ids = np.array(range(0, self.ingredient_embeddings.shape[0]))
            self.ids = np.asarray(self.ids.astype("int64"))
            self.index.add_with_ids(self.ingredient_embeddings, self.ids)
            end = time.perf_counter()
            self.build_time = end - start

        # Build FAISS IndexFlatL2
        elif self.index_key == "FlatL2":
            start = time.perf_counter()
            self.vector_size = self.ingredient_embeddings.shape[1]
            self.index = faiss.IndexFlatL2(self.vector_size)
            self.index.add(self.ingredient_embeddings)
            end = time.perf_counter()
            self.build_time = end - start

        # Build FAISS IndexFlatIP
        elif self.index_key == "FlatIP":
            start = time.perf_counter()
            self.vector_size = self.ingredient_embeddings.shape[1]
            self.index = faiss.IndexFlatIP(self.vector_size)
            self.index.add(self.ingredient_embeddings)
            end = time.perf_counter()
            self.build_time = end - start

        else:
            print(
                'Invalid \'index\' parameter. Index must either be index = ("IndexFlatL2", "IndexFlatIP", "IndexIVFFLat")'
            )

    def get_encode_time(self):
        return self.encode_time

    def get_build_time(self):
        return self.build_time

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

    # Filtering function
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
                _, score, _ = process.extractOne(ingredient, allergens)
                if score >= threshold:
                    return True  # Filter out this row
            return False  # Keep this row

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
        # Assign default min max values for nutritions
        nutrition_defaults = {
            "calories": (self.CALORIE_MIN, self.CALORIE_MAX),
            "total_fat": (self.TOTAL_FAT_MIN, self.TOTAL_FAT_MAX),
            "sugar": (self.SUGAR_MIN, self.SUGAR_MAX),
            "sodium": (self.SODIUM_MIN, self.SODIUM_MAX),
            "protein": (self.PROTEIN_MIN, self.PROTEIN_MAX),
            "saturated_fat": (self.SATURATED_FAT_MIN, self.SATURATED_FAT_MAX),
            "carbs": (self.CARBS_MIN, self.CARBS_MAX),
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

        # Filter out recipes that contain allergens
        if allergens is not None:
            filtered_recipes = self.recipes_df[
                ~self.recipes_df["ingredient_list"].apply(
                    lambda x: self._contains_allergen(x, allergens)
                )
            ]
        else:
            filtered_recipes = self.recipes_df

        # Filter out recipes that are not within the specified nutrition range
        filtered_recipes = self._nutrition_filter(
            filtered_recipes,
            nutrition_constraints["calories"],
            nutrition_constraints["total_fat"],
            nutrition_constraints["sugar"],
            nutrition_constraints["sodium"],
            nutrition_constraints["protein"],
            nutrition_constraints["saturated_fat"],
            nutrition_constraints["carbs"],
        )

        # Get ids of relevant recipes
        filtered_ids = filtered_recipes.index
        id_selector = faiss.IDSelectorBatch(filtered_ids)

        # SEARCH
        user_vector = (
            self.model.encode(user_ingredients, convert_to_tensor=True).cpu().numpy()
        )
        _, filtered_indices = self.index.search(
            user_vector,
            k=top_n,
            params=(
                faiss.SearchParametersIVF(sel=id_selector, nprobe=10)
                if self.index_key == "IVF"
                else faiss.SearchParameters(sel=id_selector)
            ),
        )
        return self.recipes_df.iloc[filtered_indices[0]]["name"].tolist()


###################
## Example usage ##
###################

if __name__ == "__main__":
    # Load data
    simple_recipes = pd.read_csv("data/simple_recipes.csv")

    # Test on "aromatic basmati rice  rice cooker" ingredients
    model = faiss_model(simple_recipes.head(1000), index="FlatIP")
    start = time.perf_counter()
    recs = model.recommend_recipes(
        user_ingredients=[
            "basmati rice",
            "water",
            "salt",
            "cinnamon stick",
            "green cardamom pod",
        ],
        allergens=["goat cheese", "peanut"],
        calories=(202.5, 247.5),
        sugar=(0.9, 1.1),
        sodium=(0, 10000),
        saturated_fat=(0, 10000),
        total_fat=(0, 10000),
        carbs=(0, 10000),
        protein=(0, 10000),
        top_n=11,
    )
    end = time.perf_counter()
    print("Recommended Recipes:")
    for rec in recs:
        print(rec)
    print("Search time: ", end - start)

    # Test on "pumpkin" with "cashew" allergies
    start = time.perf_counter()
    recs = model.recommend_recipes(
        user_ingredients=["pumpkin"], allergens=["cashew"], top_n=11
    )
    end = time.perf_counter()
    print("Recommended Recipes:")
    for rec in recs:
        print(rec)
    print("Search time: ", end - start)

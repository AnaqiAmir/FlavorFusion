import pandas as pd
import numpy as np
from typing import Tuple, List
from rapidfuzz import process
import faiss
import ast


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
        index (faiss.IndexFlatL2): The FAISS index used for nearest-neighbor search.
        vectors (np.ndarray): A matrix of encoded ingredient vectors, where each row represents a recipe.

    Methods:
        Methods:
        __init__(recipes_df):
            Initializes the FAISS model, extracts unique ingredients, builds a mapping of ingredients to indices,
            encodes the recipe ingredients into vectors, and builds the FAISS index.

        encode_ingredients(ingredients, ingredient_to_idx, unique_ingredients):
            Encodes a list of ingredients into a vector representation using one-hot encoding.

        nutrition_filter(recipes_df, calories, total_fat, sugar, sodium, protein, saturated_fat, carbs):
            Filters recipes based on user-specified nutritional constraints.

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
        recommendations = model.recommend_recipes(user_ingredients, calories=500, top_n=5)
        print(recommendations)
    """

    def __init__(self, recipes_df: pd.DataFrame):
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

        # BUILD: Initialize index for FAISS
        self.vector_size = len(self.unique_ingredients)
        self.vectors = np.vstack(
            [
                self._encode_all_ingredients(ingredients)
                for ingredients in self.recipes_df["ingredient_names"]
            ]
        )
        self.quantizer = faiss.IndexFlatIP(self.vector_size)
        self.num_of_cells = 200
        self.index = faiss.IndexIVFFlat(
            self.quantizer, self.vector_size, self.num_of_cells
        )
        self.index.train(self.vectors)
        ids = np.arange(self.vectors.shape[0])
        self.index.add_with_ids(self.vectors, ids)

    def _encode_all_ingredients(self, ingredients: list) -> np.array:
        """
        Generate vector encodings for ingredients-to-recipes.

        Args:
        - ingredients (list): A list of ingredients in the recipe.
        - ingredient_to_idx (dict): A dict mapping ingredients to their respective ids.
        - unique_ingredients (list): A list of unique ingredients.

        Returns:
        - vector (np.array): An encoding of which ingredients are in the recipe w.r.t to all available ingredients.
        """
        vector = np.zeros(len(self.unique_ingredients), dtype="float32")
        for ingredient in ingredients:
            vector[self.ingredient_to_idx[ingredient]] = 1.0
        return vector

    def _encode_user_ingredients(self, ingredients: list) -> np.array:
        """
        Generate vector encodings for ingredients-to-recipes.

        Args:
        - ingredients (list): A list of ingredients in the recipe.
        - ingredient_to_idx (dict): A dict mapping ingredients to their respective ids.
        - unique_ingredients (list): A list of unique ingredients.

        Returns:
        - vector (np.array): An encoding of which ingredients are in the recipe w.r.t to all available ingredients.
        """
        vector = np.zeros(len(self.unique_ingredients), dtype="float32")
        for ingredient in ingredients:
            if ingredient in self.unique_ingredients:
                vector[self.ingredient_to_idx[ingredient]] = 1.0
        return vector

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
        allergens: List[str] = [""],
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

        # # FILTER: Filter out index of filtered recipes
        # filtered_recipes = self.recipes_df[
        #     ~self.recipes_df["ingredient_names"].apply(
        #         lambda ingredients: any(item in allergens for item in ingredients)
        #     )
        # ]

        # Apply filtering
        filtered_recipes = self.recipes_df[
            ~self.recipes_df["ingredient_names"].apply(
                lambda x: self._contains_allergen(x, allergens)
            )
        ]

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
        filtered_ids = filtered_recipes.index
        id_selector = faiss.IDSelectorBatch(filtered_ids)

        # SEARCH
        user_vector = self._encode_user_ingredients(user_ingredients).reshape(1, -1)
        _, filtered_indices = self.index.search(
            user_vector,
            k=top_n,
            params=faiss.SearchParametersIVF(sel=id_selector, nprobe=20),
        )
        return self.recipes_df.iloc[filtered_indices[0]]["name"].tolist()


###################
## Example usage ##
###################

if __name__ == "__main__":
    import time

    # Load data
    simple_recipes = pd.read_csv("data/simple_recipes.csv")
    simple_recipes["ingredient_names"] = simple_recipes["ingredient_names"].apply(
        ast.literal_eval
    )

    # Test on "aromatic basmati rice  rice cooker" ingredients
    start = time.time()
    model = faiss_model(simple_recipes)
    end = time.time()
    print("Build time: ", end - start)

    start = time.time()
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
    end = time.time()
    print("Recommended Recipes:")
    for rec in recs:
        print(rec)
    print("Search time: ", end - start)

    # Test on "pumpkin" with "cashew" allergies
    start = time.time()
    recs = model.recommend_recipes(
        user_ingredients=["pumpkin"], allergens=["cashew"], top_n=11
    )
    end = time.time()
    print("Recommended Recipes:")
    for rec in recs:
        print(rec)
    print("Search time: ", end - start)

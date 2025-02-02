import pandas as pd
import numpy as np
import faiss
import ast

class faiss_model():
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

        get_min_max_calories(value):
            Computes a calorie range within ±10% of a specified value. Defaults to (0, 10000) if the value is None.

        get_min_max(value):
            Computes a nutritional range within ±50% of a specified value. Defaults to (0, 10000) if the value is None.

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
        self.recipes_df = recipes_df
        self.unique_ingredients = sorted(set(ingredient for ingredients in recipes_df['ingredient_names']
                                             for ingredient in ingredients))
        self.ingredient_to_idx = {ingredient: idx for idx, ingredient in enumerate(self.unique_ingredients)}

        # Find min max values of each nutrition
        self.CALORIE_MIN = self.recipes_df['calories (#)'].min()
        self.CALORIE_MAX = self.recipes_df['calories (#)'].max()

        self.TOTAL_FAT_MIN = self.recipes_df['total_fat (g)'].min()
        self.TOTAL_FAT_MAX = self.recipes_df['total_fat (g)'].max()

        self.SUGAR_MIN = self.recipes_df['sugar (g)'].min()
        self.SUGAR_MAX = self.recipes_df['sugar (g)'].max()

        self.SODIUM_MIN = self.recipes_df['sodium (mg)'].min()
        self.SODIUM_MAX = self.recipes_df['sodium (mg)'].max()

        self.PROTEIN_MIN = self.recipes_df['protein (g)'].min()
        self.PROTEIN_MAX = self.recipes_df['protein (g)'].max()

        self.SATURATED_FAT_MIN = self.recipes_df['saturated_fat (g)'].min()
        self.SATURATED_FAT_MAX = self.recipes_df['saturated_fat (g)'].max()

        self.CARBS_MIN = self.recipes_df['carbs (g)'].min()
        self.CARBS_MAX = self.recipes_df['carbs (g)'].max()

        # BUILD: Initialize index for FAISS
        self.vector_size = len(self.unique_ingredients)
        self.index = faiss.IndexFlatL2(self.vector_size)
        self.vectors = np.vstack([self._encode_all_ingredients(ingredients, self.ingredient_to_idx, self.unique_ingredients)
                                  for ingredients in recipes_df['ingredient_names']])
        self.index.add(self.vectors)

    def _encode_all_ingredients(self, ingredients: list, ingredient_to_idx: dict, unique_ingredients: list):
        """
        Generate vector encodings for ingredients-to-recipes.

        Args:
        - ingredients (list): A list of ingredients in the recipe.
        - ingredient_to_idx (dict): A dict mapping ingredients to their respective ids.
        - unique_ingredients (list): A list of unique ingredients.

        Returns:
        - vector (np.array): An encoding of which ingredients are in the recipe w.r.t to all available ingredients.
        """
        vector = np.zeros(len(unique_ingredients), dtype='float32')
        for ingredient in ingredients:
            vector[ingredient_to_idx[ingredient]] = 1.0
        return vector

    def _encode_user_ingredients(self, ingredients: list, ingredient_to_idx: dict, unique_ingredients: list):
        """
        Generate vector encodings for ingredients-to-recipes.

        Args:
        - ingredients (list): A list of ingredients in the recipe.
        - ingredient_to_idx (dict): A dict mapping ingredients to their respective ids.
        - unique_ingredients (list): A list of unique ingredients.

        Returns:
        - vector (np.array): An encoding of which ingredients are in the recipe w.r.t to all available ingredients.
        """
        vector = np.zeros(len(unique_ingredients), dtype='float32')
        for ingredient in ingredients:
            if ingredient in unique_ingredients:
                vector[ingredient_to_idx[ingredient]] = 1.0
        return vector

    def _nutrition_filter(
        self,
        recipes_df: pd.DataFrame,
        calories,
        total_fat,
        sugar,
        sodium,
        protein,
        saturated_fat,
        carbs
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
        # Get min max of each nutrition
        calorie_min, calorie_max = calories
        total_fat_min, total_fat_max = total_fat
        sugar_min, sugar_max = sugar
        sodium_min, sodium_max = sodium
        protein_min, protein_max = protein
        saturated_fat_min, saturated_fat_max = saturated_fat
        carbs_min, carbs_max = carbs

        # Filter recipes to be within the specified range for any nutritional component
        unwanted_recipes = recipes_df[
            (recipes_df['calories (#)'] <= calorie_min) | (recipes_df['calories (#)'] >= calorie_max) |
            (recipes_df['total_fat (g)'] <= total_fat_min) | (recipes_df['total_fat (g)'] >= total_fat_max) |
            (recipes_df['sugar (g)'] <= sugar_min) | (recipes_df['sugar (g)'] >= sugar_max) |
            (recipes_df['sodium (mg)'] <= sodium_min) | (recipes_df['sodium (mg)'] >= sodium_max) |
            (recipes_df['protein (g)'] <= protein_min) | (recipes_df['protein (g)'] >= protein_max) |
            (recipes_df['saturated_fat (g)'] <= saturated_fat_min) | (recipes_df['saturated_fat (g)'] >= saturated_fat_max) |
            (recipes_df['carbs (g)'] <= carbs_min) | (recipes_df['carbs (g)'] >= carbs_max)
        ]
        filtered_recipes = recipes_df.drop(unwanted_recipes.index)

        return filtered_recipes

    def recommend_recipes(
        self,
        user_ingredients: list,
        allergens: list=[''],
        calories: tuple[float,float] = (None, None),
        total_fat: tuple[float,float] = (None, None),
        sugar: tuple[float,float] = (None, None),
        sodium: tuple[float,float] = (None, None),
        protein: tuple[float,float] = (None, None),
        saturated_fat: tuple[float,float] = (None, None),
        carbs: tuple[float,float] = (None, None),
        top_n: int=5
        ) -> list:
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
        # Assign min max values for nutritions
        if calories == (None, None):
            calories = (self.CALORIE_MIN, self.CALORIE_MAX)
        if total_fat == (None, None):
            total_fat = (self.TOTAL_FAT_MIN, self.TOTAL_FAT_MAX)
        if sugar == (None, None):
            sugar = (self.SUGAR_MIN, self.SUGAR_MAX)
        if sodium == (None, None):
            sodium = (self.SODIUM_MIN, self.SODIUM_MAX)
        if protein == (None, None):
            protein = (self.PROTEIN_MIN, self.PROTEIN_MAX)
        if saturated_fat == (None, None):
            saturated_fat = (self.SATURATED_FAT_MIN, self.SATURATED_FAT_MAX)
        if carbs == (None, None):
            carbs = (self.CARBS_MIN, self.CARBS_MAX)

        # FILTER: Filter out index of filtered recipes
        filtered_recipes = self.recipes_df[~self.recipes_df['ingredient_names']
                                           .apply(lambda ingredients: any(item in allergens for item in ingredients))]
        filtered_recipes = self._nutrition_filter(filtered_recipes, calories, total_fat,
                                                  sugar, sodium, protein, saturated_fat, carbs)
        filtered_ids = filtered_recipes.index
        id_selector = faiss.IDSelectorArray(filtered_ids)

        # SEARCH
        user_vector = self._encode_user_ingredients(user_ingredients, self.ingredient_to_idx, self.unique_ingredients).reshape(1,-1)
        _, filtered_indices = self.index.search(user_vector, k=top_n, params=faiss.SearchParameters(sel=id_selector))
        return self.recipes_df.iloc[filtered_indices[0]]['name'].tolist()

###################
## Example usage ##
###################

if __name__ == "__main__":
    import time

    # Load data
    simple_recipes = pd.read_csv("data/simple_recipes.csv")
    simple_recipes['ingredient_names'] = simple_recipes['ingredient_names'].apply(ast.literal_eval)

    # Test on "aromatic basmati rice  rice cooker" ingredients
    start = time.time()
    model = faiss_model(simple_recipes)
    end = time.time()
    print("Build time: ", end-start)

    start = time.time()
    recs = model.recommend_recipes(user_ingredients=['basmati rice', 'water', 'salt', 'cinnamon stick', 'green cardamom pod'],
                                   allergens=['goat cheese'],
                                   calories = (202.5, 247.5),
                                   sugar = (0.9,1.1),
                                   sodium = (0,10000),
                                   saturated_fat = (0,10000),
                                   total_fat = (0,10000),
                                   carbs = (0,10000),
                                   protein = (0,10000),
                                   top_n=11)
    end = time.time()
    print("Search time: ", end-start)

    print(recs)

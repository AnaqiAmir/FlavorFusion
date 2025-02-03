import re
import json
from typing import Dict, Tuple, List, Any

# from models.dev.faiss_rec import faiss_model
# import pandas as pd, ast


class NLSParser:
    """
    Optimized Natural Language Synthesis (NLS) module for extracting:
      - Nutritional constraints (calories, fat, sugar, sodium, protein, etc.)
      - Allergens (list of allergens based on ontology)
      - Required ingredients (list of requested ingredients)
      - Dietary preferences (vegetarian, vegan, etc. from ontology)
    """

    # Recommended Daily Values (for reference)
    RECOMMENDED_DV = {
        "total_fat": 80,
        "sugar": 50,
        "sodium": 2300,
        "protein": 50,
        "saturated_fat": 20,
        "carbs": 275,
    }

    # Map keywords found in text to our nutritional attributes.
    NUTRITION_KEYWORDS = {
        "calories": "calories",
        "fat": "total_fat",  # Note: We assume “fat” in the text means total fat.
        "sugar": "sugar",
        "sodium": "sodium",
        "protein": "protein",
        "saturated fat": "saturated_fat",
        "carbs": "carbs",
    }

    def __init__(self, allergen_path: str, diet_path: str):
        """
        Initializes the parser by loading allergen and diet ontologies.

        Args:
            allergen_path (str): Path to allergens JSON ontology.
            diet_path (str): Path to diets JSON ontology.
        """
        self.allergen_ontology = self._load_ontology(allergen_path)
        self.diet_ontology = self._load_ontology(diet_path)

        # Flatten synonyms for quick lookup
        self.allergen_synonyms = self._flatten_ontology(self.allergen_ontology)
        self.diet_synonyms = self._flatten_ontology(self.diet_ontology)

    def parse_input(self, user_text: str) -> Dict[str, Any]:
        """
        Extracts nutritional constraints, allergens, dietary preferences, and ingredients from user input.

        Args:
            user_text (str): The raw text input from the user.

        Returns:
            Dict[str, Any]: Extracted features, e.g.:
                {
                    "nutrition": {"total_fat": (min, max), "sugar": (min, max), ...},
                    "allergen": ["peanut", "egg"],
                    "diet": ["vegetarian"],
                    "ingredients": ["chicken", "garlic"]
                }
        """
        user_text = user_text.lower()

        nutrition_constraints = self._extract_nutrition_constraints(user_text)
        detected_allergens = self._detect_categories(user_text, self.allergen_synonyms)
        detected_diets = self._detect_categories(user_text, self.diet_synonyms)
        ingredients = self._extract_ingredients(
            user_text, detected_allergens, detected_diets, nutrition_constraints
        )

        return {
            "nutrition": nutrition_constraints,
            "allergen": detected_allergens,
            "diet": detected_diets,
            "ingredients": ingredients,
        }

    def _load_ontology(self, path: str) -> Dict[str, List[str]]:
        """
        Loads an ontology from a JSON file.

        Expected JSON format:
            {
                "category_name": ["synonym1", "synonym2", ...],
                ...
            }

        Returns:
            Dict[str, List[str]]: Loaded ontology.
        """
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _flatten_ontology(self, ontology: Dict[str, List[str]]) -> Dict[str, str]:
        """
        Converts an ontology into a flat dictionary for fast lookup.

        Returns:
            Dict[str, str]: {synonym: category}
        """
        return {
            syn.lower(): category
            for category, synonyms in ontology.items()
            for syn in synonyms
        }

    def _extract_nutrition_constraints(
        self, text: str
    ) -> Dict[str, Tuple[float, float]]:
        """
        Extracts numeric nutritional constraints from text.

        For 'calories', we use a ±10% range; for all other fields, a ±50% range.

        Returns:
            Dict[str, Tuple[float, float]]: A mapping from each nutritional attribute to its (min, max) range.
        """
        constraints = {}
        for keyword, attr in self.NUTRITION_KEYWORDS.items():
            match = re.search(rf"(\d+)\s*{keyword}", text)
            if match:
                value = float(match.group(1))
                if attr == "calories":
                    min_val, max_val = value * 0.9, value * 1.1
                else:
                    min_val, max_val = value * 0.5, value * 1.5
                constraints[attr] = (min_val, max_val)
        return constraints

    def _detect_categories(self, text: str, category_dict: Dict[str, str]) -> List[str]:
        """
        Detects allergens or diets from text using the provided ontology mapping.

        Returns:
            List[str]: List of detected categories.
        """
        detected = set()
        for word in text.split():
            if word in category_dict:
                detected.add(category_dict[word])
        return list(detected)

    def _extract_ingredients(
        self,
        text: str,
        allergens: List[str],
        diets: List[str],
        nutrition: Dict[str, Any],
    ) -> List[str]:
        """
        Extracts ingredients while excluding words that are nutrition keywords, allergens, or diet terms.

        Returns:
            List[str]: Extracted ingredient names.
        """
        words = re.findall(r"\b[a-zA-Z]+\b", text)
        excluded_words = (
            set(self.NUTRITION_KEYWORDS.keys()) | set(allergens) | set(diets)
        )
        return [word for word in words if word not in excluded_words]

    def extract_faiss_features(self, user_text: str) -> Dict[str, Any]:
        """
        Processes the user input and returns a dictionary that meets the FAISS model’s type contract.

        The returned dictionary has the following keys:
            - nutrition: list of ingredients (to be used as the query vector)
            - allergen: list of ingredients to filter out recipes
            - calories, total_fat, sugar, sodium, protein, saturated_fat, carbs:
                tuples (min, max) representing nutritional constraints.
                If a constraint is not specified in the input, a default range of (0, 10000) is used.

        Args:
            user_text (str): Raw text input from the user.

        Returns:
            Dict[str, Any]: Dictionary with keys and types per the FAISS model contract.
        """
        parsed = self.parse_input(user_text)
        nutrition_constraints = parsed.get("nutrition", {})

        # For each nutritional field expected by the FAISS model, provide the parsed range or a default.
        def get_range(key: str) -> Tuple[float, float]:
            return nutrition_constraints.get(key, (0, 10000))

        features = {
            "nutrition": parsed.get(
                "ingredients", []
            ),  # List of ingredients for one-hot encoding.
            "allergen": parsed.get("allergen", []),
            "calories": get_range("calories"),
            "total_fat": get_range("total_fat"),
            "sugar": get_range("sugar"),
            "sodium": get_range("sodium"),
            "protein": get_range("protein"),
            "saturated_fat": get_range("saturated_fat"),
            "carbs": get_range("carbs"),
        }
        return features


# -------------------------------
# Example Usage of the NLU Engine
# -------------------------------
if __name__ == "__main__":
    # Paths to your ontology files (ensure these exist and follow the expected JSON format)
    allergen_path = "assets/allergens.json"
    diet_path = "assets/diets.json"

    nls_parser = NLSParser(allergen_path, diet_path)

    # Example user query
    user_query = (
        "I want a high-protein vegan meal with tofu and spinach, "
        "but no peanuts or dairy. Max 500 calories."
    )

    # Extract features that satisfy the FAISS model contract.
    faiss_features = nls_parser.extract_faiss_features(user_query)
    print("Extracted NLU Features:")
    for key, value in faiss_features.items():
        print(f"{key}: {value}")

    # Example integration with the FAISS model:

    # # Load your recipes dataframe (example with simple_recipes.csv)
    # simple_recipes = pd.read_csv("data/simple_recipes.csv")
    # simple_recipes["ingredient_names"] = simple_recipes["ingredient_names"].apply(
    #     ast.literal_eval
    # )

    # model = faiss_model(simple_recipes)
    # recommendations = model.recommend_recipes(
    #     user_ingredients=faiss_features["nutrition"],
    #     allergens=faiss_features["allergen"],
    #     calories=faiss_features["calories"],
    #     total_fat=faiss_features["total_fat"],
    #     sugar=faiss_features["sugar"],
    #     sodium=faiss_features["sodium"],
    #     protein=faiss_features["protein"],
    #     saturated_fat=faiss_features["saturated_fat"],
    #     carbs=faiss_features["carbs"],
    #     top_n=5,
    # )

    # print("Recommended Recipes:")
    # for rec in recommendations:
    #     print(rec)

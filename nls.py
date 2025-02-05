"""Module for natural language parsing of nutritional constraints, allergens, diets,
and ingredients from user input.
"""

import re
import json
import ast
from typing import Dict, Tuple, List, Any

import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util
from fuzzywuzzy import fuzz


class NLSParser:
    """Natural Language Synthesis (NLS) module for extracting parameters from user input.

    This class extracts nutritional constraints (e.g. calories, fat, sugar), allergens,
    requested ingredients, and dietary preferences from raw text.
    """

    # Map keywords found in text to our nutritional attributes.

    NUTRITION_KEYWORDS = {
        "calories": "calories",
        "fat": "total_fat",
        "sugar": "sugar",
        "sodium": "sodium",
        "protein": "protein",
        "saturated fat": "saturated_fat",
        "carbs": "carbs",
    }

    RECOMMENDED_DV = json.load(open("assets/nutrition_vals.json"))

    def __init__(
        self,
        allergen_path: str,
        diet_path: str,
        ingr_path: str,
        ingr_similarity_threshold: float,
        ingr_model_name: str,
    ):
        """Initializes the parser by loading allergen and diet ontologies and ingredient data.

        Args:
            allergen_path: Path to allergens JSON ontology.
            diet_path: Path to diets JSON ontology.
            ingr_path: Path to the CSV file containing ingredients.
            ingr_similarity_threshold: Minimum cosine similarity to consider a match.
            ingr_model_name: Name of the pretrained SentenceTransformer model.
        """
        self.allergen_ontology = self._load_ontology(allergen_path)
        self.diet_ontology = self._load_ontology(diet_path)
        self.ingr_df = pd.read_csv(ingr_path)

        # Normalize and prepare ingredient names and embeddings.
        self.ingr_df["ingredient_names"] = (
            self.ingr_df["ingredient_names"].str.strip().str.lower()
        )
        self.ingredients = self.ingr_df["ingredient_names"].tolist()
        self.ingr_ids = self.ingr_df["ingredient_ids"].tolist()
        self.ingr_similarity_threshold = ingr_similarity_threshold
        self.ingr_rec_model = SentenceTransformer(ingr_model_name)
        self.ingr_embeddings = self.ingr_rec_model.encode(
            self.ingredients, convert_to_tensor=True
        )

        # Flatten synonyms for quick lookup.
        self.allergen_synonyms = self._flatten_ontology(self.allergen_ontology)
        self.diet_synonyms = self._flatten_ontology(self.diet_ontology)

    def parse_input(self, user_text: str) -> Dict[str, Any]:
        """Extracts nutritional constraints, allergens, dietary preferences, and ingredients.

        Args:
            user_text: Raw text input from the user.

        Returns:
            A dictionary with keys:
              - nutrition: dict mapping nutritional attributes to (min, max) ranges.
              - allergen: list of detected allergen categories.
              - diet: list of detected dietary preferences.
              - ingredients: list of recognized ingredient names (excluding allergens).
        """
        # Convert text to lowercase for consistent matching.
        user_text = user_text.lower()

        nutrition_constraints = self._extract_nutrition_constraints(user_text)
        detected_allergens = self._detect_allergens(user_text, self.allergen_synonyms)
        # Recognize ingredients and immediately filter out any that match a detected allergen.
        recognized_ingredients = self._recognize_ingr(user_text, detected_allergens)
        detected_diets = self._detect_categories(user_text, self.diet_synonyms)

        return {
            "nutrition": nutrition_constraints,
            "allergen": detected_allergens,
            "diet": detected_diets,
            "ingredients": recognized_ingredients,
        }

    def _detect_allergens(self, text: str, allergen_dict: Dict[str, str]) -> List[str]:
        """Detects allergens only if an allergen trigger word is present in the sentence.

        The text is split into sentences, and a sentence is considered if it contains a
        trigger word starting with 'allerg' (e.g., "allergic", "allergy"). Within such a
        sentence, the method searches for any allergen synonyms from the ontology.

        Args:
            text: Input text.
            allergen_dict: Dictionary mapping allergen synonyms to canonical allergen names.

        Returns:
            A list of detected allergen categories.
        """
        detected_allergens = set()
        # Split text into sentences using punctuation as delimiters.
        sentences = re.split(r"[.!?]+", text)
        allerg_trigger = re.compile(r"\ballerg\w*\b")
        for sentence in sentences:
            if allerg_trigger.search(sentence):
                for synonym, category in allergen_dict.items():
                    pattern = r"\b" + re.escape(synonym) + r"\b"
                    if re.search(pattern, sentence):
                        detected_allergens.add(category)
        return list(detected_allergens)

    def _matches_allergen_fuzzy(
        self, allergen: str, ingredient: str, threshold: int = 80
    ) -> bool:
        """Performs a fuzzy match between an ingredient and each synonym for a given allergen.

        Args:
            allergen: The canonical allergen category (e.g. "egg").
            ingredient: The recognized ingredient name.
            threshold: The fuzzy matching score threshold (default is 80).

        Returns:
            True if the fuzzy matching score for any synonym meets or exceeds the threshold;
            otherwise False.
        """
        synonyms = self.allergen_ontology.get(allergen, [])
        for synonym in synonyms:
            # Use token_set_ratio for a robust fuzzy match between the ingredient and allergen synonym.
            score = fuzz.token_set_ratio(ingredient, synonym)
            if score >= threshold:
                return True
        return False

    def _detect_categories(self, text: str, category_dict: Dict[str, str]) -> List[str]:
        """Detects categories (such as dietary preferences) using the provided ontology mapping.

        Args:
            text: Input text.
            category_dict: Dictionary mapping synonyms to canonical categories.

        Returns:
            A list of detected categories.
        """
        detected = set()
        for word in text.split():
            if word in category_dict:
                detected.add(category_dict[word])
        return list(detected)

    def _recognize_ingr(
        self, user_text: str, detected_allergens: List[str]
    ) -> List[str]:
        """Recognizes ingredient names in the input text based on semantic similarity,
        and immediately excludes any candidate that fuzzily matches any synonym for a
        detected allergen.

        Candidate phrases (both unigrams and simple bigrams) are compared against the
        canonical ingredient names using SentenceTransformer embeddings.

        Args:
            user_text: Raw text input from the user.
            detected_allergens: List of allergen categories detected from the text.

        Returns:
            A list of recognized ingredient names that do not fuzzy-match any detected allergen.
        """
        tokens = user_text.split()
        candidates = set(tokens)

        # Add simple bi-grams to capture multi-word ingredients.
        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                candidates.add(tokens[i] + " " + tokens[i + 1])

        recognized = set()
        for candidate in candidates:
            candidate_embedding = self.ingr_rec_model.encode(
                candidate, convert_to_tensor=True
            )
            cosine_scores = util.cos_sim(candidate_embedding, self.ingr_embeddings)[0]
            max_score, max_idx = torch.max(cosine_scores, dim=0)
            if max_score.item() >= self.ingr_similarity_threshold:
                recognized_candidate = self.ingredients[max_idx]
                # Exclude the recognized candidate if it fuzzily matches any synonym for any detected allergen.
                if not any(
                    self._matches_allergen_fuzzy(allergen, recognized_candidate)
                    for allergen in detected_allergens
                ):
                    recognized.add(recognized_candidate)
        return list(recognized)

    def _load_ontology(self, path: str) -> Dict[str, List[str]]:
        """Loads an ontology from a JSON file.

        The expected JSON format is:
            {
                "category_name": ["synonym1", "synonym2", ...],
                ...
            }

        Args:
            path: Path to the JSON file.

        Returns:
            The loaded ontology dictionary.
        """
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)

    def _flatten_ontology(self, ontology: Dict[str, List[str]]) -> Dict[str, str]:
        """Flattens an ontology dictionary for fast lookup.

        Args:
            ontology: Ontology dictionary mapping categories to a list of synonyms.

        Returns:
            A dictionary mapping each synonym to its canonical category.
        """
        return {
            syn.lower(): category
            for category, synonyms in ontology.items()
            for syn in synonyms
        }

    def _extract_nutrition_constraints(
        self, text: str
    ) -> Dict[str, Tuple[float, float]]:
        """Extracts numeric nutritional constraints from the input text.

        For 'calories', a ±10% range is used; for all other attributes, a ±50% range is applied.

        Args:
            text: Input text.

        Returns:
            A dictionary mapping each nutritional attribute to its (min, max) range.
        """
        constraints = {}
        for keyword, attr in self.NUTRITION_KEYWORDS.items():
            match = re.search(rf"(\d+)\s*{re.escape(keyword)}\b", text)
            if match:
                value = float(match.group(1))
                if attr == "calories":
                    min_val, max_val = value * 0.9, value * 1.1
                else:
                    min_val, max_val = value * 0.5, value * 1.5
                constraints[attr] = (min_val, max_val)
        return constraints

    def extract_faiss_features(self, user_text: str) -> Dict[str, Any]:
        """Processes user input and returns a dictionary matching the FAISS model contract.

        The returned dictionary contains the following keys:
          - nutrition: List of recognized ingredients (to be used as the query vector).
          - allergen: List of allergens to filter out recipes.
          - calories, total_fat, sugar, sodium, protein, saturated_fat, carbs:
              Nutritional constraints as (min, max) tuples.
              If a constraint is not specified, a default range of (0, 10000) is used.

        Args:
            user_text: Raw text input from the user.

        Returns:
            A dictionary with keys and value types as expected by the FAISS model.
        """
        parsed = self.parse_input(user_text)
        nutrition_constraints = parsed.get("nutrition", {})

        def get_range(key: str) -> Tuple[float, float]:
            return nutrition_constraints.get(key, (0, 10000))

        features = {
            "nutrition": parsed.get("ingredients", []),
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
    # Paths to ontology and ingredient files (ensure these exist and follow the expected format)
    allergen_path = "assets/allergens.json"
    diet_path = "assets/diets.json"
    ingr_path = "assets/ingredient_map.csv"

    # Example user input.
    user_input = (
        "I love peanuts and chicken, but I'm allergic to eggs. "
        "Also, add some garlic to my meal."
    )

    nls_parser = NLSParser(
        allergen_path,
        diet_path,
        ingr_path,
        ingr_similarity_threshold=0.8,
        ingr_model_name="paraphrase-MiniLM-L6-v2",
    )

    # Extract features conforming to the FAISS model contract.
    faiss_features = nls_parser.extract_faiss_features(user_input)
    print("Extracted NLU Features:")
    for key, value in faiss_features.items():
        print(f"{key}: {value}")

    # Example integration with the FAISS model (commented out):
    #
    # simple_recipes = pd.read_csv("data/simple_recipes.csv")
    # simple_recipes["ingredient_names"] = simple_recipes["ingredient_names"].apply(ast.literal_eval)
    #
    # from models.dev.faiss_rec import faiss_model
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
    #
    # print("Recommended Recipes:")
    # for rec in recommendations:
    #     print(rec)

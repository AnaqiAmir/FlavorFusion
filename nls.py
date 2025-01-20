"""
Natural Language Synthesis (NLS) module for FlavorFusion.
Extracts user constraints (allergies, preferences, etc.) from input text.
"""

import re
from typing import Dict, Tuple, List, Any, Set
import spacy
from transformers import pipeline
import json
from rapidfuzz import fuzz, process


class NaturalLanguageSynthesis:
    """
    A class responsible for parsing user input text and extracting
    key features such as preferences, allergens, nutritional needs, etc.
    """

    def __init__(
        self,
        allergen_ontology_path: str,
        diet_ontology_path: str,
        intent_model_name: str = "bert-base-uncased",
        ner_model: str = "en_core_web_sm",
        fuzz_threshold: int = 95,
    ):
        """
        Initialize the NLS module with chosen models for intent classification
        and named entity recognition.

        :param intent_model_name: Hugging Face model name for text classification
        :param ner_model: spaCy model name for NER
        :param allergen_ontology_path: Path to your JSON file containing allergen categories and synonyms
        :param diet_ontology_path: Path to your JSON file containing diets and synonyms
        :param fuzz_threshold: Minimum similarity score (0-100) for fuzzy matches to count
        """
        # Load spaCy model for NER
        self.nlp = spacy.load(ner_model)

        # Set the minimum similarity score for fuzzy matching
        self.fuzz_threshold = fuzz_threshold

        # Load ontologies
        self.allergen_ontology = self._load_ontology(allergen_ontology_path)
        self.diet_ontology = self._load_ontology(diet_ontology_path)

        # Flatten the ontology for quick reference
        # Example: [("nuts", "peanut"), ("nuts", "almond"), ...]
        self.allergen_synonyms_list = self._flatten_ontology(self.allergen_ontology)
        self.diet_synonyms_list = self._flatten_ontology(self.diet_ontology)

        # Load a simple text classification pipeline from Hugging Face
        self.intent_pipeline = pipeline("text-classification", model=intent_model_name)

    def parse_user_input(self, text: str) -> Dict[str, Any]:
        """
        Main entry point to parse user text input and extract relevant features.

        :param text: The user input text to analyze
        :return: A dictionary with extracted features (intent, allergens, diets, etc.)
        """
        cleaned_text = self._clean_text(text)

        # 1. Identify intent (e.g., "request_recipe", "ask_nutrition", "greeting", etc.)
        intent = self._classify_intent(cleaned_text)

        # 2. Extract detailed entities using spaCy or a custom approach
        recognized_entities = self._extract_entities(cleaned_text)

        # 3. Rule-based detection of allergens/diet keywords
        constraints = self._detect_constraints(cleaned_text)

        # 4. Construct a final structured representation
        parsed_output = {
            "original_text": text,
            "cleaned_text": cleaned_text,
            "intent": intent,
            "allergens": constraints["allergens"],
            "diets": constraints["diets"],
            "entities": recognized_entities,
        }

        return parsed_output

    def _load_ontology(self, path: str) -> Dict[str, List[str]]:
        """
        Load the allergen ontology from a JSON file.

        Expected format:
            {
              "nuts": ["peanut", "almond", ...],
              "dairy": ["milk", "cheese", ...],
              ...
            }
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data

    def _flatten_ontology(
        self, ontology: Dict[str, List[str]]
    ) -> List[Tuple[str, str]]:
        """
        Turn a dict {category: [syn1, syn2, ...]} into a list of (category, synonym).
        """
        pairs = []
        for category, synonyms in ontology.items():
            for syn in synonyms:
                pairs.append((category, syn))
        return pairs

    def _clean_text(self, text: str) -> str:
        """
        Basic text cleaning and normalization.

        :param text: The raw input text
        :return: A lowercased, cleaned version of the text
        """
        text = text.lower()
        # Remove unwanted characters (keep alpha-numerics and some punctuation)
        text = re.sub(r"[^a-zA-Z0-9\s,.?!]", "", text)
        return text.strip()

    def _classify_intent(self, text: str) -> str:
        """
        Use a text classification model to determine the user's intent.

        :param text: Cleaned user input text
        :return: String label of the predicted intent
        """
        # For demonstration, this might return labels like "LABEL_0", "LABEL_1"
        # which you'd map to domain-specific classes.
        # In practice, you’d fine-tune a model or have a pre-trained domain-specific model.
        predictions = self.intent_pipeline(text)
        label = predictions[0]["label"]  # e.g. "LABEL_0"

        # Simple example label mapping
        label_map = {
            "LABEL_0": "request_recipe",
            "LABEL_1": "ask_nutrition",
            "LABEL_2": "provide_feedback",
            "LABEL_3": "other",
        }
        return label_map.get(label, "other")

    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Run spaCy NER to extract recognized entities (ORG, PERSON, etc.) from text.

        :param text: Cleaned user input
        :return: A list of dictionaries for each entity found
        """
        doc = self.nlp(text)
        entities = []
        for ent in doc.ents:
            entities.append({"text": ent.text, "label": ent.label_})
        return entities

    def _detect_constraints(self, text: str) -> Dict[str, List[str]]:
        """
        Detect both allergens and diets in the given text.

        :param text: Raw user input
        :return: {"allergens": [...], "diets": [...]}
        """
        # Preprocess
        doc = self.nlp(text.lower().strip())

        # Generate n-grams (up to 3-grams by default)
        all_token_strings = self._generate_ngrams(doc, n=3)

        found_allergens = self._detect_categories(
            all_token_strings, self.allergen_synonyms_list
        )
        found_diets = self._detect_categories(
            all_token_strings, self.diet_synonyms_list
        )

        return {
            "allergens": sorted(list(found_allergens)),
            "diets": sorted(list(found_diets)),
        }

    def _generate_ngrams(self, doc, n=3, min_token_len=3) -> List[str]:
        """
        Generate 1- to n-grams from the spaCy doc tokens.
        E.g., If doc has tokens ["peanut", "butter", "is", "great"],
        1-grams = ["peanut", "butter", "is", "great"]
        2-grams = ["peanut butter", "butter is", "is great"]
        3-grams = ["peanut butter is", "butter is great"]
        """
        tokens = [t.text for t in doc if len(t.text) >= min_token_len and not t.is_stop]
        results = []
        max_n = min(len(tokens), n)

        for size in range(1, max_n + 1):
            for start_idx in range(len(tokens) - size + 1):
                ngram = tokens[start_idx : start_idx + size]
                results.append(" ".join(ngram))
        return results

    def _detect_categories(
        self, token_strings: List[str], synonyms_list: List[Tuple[str, str]]
    ) -> Set[str]:
        """
        Given a list of token-based n-grams and an ontology (list of (category, synonym)),
        fuzzy match each token_string to find categories with score >= fuzz_threshold.
        Returns a set of category names found.
        """
        found_categories = set()

        # The second element in synonyms_list is the synonym string
        possible_synonyms = [syn for (_, syn) in synonyms_list]

        for t_str in token_strings:
            # Attempt fuzzy matching
            matches = process.extract(t_str, possible_synonyms, scorer=fuzz.ratio)
            if not matches:
                continue

            best_match_str, best_score, _best_idx = matches[0]
            if best_score >= self.fuzz_threshold:
                if len(best_match_str) < 3:
                    # Skip matches to synonyms that are too short
                    continue
                # Identify which category this matched_str belongs to
                cat = self._get_category_for_synonym(best_match_str, synonyms_list)
                if cat:
                    found_categories.add(cat)
        return found_categories

    def _get_category_for_synonym(
        self, matched_synonym: str, synonyms_list: List[Tuple[str, str]]
    ) -> str:
        """
        Given a matched synonym string, find its corresponding category
        by looking it up in synonyms_list.
        """
        for category, syn in synonyms_list:
            if syn == matched_synonym:
                return category
        return None


# --------------------
# Example Usage
# --------------------
if __name__ == "__main__":
    nls = NaturalLanguageSynthesis(
        allergen_ontology_path="assets/allergens.json",
        diet_ontology_path="assets/diets.json",
    )

    user_input = "I'm looking for a veg recipe, but I have a nut and egg allergy and I'd like it to be high protein."
    result = nls.parse_user_input(user_input)
    print("Parsed Output:")
    print(result)

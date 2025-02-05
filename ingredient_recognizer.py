import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util


class IngredientRecognizer:
    def __init__(
        self,
        csv_path: str,
        model_name: str = "paraphrase-MiniLM-L6-v2",
        similarity_threshold: float = 0.8,
    ):
        """
        Initializes the ingredient recognizer by:
          - Loading the ingredient CSV
          - Creating a list of canonical ingredient names
          - Precomputing embeddings for every ingredient in the CSV

        Args:
            csv_path (str): Path to the CSV file containing ingredients.
            model_name (str): Name of the pretrained SentenceTransformer model.
            similarity_threshold (float): Minimum cosine similarity to consider a match.
        """
        self.df = pd.read_csv(csv_path)
        # Ensure consistent formatting by stripping extra spaces and lowering text.
        self.df["ingredient_names"] = (
            self.df["ingredient_names"].str.strip().str.lower()
        )
        self.ingredients = self.df["ingredient_names"].tolist()
        self.ingredient_ids = self.df["ingredient_ids"].tolist()

        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(model_name)
        # Precompute embeddings for all ingredients in the CSV.
        self.ingredient_embeddings = self.model.encode(
            self.ingredients, convert_to_tensor=True
        )

    def recognize(self, user_text: str) -> list:
        """
        Recognizes and returns a list of ingredient names (from the CSV) that are closely
        related to any words or phrases in the input text.

        Args:
            user_text (str): Raw text input from the user.

        Returns:
            list: Canonical ingredient names (from the CSV) that match the input.
        """
        # In a production system, you might want a more advanced tokenizer or even a
        # fine-tuned NER model to extract candidate ingredient mentions.
        # Here, we simply split the text into words and also consider some n-grams.
        tokens = user_text.lower().split()
        candidates = set(tokens)

        # Optionally, you could add simple bi-grams to capture multi-word ingredients.
        tokens_list = user_text.lower().split()
        if len(tokens_list) >= 2:
            for i in range(len(tokens_list) - 1):
                candidates.add(tokens_list[i] + " " + tokens_list[i + 1])

        recognized = set()
        for candidate in candidates:
            # Compute embedding for the candidate phrase.
            candidate_embedding = self.model.encode(candidate, convert_to_tensor=True)
            # Compute cosine similarities with all ingredient embeddings.
            cosine_scores = util.cos_sim(
                candidate_embedding, self.ingredient_embeddings
            )[0]
            # Find the best matching ingredient.
            max_score, max_idx = torch.max(cosine_scores, dim=0)
            if max_score.item() >= self.similarity_threshold:
                recognized.add(self.ingredients[max_idx])
        return list(recognized)


# ------------------------------
# Example Usage in the Pipeline
# ------------------------------

if __name__ == "__main__":
    # Assume you have a CSV file named "ingredients.csv" with columns "ingredient_names" and "ingredient_ids".
    csv_path = "assets/ingredient_map.csv"
    recognizer = IngredientRecognizer(csv_path)

    # Example user input.
    user_input = (
        "I'm craving a dish with creamy cheddar and maybe some mozzarella, "
        "but I also love a good pasta sauce. Perhaps some extra shredded three cheese?"
    )

    # Get the recognized ingredients.
    normalized_ingredients = recognizer.recognize(user_input)
    print("Normalized Ingredients:", normalized_ingredients)

    # Now you can pass these normalized ingredients to your FAISS recommendation system.
    # For example:
    #
    # from faiss_rec import faiss_model
    # import ast
    # import pandas as pd
    #
    # # Load your recipes DataFrame.
    # recipes_df = pd.read_csv("data/simple_recipes.csv")
    # recipes_df['ingredient_names'] = recipes_df['ingredient_names'].apply(ast.literal_eval)
    #
    # model = faiss_model(recipes_df)
    # recommendations = model.recommend_recipes(
    #     user_ingredients=normalized_ingredients,
    #     allergens=[],          # You can also process allergens similarly.
    #     calories=None,
    #     total_fat=None,
    #     sugar=None,
    #     sodium=None,
    #     protein=None,
    #     saturated_fat=None,
    #     carbs=None,
    #     top_n=5
    # )
    #
    # print("Recommended Recipes:")
    # for rec in recommendations:
    #     print(rec)

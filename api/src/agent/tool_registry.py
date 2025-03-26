import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from api.src.nlp.NLSPipeline import extract_nutritional_features
from api.src.models.dev.faiss_indexes import FlatIndex

# Load env variables
load_dotenv()


# Create index instance
index = FlatIndex("recipe_embeddings_small.json")


class ToolRegistry:
    def __init__(self):
        self._tools = {}

    def register_tool(self, name: str, tool_func):
        """Register a tool function with a unique name."""
        self._tools[name] = tool_func

    def get_tool(self, name: str):
        """Retrieve a tool by its name."""
        return self._tools.get(name)

    def get_all_tools(self):
        """Return a list of all registered tools."""
        return list(self._tools.values())


# Global registry instance
tool_registry = ToolRegistry()


@tool
def recommend_recipes(user_input: str) -> str:
    """Extracts features from user input and outputs recommended recipes."""
    print(f"\nParsed user input:\n{user_input}\n")
    extracted_features = extract_nutritional_features(user_input)
    print(f"\nExtracted features:\n{extracted_features}\n")

    recs = index.recommend_recipes(
        user_ingredients=extracted_features.user_ingredients,
        allergens=extracted_features.allergens,
        calories=extracted_features.calories,
        total_fat=extracted_features.total_fat,
        protein=extracted_features.protein,
        saturated_fat=extracted_features.saturated_fat,
        carbs=extracted_features.carbs,
        sodium=extracted_features.sodium,
        sugar=extracted_features.sugar,
        top_n=10,
    )
    print(f"\nRecommendations:\n{recs}\n")
    return ", ".join(recs)


@tool
def final_answer(answer: str) -> str:
    """Format the LLM output to provide an appropriate answer to the user."""
    # TODO: Implement this function
    return answer


# Register tools
tool_registry.register_tool("recommend_recipes", recommend_recipes)
tool_registry.register_tool("final_answer", final_answer)

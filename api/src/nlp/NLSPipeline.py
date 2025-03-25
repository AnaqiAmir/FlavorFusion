import os
import json
from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser


# Define the output schema using a Pydantic model.
class NutritionFeatures(BaseModel):
    user_ingredients: List[str] = Field(default_factory=list)
    allergens: List[str] = Field(
        default_factory=list,
        description=(
            "List of allergens to avoid. Only include allergens from the following categories: "
            "tree nuts, peanut, milk, wheat, soy, fish, shellfish, eggs, sesame, pollen, sulfites."
        ),
    )
    calories: Optional[Tuple[Optional[float], Optional[float]]] = Field(
        default_factory=lambda: (None, None)
    )
    total_fat: Optional[Tuple[Optional[float], Optional[float]]] = Field(
        default_factory=lambda: (None, None)
    )
    saturated_fat: Optional[Tuple[Optional[float], Optional[float]]] = Field(
        default_factory=lambda: (None, None)
    )
    carbs: Optional[Tuple[Optional[float], Optional[float]]] = Field(
        default_factory=lambda: (None, None)
    )
    sugar: Optional[Tuple[Optional[float], Optional[float]]] = Field(
        default_factory=lambda: (None, None)
    )
    sodium: Optional[Tuple[Optional[float], Optional[float]]] = Field(
        default_factory=lambda: (None, None)
    )
    protein: Optional[Tuple[Optional[float], Optional[float]]] = Field(
        default_factory=lambda: (None, None)
    )


def extract_nutritional_features(user_input: str) -> NutritionFeatures:
    """
    Extract nutritional features from a user's natural language input.

    Args:
        user_input (str): The raw user input containing nutrition requirements,
                          allergens, liked ingredients, and disliked ingredients.

    Returns:
        NutritionFeatures: A structured object containing the extracted features.
    """
    # Initialize the output parser using the Pydantic model.
    output_parser = PydanticOutputParser(pydantic_object=NutritionFeatures)

    # Retrieve format instructions that tell the LLM how to output the JSON.
    format_instructions = output_parser.get_format_instructions()

    # Create a system message to set the context.
    system_message = SystemMessagePromptTemplate.from_template(
        "You are an assistant that extracts nutritional features from user input, "
        "including recognized allergens from the following categories: tree nuts, peanut, milk, "
        "wheat, soy, fish, shellfish, eggs, sesame, pollen, sulfites."
    )

    # Create a human message prompt template that includes extraction instructions,
    # formatting instructions, and explicit allergen categorization.
    human_template = (
        "Extract the following nutritional features from the user's input. "
        "For numerical fields (calories, total_fat, saturated_fat, carbs, sugar, sodium, protein), "
        "apply these rules:\n"
        "1. If an explicit numerical value is mentioned, return a pair [provided_value, max_possible_value] where:\n"
        "   - For phrases like 'around X', use [X, max_possible_value] for that field.\n"
        "   - For phrases like 'no more than Y', use [0.0, Y].\n"
        "2. If a qualitative adjective is mentioned without a specific number, do not infer a number; return [null, null].\n"
        "3. Always return the numerical field as a complete pair. For example, [null, X] or [X, null] is not acceptable.\n\n"
        "The maximum possible values are as follows:\n"
        "   calories: 434360.2, total_fat: 3464.8, saturated_fat: 1375.0, protein: 3276.0, "
        "carbs: 99269.5, sugar: 181364.5, sodium: 337271.99999999994.\n\n"
        "For list fields, return a JSON array of strings:\n"
        "- For user_ingredients, extract only the ingredients the user explicitly states they love or prefer.\n"
        "- For allergens, extract any ingredient to avoid and map them only to the following allowed allergen categories: "
        "tree nuts, peanut, milk, wheat, soy, fish, shellfish, eggs, sesame, pollen, sulfites. "
        "For example, map 'macadamia nuts' to 'tree nuts' and 'whey protein' to 'milk'.\n\n"
        "Ensure that the fields are either filled out with min and max values (e.g. calories=[400,500]) or null values. Ranges where one of the two values is null is invalid (e.g. protein=[null, 3276.0] is not acceptable, if it occurs replace the null value with the min or max value of the nutritional field)."
        "{format_instructions}\n\n"
        "User input: {user_input}"
    )
    human_message = HumanMessagePromptTemplate.from_template(human_template)

    # Build the full chat prompt.
    chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # Initialize the ChatOpenAI model with a low temperature for deterministic output.
    llm = ChatOpenAI(temperature=0.1, api_key=os.environ.get("OPENAI_API_KEY"))

    # Create the LLM chain with the prompt.
    chain = LLMChain(llm=llm, prompt=chat_prompt)

    # Run the chain by providing the user input and format instructions.
    response = chain.invoke(
        {"user_input": user_input, "format_instructions": format_instructions}
    )

    # Parse the LLM's response using our structured output parser.
    features = output_parser.parse(response["text"])
    return features


# Example usage
if __name__ == "__main__":
    sample_input = (
        "I need a meal with around 500 calories, low fat (no more than 10g total fat), "
        "high protein, and please avoid macadamia nuts and whey protein. I love tomatoes and basil, "
        "but I don't like onions."
    )
    extracted_features = extract_nutritional_features(sample_input)
    print("Extracted Features:")
    # Pretty print the model output using json.dumps on model_dump() (Pydantic v2 compliant).
    print(json.dumps(extracted_features.model_dump(), indent=2))

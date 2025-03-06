import os
import json
from typing import List, Optional
from pydantic import BaseModel, Field
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser


# Define the output schema using a Pydantic model.
class NutritionFeatures(BaseModel):
    calories: Optional[float] = Field(None, description="Calories in kcal")
    total_fat: Optional[float] = Field(None, description="Total fat in grams")
    saturated_fat: Optional[float] = Field(None, description="Saturated fat in grams")
    carbs: Optional[float] = Field(None, description="Carbohydrates in grams")
    sugar: Optional[float] = Field(None, description="Sugar in grams")
    sodium: Optional[float] = Field(None, description="Sodium in mg")
    protein: Optional[float] = Field(None, description="Protein in grams")
    allergens: List[str] = Field(
        default_factory=list,
        description=(
            "List of allergens to avoid. Only include allergens from the following categories: "
            "tree nuts, peanut, milk, wheat, soy, fish, shellfish, eggs, sesame, pollen, sulfites."
        ),
    )
    liked_ingredients: List[str] = Field(
        default_factory=list, description="List of ingredients the user likes"
    )
    disliked_ingredients: List[str] = Field(
        default_factory=list, description="List of ingredients the user dislikes"
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
        "For numerical fields, return a number (or null if unspecified). "
        "For list fields, return a JSON array of strings. "
        "Ensure that allergens are only taken from the following categories: "
        "tree nuts, peanut, milk, wheat, soy, fish, shellfish, eggs, sesame, pollen, sulfites. "
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
    response = chain.run(
        {"user_input": user_input, "format_instructions": format_instructions}
    )

    # Parse the LLM's response using our structured output parser.
    features = output_parser.parse(response)
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

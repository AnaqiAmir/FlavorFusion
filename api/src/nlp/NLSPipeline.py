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
        "For numerical fields (calories, total_fat, saturated_dat, carbs, sugar, sodium, protein), return a number (or null if unspecified)."
        "If no specific numbers are provided for numerical fields, take liberty in filling out what is appropriate numerically."
        "Ensure that the fields are either filled out as a pair with a range (e.g. calories=[400,500]) or none at all (e.g. sugar=[null, 10.0] is unacceptable)."
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

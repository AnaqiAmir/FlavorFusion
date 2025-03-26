import os
import sys

# Set the root project directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
sys.path.append(project_root)

from langchain_core.tools import tool
from langchain_openai.chat_models import ChatOpenAI

from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver

from api.src.nlp.NLSPipeline import extract_nutritional_features
from api.src.models.dev.faiss_indexes import FlatIndex

from dotenv import load_dotenv


# Load env
load_dotenv()

# Load index
index = FlatIndex("../../../recipe_embeddings_small.json")


# Define tools
@tool
def recommend_recipes(user_input: str) -> str:
    """Takes in user input, extracts relevant features, and output recommended recipes from database"""

    print(f"\n Parsed user input: \n {user_input} \n")

    extracted_features = extract_nutritional_features(user_input)

    print(f"\n Extracted features: \n {extracted_features} \n")

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

    print(f"\n recs: \n {recs} \n")

    return ", ".join(recs)


@tool
def final_answer(answer: str) -> str:
    """Format the LLM output to provide an appropriate answer to the user."""
    # TODO: Implement this function
    return answer


# Define tool node
tools = [recommend_recipes, final_answer]
tool_node = ToolNode(tools)  # A single node that contains all the tools


# Define llm
llm = ChatOpenAI().bind_tools(tools)


# Define nodes
def router(state: MessagesState) -> str:
    """Routes to which node invoke in the graph"""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return "end"


def agent(state: MessagesState) -> MessagesState:
    """
    Interacts with the user and decides what action to take next (e.g. call tools, end workflow)
    """
    messages = state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}


# Define graph
workflow = StateGraph(MessagesState)

workflow.add_node("agent", agent)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", router, {"tools": "tools", "end": END})
workflow.add_edge("tools", "agent")

# Define memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# IO
def stream_graph_updates(user_input: str) -> None:
    """I/O for user to interact with the program"""
    for event in app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
    ):
        entity = [list(event)[0]][0].upper()  # either agent or tool
        print(f"##### {entity} MESSAGE #####")
        print(event[list(event)[0]]["messages"][-1].content)
        print("\n")


while True:
    try:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        print("##### HUMAN MESSAGE #####")
        print(user_input)
        print("\n")

        stream_graph_updates(user_input)
    except:
        # fallback if input() is not available
        user_input = "What do you know about LangGraph?"
        print("User: " + user_input)
        stream_graph_updates(user_input)
        break

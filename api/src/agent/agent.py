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

from api.src.agent.tool_registry import tool_registry

from dotenv import load_dotenv

# Load env
load_dotenv()

# Define tool node
tools = tool_registry.get_all_tools()
tool_node = ToolNode(tools)

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


# Expose this to be used in Flask
def stream_graph_updates(user_input: str) -> str:
    """I/O for user to interact with the program"""
    final_response = None
    for event in app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
    ):
        node = list(event)[0]
        messages = event[node]["messages"]
        final_response = messages[-1].content
    return final_response

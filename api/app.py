from flask import Flask, request, jsonify
from flask_cors import CORS
from langgraph.graph import MessagesState
from dotenv import load_dotenv

load_dotenv()

app_api = Flask(__name__)
CORS(app_api, origins=["http://localhost:3000"], supports_credentials=True)


@app_api.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    # Run LangGraph with input
    final_response = None
    for event in agent.app.stream(
        {"messages": [{"role": "user", "content": user_input}]},
        config={"configurable": {"thread_id": "1"}},
    ):
        node = list(event)[0]
        messages = event[node]["messages"]
        final_response = messages[-1]["content"]

    return jsonify({"response": final_response})

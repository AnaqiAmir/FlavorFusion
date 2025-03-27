from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from src.agent.agent import stream_graph_updates

load_dotenv()

app_api = Flask(__name__)
CORS(app_api, origins=["http://localhost:3000"], supports_credentials=True)

@app_api.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_input = data.get("message")

    final_response = stream_graph_updates(user_input)

    return jsonify({"response": final_response})

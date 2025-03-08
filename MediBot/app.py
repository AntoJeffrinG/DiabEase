from flask import Flask, request, Response, stream_with_context
from flask_cors import CORS
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import gevent
from gevent.pywsgi import WSGIServer
import re

app = Flask(__name__)
CORS(app)  # Allows cross-origin requests

# Initialize the Ollama LLM model
model = OllamaLLM(model="llama2")

# Define the chat prompt template
template = """
Answer the question below in a structured format.
Use numbered points, bullet points, or paragraphs when necessary.

Here is the conversation history: {context}
Question: {question}
Answer:
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Store conversation history for each session
conversations = {}

@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    session_id = data.get("session_id", "default")  # Unique session ID
    user_input = data.get("question", "")

    if not user_input:
        return Response("Error: No question provided.", status=400)

    # If it's a new session, greet the user first
    if session_id not in conversations:
        initial_greeting = "Hi, I am DiabEase Medibot. How can I assist you today?"
        conversations[session_id] = initial_greeting
        return Response(stream_with_context(stream_response(initial_greeting)), content_type='text/plain')

    # Retrieve conversation history
    context = conversations.get(session_id, "")

    # Generate response asynchronously
    result = chain.invoke({"context": context, "question": user_input})

    # Limit the response to 50 characters, ensuring it's meaningful
    result = generate_meaningful_response(result)

    # Update conversation history
    conversations[session_id] = context + f"\nUser: {user_input}\nAI: {result}"

    return Response(stream_with_context(stream_response(result)), content_type='text/plain')

def generate_meaningful_response(response, max_length=500):
    """Ensure that the response is meaningful and within the character limit."""
    
    # First, trim the response to the max length
    truncated_response = response[:max_length]

    # Find the last punctuation mark (., ?, or !) within the truncated response
    last_punctuation_pos = max(
        truncated_response.rfind(punct) for punct in ['.', '!', '?']
    )
    
    # If a punctuation mark was found, truncate the response at that position
    if last_punctuation_pos != -1:
        truncated_response = truncated_response[:last_punctuation_pos + 1]
    else:
        # If no punctuation mark is found, just append ellipsis to indicate truncation
        truncated_response += '...'
    
    return truncated_response
def stream_response(text):
    """Generator function to stream text letter by letter."""
    # Split the text into sentences to avoid splitting in the middle
    sentences = re.split(r'(?<=[.!?])\s+', text)

    for sentence in sentences:
        for letter in sentence:
            yield letter
            gevent.sleep(0.05)  # Adjust sleep for letter-by-letter streaming
        yield '\n'  # Add newline after each sentence for clarity

if __name__ == "__main__":
    # Print success message and port
    port = 3000
    print(f"Server is running successfully on port {port}...")

    # Run the app with Gevent for asynchronous processing
    http_server = WSGIServer(('0.0.0.0', port), app)
    app.debug = True
    http_server.serve_forever()
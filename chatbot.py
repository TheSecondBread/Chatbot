from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_chroma import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from secret import OPENAI_API_KEY
import os

app = Flask(__name__)
CORS(app)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

llm = ChatOpenAI(
    model="gpt-4o-mini",
    openai_api_key=OPENAI_API_KEY
)

# Load the embeddings function
embed_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the existing Chroma
data=Chroma(
    persist_directory="./db",
    embedding_function=embed_function
)

retriever=data.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k":10,"score_threshold":0.0003}
)   

prompt = "You are a helpful AI assistant. Your only job is to provide users with the best answers. You will also be given relevant information regarding the query, so use the information to explain the query very briefly. If the query is not relevant to the data provided, then cover it up. If the question is outside the data's context, like 'What is 1+1?' or 'What is an apple?', just say, 'I am an AI assistant, and my job is to provide the users with the best responses on resume generation and etc.'"

chat_history = [
    SystemMessage(content=prompt)
]

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    query = data.get("query", "").strip()
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    # Fetch relevant data using the retriever
    fetched_data = retriever.invoke(query)

    print(len(fetched_data))
    a=""
    for i in fetched_data:
        if i.page_content not in a:
            a+=i.page_content
    print(a,"...........................")
    chat_history.append(HumanMessage(content=query + " " + a))
    
    # Generate the response using the language model
    result = llm.invoke(chat_history)
    response = result.content
    
    # Append the AI response to the chat history
    chat_history.append(AIMessage(content=response))
    
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

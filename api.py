from flask import Flask, request, jsonify
from llama_index.core import VectorStoreIndex,  Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader

app = Flask(__name__)
embedding_model = OllamaEmbedding(model_name="llama3")

@app.route('/embedding', methods=['POST'])
def get_response():

    body = request.get_json()['prompt']
    parser = PDFReader()
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
    "docs", file_extractor=file_extractor
    ).load_data()
    
    ollama_embedding = OllamaEmbedding(
        model_name="llama3",
        base_url="http://localhost:11434", 
        ollama_additional_kwargs={"mirostat": 0},
    )
    Settings.embed_model = ollama_embedding

    Settings.llm = Ollama(model="llama3", temperature=0.2)

    index = VectorStoreIndex.from_documents(
        documents,
    )

    query_engine = index.as_query_engine()
    response = query_engine.query(str(body))

    return jsonify({'msg':response.response})

if __name__ == '__main__':
    app.run(debub=True)
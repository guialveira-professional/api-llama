from llama_index.core import VectorStoreIndex,  Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import SimpleDirectoryReader
from llama_index.readers.file import PDFReader
import streamlit as st

st.set_page_config(page_title= "Construtor ATS",
                   initial_sidebar_state="expanded",
                   layout='wide',
                   page_icon='')

col1, col2 = st.columns([0.5,1])



with st.sidebar:

    st.title("Personal CV bot")
    
    parser = PDFReader()
    file_extractor = {".pdf": parser}
    documents = SimpleDirectoryReader(
        "docs", file_extractor=file_extractor
        ).load_data()


    files_uploaded = st.file_uploader(label = "", type=['pdf'],accept_multiple_files=True)
    
for uploaded_file in files_uploaded:
        bytes_data = uploaded_file.read()
        documents.append(bytes_data)

with col1:
    job_description = st.text_area("Descrição da vaga ", height=800)
    add = st.button('add')

if add: 
        context = job_description
        key = 1
else:
        key=0



def bot(prompt, documents = documents):
            
    ollama_embedding = OllamaEmbedding(
        model_name="llama3",
        base_url="http://localhost:11434", 
        ollama_additional_kwargs={"mirostat": 0},
        )
    Settings.embed_model = ollama_embedding

    Settings.llm = Ollama(model="llama3", temperature=1)

    index = VectorStoreIndex.from_documents(
        documents,
        )

    query_engine = index.as_query_engine()
    response = query_engine.query(str(prompt))
    return response


with col2:
    st.title("💬 Chatbot")
    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    if prompt := st.chat_input():

        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        if key==1:
            total_promp = f"""*Context:
            Job description: {context}
            """ + " "+ str(st.session_state.messages)
            response = bot(prompt=total_promp)
        else:
            response = bot(prompt=st.session_state.messages)
        msg = response.response
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.chat_message("assistant").write(msg)

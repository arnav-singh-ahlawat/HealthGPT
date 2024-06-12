import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.memory import ConversationBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# Initialize Streamlit app title
st.title("HealthGPT 💊")

# Define conversation_chat function
def conversation_chat(query):
    result = chain({"question": query, "chat_history": st.session_state['history']})
    st.session_state['history'].append((query, result["answer"]))
    return result["answer"]

# Define initialize_session_state function
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! How can I treat you today?"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

# Define display_chat_history function
def display_chat_history():
    # Create containers for input and output messages
    reply_container = st.container()
    container = st.container()

    with container:
        # Create a form for user input
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("message:", placeholder="Ask me any medical question", key='input')
            submit_button = st.form_submit_button(label='send')

            # If user submits a message, get the bot's response
            if submit_button and user_input:
                output = conversation_chat(user_input)

                # Append user input and bot's response to session state
                st.session_state['past'].append(user_input)
                st.session_state['generated'].append(output)

    # Display chat history
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                # Display user's message
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
                # Display bot's response
                message(st.session_state["generated"][i], key=str(i))

# Load documents and initialize components
loader = DirectoryLoader('data/', glob="*.pdf", loader_cls=PyPDFLoader)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': "cpu"})
vector_store = FAISS.from_documents(text_chunks, embeddings)
llm = CTransformers(model="llama-2-7b-chat.ggmlv3.q4_0.bin", model_type="llama", config={'max_new_tokens': 128, 'temperature': 0.01})
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff', retriever=vector_store.as_retriever(search_kwargs={"k": 2}), memory=memory)

# Initialize session state and display chat history
initialize_session_state()
display_chat_history()
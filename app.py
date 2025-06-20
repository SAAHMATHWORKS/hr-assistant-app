import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
# from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
# 1. Setup OpenAI (Make sure your key is properly set in your environment)

from dotenv import load_dotenv

# Load config from a .env file:
load_dotenv()


OPENAI_API_KEY = os.environ['OPENAI_API_KEY']
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# 2. HR Prompt in French
HR_PROMPT_TEMPLATE = """
Vous êtes un assistant RH pour {company_name}.
Fournissez des réponses précises et professionnelles basées sur le contexte fourni.
Si vous n'êtes pas sûr(e), invitez l'utilisateur à contacter le service RH.

Contexte : {context}

Question : {question}
Répondez de manière professionnelle et utile :
"""

HR_PROMPT = PromptTemplate(
    template=HR_PROMPT_TEMPLATE,
    input_variables=["context", "question", "company_name"]
)

# 3. Initialize Vector Store (load your existing Chroma DB)
persist_directory = 'chroma_db'  # Update this path
embedding = OpenAIEmbeddings()
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embedding)

# 4. Initialize Conversational Retrieval QA Chain
def initialize_hr_qa_chain(vector_store, company_name="Gozem Africa"):
    llm = ChatOpenAI(model_name="gpt-4", temperature=0.2)
    
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )
    
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "filter": {"is_confidential": False}}
        ),
        memory=memory,
        combine_docs_chain_kwargs={
            "prompt": HR_PROMPT.partial(company_name=company_name)
        },
        return_source_documents=True
    )

qa_chain = initialize_hr_qa_chain(vectorstore)

# 5. Streamlit App Interface
st.title("Assistant RH - Gozem Africa")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_question = st.text_input("Posez votre question RH ici :")

if user_question:
    result = qa_chain({"question": user_question, "chat_history": st.session_state.chat_history})
    
    st.session_state.chat_history.append((user_question, result["answer"]))
    
    st.markdown("### Réponse :")
    st.write(result["answer"])
    
    with st.expander("Voir les documents sources :"):
        for doc in result["source_documents"]:
            st.write(f"- Source : {doc.metadata.get('source', 'Inconnue')}")
    
    st.markdown("---")
    st.markdown("### Historique de la conversation :")
    for i, (q, a) in enumerate(st.session_state.chat_history):
        st.write(f"**Q{i+1}:** {q}")
        st.write(f"**A{i+1}:** {a}")


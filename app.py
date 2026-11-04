 import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

# Page configuration
st.set_page_config(
    page_title="Zyro Dynamics - HR Help Desk",
    page_icon="💼",
    layout="centered"
)

# Custom Styling
st.markdown("""
    <style>
    .main-header { font-size: 2.2rem; color: #1E3A8A; font-weight: 700; }
    .sub-header { font-size: 1.1rem; color: #4B5563; margin-bottom: 25px; }
    .stButton>button { background-color: #2563EB; color: white; font-weight: 600; border-radius: 8px; width: 100%; }
    .stButton>button:hover { background-color: #1D4ED8; color: white; }
    .answer-box { background-color: #F8FAFC; padding: 20px; border-radius: 10px; border-left: 5px solid #2563EB; margin-top: 15px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💼 Zyro Dynamics HR Help Desk</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Your intelligent assistant for company policies, benefits, and workplace guidelines.</p>', unsafe_allow_html=True)
st.markdown("---")

# Load Secrets safely
os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = st.secrets.get("LANGCHAIN_API_KEY", "dummy")
os.environ["LANGCHAIN_PROJECT"] = "Zyro-HR-Bot"

@st.cache_resource
def load_bot():
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = FAISS.load_local("faiss_hr_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, "fetch_k": 15, "lambda_mult": 0.7})
    llm = ChatGroq(model="openai/gpt-oss-20b", temperature=0.0)

    # Clean, error-free document formatting without f-string bracket conflicts
    def format_docs(docs):
        formatted = []
        for d in docs:
            source_name = d.metadata.get("file_name", "HR_Policy.pdf")
            page_num = d.metadata.get("page_number", 1)
            text = f"(Source: {source_name}, Page {page_num})\n{d.page_content}"
            formatted.append(text)
        return "\n\n".join(formatted)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the HR Help Desk chatbot for Zyro Dynamics.
Answer ONLY using the provided HR policy context. Always cite the document and page at the end: (Source: <file_name>, Page <page_number>).
If the question is not about Zyro Dynamics HR policies, reply ONLY with:
"I can only help with HR policy questions at Zyro Dynamics." """),
        ("human", "Context:\n{context}\n\nQuestion: {question}")
    ])

    def check_scope(data):
        res = vectorstore.similarity_search_with_score(data["question"], k=1)
        data["in_scope"] = float(res[0][1]) <= 1.05
        return data

    in_scope_chain = (
        {"context": RunnableLambda(lambda x: x["question"]) | retriever | RunnableLambda(format_docs),
         "question": RunnableLambda(lambda x: x["question"])}
        | prompt | llm | StrOutputParser()
    )
    refusal_chain = RunnableLambda(lambda x: "I can only help with HR policy questions at Zyro Dynamics.")
    
    return RunnableLambda(lambda q: {"question": q}) | RunnableLambda(check_scope) | RunnableBranch((lambda x: x["in_scope"], in_scope_chain), refusal_chain)

with st.spinner("Loading HR Knowledge Base..."):
    bot_chain = load_bot()

query = st.text_input("💡 Ask a question about HR policies:", placeholder="e.g., What is the leave policy?")

if st.button("Ask Bot") and query:
    with st.spinner("Analyzing HR documents..."):
        response = bot_chain.invoke(query)
        st.markdown("### 📋 Answer")
        st.markdown(f'<div class="answer-box">{response}</div>', unsafe_allow_html=True)
elif st.button("Ask Bot") and not query:
    st.warning("Please type a question first.")

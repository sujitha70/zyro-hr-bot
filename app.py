 import os
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableBranch

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="💼", layout="centered")
st.title("💼 Zyro Dynamics - HR Help Desk Chatbot")
st.caption("Ask any question across our official HR policy documents. Every answer includes source citations.")

# Load API keys from Streamlit Secrets
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

    def format_docs(docs):
        return "\n\n".join([f"}, Page {d.metadata['page_number']}]\n{d.page_content}" for d in docs])

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

bot_chain = load_bot()

query = st.text_input("Enter your HR Policy Question:")
if st.button("Ask Bot") and query:
    with st.spinner("Searching policies..."):
        response = bot_chain.invoke(query)
        st.write("### Answer")
        st.write(response)
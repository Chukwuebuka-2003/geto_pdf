import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Streamlit Page Configuration
st.set_page_config(page_title="AI-Powered PDF Assistant", page_icon="📄", layout="wide")

# Initialize the Google Gemini LLM
api_key = st.secrets["google_genai"]["api_key"]  # Replace with your actual API key
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)

# Define prompt templates
summarization_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following text in a clear and concise manner:\n\n"
        "{text}\n\n"
        "Focus on key points and main ideas."
    ),
)

query_prompt_template = PromptTemplate(
    input_variables=["text", "query"],
    template=(
        "You are an expert assistant. Use the following document text to answer the question below:\n\n"
        "Document:\n{text}\n\n"
        "Question:\n{query}\n\n"
        "Provide a clear and detailed answer."
    ),
)

# Create LangChain LLM Chains
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt_template)
query_chain = LLMChain(llm=llm, prompt=query_prompt_template)

# Streamlit App Layout
st.title("AI-Powered PDF Assistant")
tab1, tab2 = st.tabs(["Summarize PDF", "Query PDF"])

# Tab 1: Summarization
with tab1:
    st.header("Summarize PDF")
    uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

    if uploaded_file:
        # Extract text from the PDF
        try:
            reader = PdfReader(uploaded_file)
            pdf_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            pdf_text = ""
        
        if pdf_text.strip():
            st.text_area("Extracted PDF Content", value=pdf_text, height=300)

            if st.button("Summarize PDF"):
                with st.spinner("Summarizing the PDF..."):
                    try:
                        summary = summarization_chain.run({"text": pdf_text})
                        st.subheader("Summary")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Error generating summary: {e}")
        else:
            st.error("Could not extract text from the PDF.")

# Tab 2: Querying
with tab2:
    st.header("Query PDF")
    uploaded_file = st.file_uploader("Upload a PDF file for querying", type=["pdf"], key="query_uploader")

    if uploaded_file:
        # Extract text from the PDF
        try:
            reader = PdfReader(uploaded_file)
            pdf_text = "\n".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            st.error(f"Error processing PDF: {e}")
            pdf_text = ""

        if pdf_text.strip():
            st.text_area("Extracted PDF Content", value=pdf_text, height=300)

            user_query = st.text_input("Enter your question about the PDF:")
            if st.button("Get Answer"):
                with st.spinner("Fetching the answer..."):
                    try:
                        answer = query_chain.run({"text": pdf_text, "query": user_query})
                        st.subheader("Answer")
                        st.write(answer)
                    except Exception as e:
                        st.error(f"Error generating answer: {e}")
        else:
            st.error("Could not extract text from the PDF.")

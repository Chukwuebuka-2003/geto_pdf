import streamlit as st
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Streamlit Page Configuration
st.set_page_config(page_title="AI-Powered PDF Summarizer", page_icon="📄", layout="wide")

# Initialize the Google Gemini LLM
api_key = st.secrets["google_genai"]["api_key"]  # Replace with your actual API key
llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)

# Define prompt template for summarization
summarization_prompt_template = PromptTemplate(
    input_variables=["text"],
    template=(
        "Summarize the following text in a clear and concise manner:\n\n"
        "{text}\n\n"
        "Focus on key points and main ideas."
    ),
)

# Create LangChain LLM Chain for summarization
summarization_chain = LLMChain(llm=llm, prompt=summarization_prompt_template)

# Streamlit App Layout
st.title("AI-Powered PDF Summarizer")

# Summarization Functionality
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

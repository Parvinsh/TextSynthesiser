import streamlit as st
import requests
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain import hub
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import os
import warnings
from fpdf import FPDF

warnings.filterwarnings("ignore")

# Load environment variables (for API key)

load_dotenv()
GroqClod_API_KEY = os.getenv('GroqClod_API_KEY_8b')

# Initialize language model
if "llm" not in st.session_state:
    st.session_state["llm"] = ChatGroq(
        model="llama3-8b-8192",
        groq_api_key=GroqClod_API_KEY, 
        temperature=0,
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

# Streamlit app layout
st.title("Web Link Information Synthesis")
st.write("Enter at least 3 links, and then ask questions based on the combined information.")

# Step 1: Collect Links
link1 = st.text_input("Enter Link 1")
link2 = st.text_input("Enter Link 2")
link3 = st.text_input("Enter Link 3")
additional_links = st.text_area("Enter additional links, one per line (optional)")

links = [link for link in [link1, link2, link3] if link] + additional_links.splitlines()

# Function to retrieve and clean text from a URL
def fetch_text_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, "html.parser")
        paragraphs = soup.find_all("p")
        text = "\n".join(paragraph.text for paragraph in paragraphs)
        return text
    except Exception as e:
        st.error(f"Error fetching content from {url}: {e}")
        return None

# Step 2: Process Links if there are at least 3 valid ones
if len(links) >= 3:
    if "vectorstore" not in st.session_state:
        st.write("Fetching and processing content from links...")
        
        # Fetch content from each link and convert it to Document format
        documents = []
        for url in links:
            text = fetch_text_from_url(url)
            if text:
                documents.append(Document(page_content=text))

        if documents:
            # Split the text for embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=80)
            docs = text_splitter.split_documents(documents=documents)

            # Generate embeddings and vectorstore
            embeddings = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_documents(docs, embeddings)

            # Store vectorstore in session state for faster queries
            st.session_state["vectorstore"] = vectorstore

            # Set up retrieval chain
            retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            combine_docs_chain = create_stuff_documents_chain(st.session_state["llm"], retrieval_qa_chat_prompt)
            retrieval_chain = create_retrieval_chain(vectorstore.as_retriever(), combine_docs_chain)

            # Store retrieval chain in session state
            st.session_state["retrieval_chain"] = retrieval_chain
            st.success("Content processed successfully! You can now ask questions.")
        else:
            st.error("Failed to retrieve content from the provided links.")
else:
    st.info("Please enter at least 3 valid links to proceed.")

# Step 3: Ask Questions about the Combined Content
if "retrieval_chain" in st.session_state:
    def get_answer(user_input):
        result = st.session_state["retrieval_chain"].invoke({"input": user_input})
        return result["answer"]

    # Text input for user query (using a text area for multi-line input)
    user_query = st.text_area("Enter your query about the information from the links:", height=150)

    # If the user has entered a query, process it and display the answer
    if user_query:
        try:
            answer = get_answer(user_query)
            st.write("Answer:", answer)
            
            # Button to save the result as a PDF
            if st.button("Save as PDF"):
                # Create a PDF with the result
                pdf = FPDF()
                pdf.set_auto_page_break(auto=True, margin=15)
                pdf.add_page()
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt="Answer to your Query", ln=True, align="C")
                pdf.ln(10)
                pdf.multi_cell(0, 10, answer)  # This allows for long text to be wrapped

                # Save PDF to a file
                pdf_output = "/tmp/answer_output.pdf"
                pdf.output(pdf_output)
                
                # Allow the user to download the generated PDF
                with open(pdf_output, "rb") as f:
                    st.download_button(
                        label="Download PDF",
                        data=f,
                        file_name="answer_output.pdf",
                        mime="application/pdf"
                    )
        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.write("Please enter and submit at least 3 valid links to proceed.")

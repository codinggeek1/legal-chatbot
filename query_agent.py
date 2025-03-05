import fitz  # PyMuPDF for extracting text from PDFs
import os
import nltk
from nltk.tokenize import word_tokenize
from rank_bm25 import BM25Okapi

nltk.download("punkt")

class QueryAgent:
    def __init__(self, pdf_folder):
        self.pdf_folder = pdf_folder
        self.legal_docs = self.load_pdfs()
        self.tokenized_docs = [word_tokenize(doc.lower()) for doc in self.legal_docs]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def load_pdfs(self):
        """Loads and extracts text from all PDFs in the folder."""
        documents = []
        for filename in os.listdir(self.pdf_folder):
            if filename.endswith(".pdf"):
                doc_text = self.extract_text_from_pdf(os.path.join(self.pdf_folder, filename))
                documents.append(doc_text)
        return documents

    def extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file."""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text

    def retrieve_relevant_text(self, query):
        """Fetches the most relevant section from PDFs based on the query."""
        tokenized_query = word_tokenize(query.lower())
        scores = self.bm25.get_scores(tokenized_query)
        best_doc_index = scores.argmax()
        return self.legal_docs[best_doc_index]

# Initialize the query agent with the PDF directory
query_agent = QueryAgent(pdf_folder="legal_pdfs")

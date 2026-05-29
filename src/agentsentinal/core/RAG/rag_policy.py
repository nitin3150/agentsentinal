import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer
 
 
CHUNK_SIZE = 500        # characters per chunk
CHUNK_OVERLAP = 50      # overlap between chunks to avoid cutting mid-sentence
TOP_K = 3               # number of chunks to retrieve per query
 
 
class PolicyRAG:
 
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")  # small, fast, good enough
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("policy")
 
    # ------------------------------------------------------------------
    # Step 1: Parse PDF into raw text
    # ------------------------------------------------------------------
 
    def _parse(self, pdf_path: str) -> str:
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text() or ""
                pages.append(text)
        return "\n\n".join(pages)
 
    # ------------------------------------------------------------------
    # Step 2: Split text into overlapping chunks
    # ------------------------------------------------------------------
 
    def _chunk(self, text: str) -> list[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunks.append(text[start:end])
            start += CHUNK_SIZE - CHUNK_OVERLAP
        return chunks
 
    # ------------------------------------------------------------------
    # Step 3: Embed and store in ChromaDB
    # ------------------------------------------------------------------
 
    def index(self, pdf_path: str):
        print(f"Parsing {pdf_path}...")
        text = self._parse(pdf_path)
 
        print("Chunking...")
        chunks = self._chunk(text)
 
        print(f"Embedding and indexing {len(chunks)} chunks...")
        embeddings = self.model.encode(chunks).tolist()
 
        self.collection.add(
            ids=[str(i) for i in range(len(chunks))],
            documents=chunks,
            embeddings=embeddings,
        )
        print("Done. Ready to query.")
 
    # ------------------------------------------------------------------
    # Step 4: Query — returns the most relevant chunks
    # ------------------------------------------------------------------
 
    def query(self, question: str) -> list[str]:
        embedding = self.model.encode([question]).tolist()
        results = self.collection.query(
            query_embeddings=embedding,
            n_results=TOP_K,
        )
        documents = results.get("documents")
        if not documents:
            return []
        return documents[0] 
 
    def query_as_context(self, question: str) -> str:
        """Returns retrieved chunks as a single string for LLM injection."""
        chunks = self.query(question)
        return "\n\n---\n\n".join(chunks)
import os
from dotenv import load_dotenv

# Try to import langchain components. Some installations of the package
# may not include all submodules; provide a graceful fallback for the
# text splitter and clear, actionable errors for other missing pieces.
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except Exception:
    # Lightweight fallback splitter (naive) so the app can still run
    # for tasks that only need basic chunking. This avoids an immediate
    # crash when langchain's text_splitter isn't available.
    class RecursiveCharacterTextSplitter:
        def __init__(self, chunk_size=500, chunk_overlap=100):
            self.chunk_size = int(chunk_size)
            self.chunk_overlap = int(chunk_overlap)

        def split_text(self, text: str):
            if not text:
                return []
            chunks = []
            i = 0
            length = len(text)
            while i < length:
                end = i + self.chunk_size
                chunks.append(text[i:end])
                # move forward but keep overlap
                i = end - self.chunk_overlap
                if i <= 0:
                    i = end
            return chunks

try:
    from langchain.embeddings import HuggingFaceEmbeddings
except Exception:
    # Provide a lightweight fallback that uses sentence-transformers directly
    HuggingFaceEmbeddings = None
    try:
        from sentence_transformers import SentenceTransformer

        class HuggingFaceEmbeddings:
            """Minimal wrapper that exposes embed_documents and embed_query
            compatible with langchain's expected interface using
            sentence-transformers directly.
            """
            def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
                # Map common HF model ids to the sentence-transformers name
                model = model_name.split('/')[-1]
                self._model = SentenceTransformer(model)

            def embed_documents(self, texts):
                # returns list[list[float]]
                vectors = self._model.encode(texts, show_progress_bar=False)
                return [list(map(float, v)) for v in vectors]

            def embed_query(self, text):
                v = self._model.encode([text], show_progress_bar=False)[0]
                return list(map(float, v))

    except Exception:
        HuggingFaceEmbeddings = None

try:
    from langchain.vectorstores import Chroma
except Exception:
    Chroma = None
    # If langchain's Chroma wrapper isn't available, try to provide a
    # lightweight adapter around the chromadb client so the rest of the
    # app (create_or_load_chroma, retriever usage) can work.
    try:
        import chromadb
        import uuid

        class Chroma:
            """Minimal adapter providing the subset of the LangChain Chroma
            interface used by this app: from_texts, persist, as_retriever.
            """
            def __init__(self, client, collection):
                self._client = client
                self._collection = collection

            @classmethod
            def from_texts(cls, texts, embedding, persist_directory=None, collection_name='default_collection', **kwargs):
                # embedding: an object with embed_documents and embed_query methods
                if persist_directory:
                    client = chromadb.PersistentClient(path=persist_directory)
                else:
                    client = chromadb.Client()
                collection = client.get_or_create_collection(name=collection_name)

                # compute embeddings
                if hasattr(embedding, 'embed_documents'):
                    vectors = embedding.embed_documents(texts)
                elif callable(embedding):
                    vectors = [embedding(t) for t in texts]
                else:
                    raise ValueError('Embedding object must provide embed_documents(texts)')

                ids = [str(uuid.uuid4()) for _ in texts]
                collection.add(documents=texts, embeddings=vectors, ids=ids)
                return cls(client, collection)

            def persist(self):
                try:
                    # chromadb client persists to configured directory
                    self._client.persist()
                except Exception:
                    # not critical; ignore if client doesn't implement persist
                    pass

            def as_retriever(self, search_kwargs=None):
                search_kwargs = search_kwargs or {}
                k = int(search_kwargs.get('k', 3))

                class Retriever:
                    def __init__(self, collection, embedder, k):
                        self._collection = collection
                        self._embedder = embedder
                        self._k = k

                    def get_relevant_documents(self, query):
                        if hasattr(self._embedder, 'embed_query'):
                            qv = self._embedder.embed_query(query)
                        elif callable(self._embedder):
                            qv = self._embedder(query)
                        else:
                            raise ValueError('Embedder must provide embed_query')
                        results = self._collection.query(query_embeddings=[qv], n_results=self._k, include=['documents'])
                        docs = []
                        for docs_list in results.get('documents', []):
                            docs.extend(docs_list)
                        return docs

                # We need access to the embedding object used when creating the collection.
                # LangChain's Chroma keeps the embedding; in our adapter we stored none, so
                # callers should pass an embedder when using create_or_load_chroma. To keep
                # compatibility, attempt to use a global get_embeddings() for queries.
                try:
                    embedder = get_embeddings()
                except Exception:
                    embedder = None

                return Retriever(self._collection, embedder, k)

    except Exception:
        Chroma = None

try:
    from langchain.llms import HuggingFaceHub
except Exception:
    HuggingFaceHub = None

try:
    from langchain.chains import ConversationalRetrievalChain
except Exception:
    ConversationalRetrievalChain = None

try:
    from langchain.memory import ConversationBufferMemory
except Exception:
    ConversationBufferMemory = None

# Load .env variables
load_dotenv()

CHROMA_PATH = "chroma_db"
HF_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")  # Add your token in .env

def get_embeddings():
    """Return HuggingFace embeddings"""
    if HuggingFaceEmbeddings is None:
        raise ImportError(
            "HuggingFaceEmbeddings is not available.\n"
            "Install/upgrade langchain and required extras, e.g:\n"
            "pip install --upgrade langchain sentence-transformers huggingface-hub"
        )
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def text_to_chunks(text, chunk_size=500, chunk_overlap=100):
    """Split text into chunks"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_text(text)

def create_or_load_chroma(chunks, collection_name="default_collection"):
    """Create or load Chroma vector DB"""
    embeddings = get_embeddings()
    os.makedirs(CHROMA_PATH, exist_ok=True)
    if Chroma is None:
        raise ImportError(
            "Chroma vectorstore (langchain.vectorstores.Chroma) is not available.\n"
            "Install chromadb and the full langchain package:\n"
            "pip install --upgrade chromadb langchain"
        )

    vectordb = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_PATH,
        collection_name=collection_name,
    )
    vectordb.persist()
    return vectordb

def create_llm(model_repo="google/flan-t5-large"):
    """Create HuggingFace LLM"""
    if not HF_API_KEY:
        raise ValueError("HuggingFace API key not found in .env")
    if HuggingFaceHub is None:
        raise ImportError(
            "HuggingFaceHub LLM wrapper is not available (langchain.llms.HuggingFaceHub).\n"
            "Install/upgrade langchain and huggingface_hub: pip install --upgrade langchain huggingface-hub"
        )
    return HuggingFaceHub(
        repo_id=model_repo,
        huggingfacehub_api_token=HF_API_KEY,
        model_kwargs={"temperature": 0.2, "max_length": 512},
    )

def create_conversational_chain(vectordb, llm=None):
    """Create ConversationalRetrievalChain with memory"""
    if llm is None:
        llm = create_llm()
    if ConversationBufferMemory is None or ConversationalRetrievalChain is None:
        raise ImportError(
            "Required langchain components for conversational chains are not available.\n"
            "Install or upgrade langchain: pip install --upgrade langchain"
        )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)
    return chain, memory

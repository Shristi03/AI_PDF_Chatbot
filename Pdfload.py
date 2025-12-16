import os
import glob
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader
from google import genai

# --- CONFIGURATION ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(" API Key missing! Make sure it is in your .env file.")

client = genai.Client(api_key=GEMINI_API_KEY)

# Persistent Vector Database
chroma_client = chromadb.PersistentClient(path="chroma_db")


# --- NEW: AUTO-DETECT MODEL FUNCTION ---
def get_best_available_model():
    """
    Asks Google which models are allowed for this API Key
    and picks the best one automatically.
    """
    print("Auto-detecting available models...")
    try:
        # Get all models
        my_models = [m.name for m in client.models.list()]

        # Preference List (Try these in order)
        priority_list = [
            "models/gemini-1.5-flash",
            "models/gemini-1.5-flash-001",
            "models/gemini-1.5-flash-002",
            "models/gemini-1.5-pro",
            "models/gemini-1.5-pro-001",
            "models/gemini-1.5-pro-002",
            "models/gemini-1.0-pro"
        ]

        # 1. Check for favorites
        for model in priority_list:
            if model in my_models:
                print(f" Auto-Selected Model: {model}")
                return model

        # 2. Fallback: Find any model with 'gemini' in the name (excluding embedding models)
        for model in my_models:
            if "gemini" in model and "embedding" not in model and "vision" not in model:
                print(f" Auto-Selected Model (Fallback): {model}")
                return model

    except Exception as e:
        print(f" Auto-detect failed: {e}")

    # 3. Last Resort
    print(" defaulting to 'gemini-1.5-pro'")
    return "gemini-1.5-pro"


# Run the detection ONCE at startup
CURRENT_MODEL = get_best_available_model()


# --- HELPER FUNCTIONS ---

def get_gemini_embedding(text):
    try:
        result = client.models.embed_content(
            model="text-embedding-004",
            contents=text
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f" Error embedding text: {e}")
        return []


def load_and_chunk_pdfs(folder_name):
    documents = []
    metadatas = []
    ids = []

    script_dir = os.path.dirname(os.path.abspath(__file__))
    target_path = os.path.join(script_dir, folder_name)

    pdf_files = glob.glob(os.path.join(target_path, "**", "*.pdf"), recursive=True)

    if not pdf_files:
        print(" No PDFs found. Please check your 'pdfs' folder.")
        return None, None, None

    print(f" Found {len(pdf_files)} PDFs. Processing...")
    id_counter = 0

    for pdf_file in pdf_files:
        filename = os.path.basename(pdf_file)
        try:
            reader = PdfReader(pdf_file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text: continue

                chunk_size = 2000
                overlap = 500

                for i in range(0, len(text), chunk_size - overlap):
                    chunk = text[i:i + chunk_size]
                    documents.append(chunk)
                    metadatas.append({"source": filename, "page": page_num + 1})
                    ids.append(f"doc_{id_counter}")
                    id_counter += 1
        except Exception as e:
            print(f"    Error reading {filename}: {e}")

    return documents, metadatas, ids


def setup_vector_db(documents, metadatas, ids):
    collection_name = "my_pdf_knowledge_base"
    try:
        chroma_client.delete_collection(name=collection_name)
    except:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    print(f" Generating embeddings for {len(documents)} chunks...")

    embeddings = []
    batch_size = 10

    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        for doc in batch_docs:
            emb = get_gemini_embedding(doc)
            embeddings.append(emb)
        print(f"   ...processed {min(i + batch_size, len(documents))}/{len(documents)}")

    collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(" Database built successfully!")
    return collection


def query_rag_system(collection, question):
    # 1. Embed Question
    query_embedding = get_gemini_embedding(question)

    # 2. Query DB
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    # 3. Build Context
    retrieved_docs = results['documents'][0]
    retrieved_meta = results['metadatas'][0]

    context_text = ""
    print(f"\n FOUND RELEVANT CHUNKS:")
    for i, doc in enumerate(retrieved_docs):
        meta = retrieved_meta[i]
        source_info = f"[Source: {meta['source']}, Page: {meta['page']}]"
        print(f"   {i + 1}. {source_info}")
        context_text += f"Content: {doc}\nReference: {source_info}\n\n"

    # 4. Generate Answer using the AUTO-DETECTED MODEL
    prompt = f"""
    You are a helpful assistant. Use the following Retrieved Context to answer the user's question.

    --- RETRIEVED CONTEXT ---
    {context_text}
    -------------------------

    USER QUESTION: {question}

    INSTRUCTIONS:
    1. Answer only based on the context above.
    2. If the answer is not in the context, say "I don't know based on the documents."
    3. Cite the 'Reference' (Source and Page) for every fact you state.
    """

    # We use the global variable we found at the start
    response = client.models.generate_content(
        model=CURRENT_MODEL,
        contents=prompt
    )

    return response.text


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    folder_name = "pdfs"

    # Check if we already built the DB?
    # For now, we rebuild it every time to be safe.
    print("--- STEP 1: LOADING PDFS ---")
    docs, meta, ids = load_and_chunk_pdfs(folder_name)

    if docs:
        print("\n--- STEP 2: BUILDING DATABASE ---")
        collection = setup_vector_db(docs, meta, ids)

        print("\n--- STEP 3: READY TO CHAT ---")
        while True:
            user_q = input("\nAsk a question (or 'quit'): ")
            if user_q.lower() in ['quit', 'exit']:
                break

            print("Thinking...")
            try:
                answer = query_rag_system(collection, user_q)
                print("\n" + "=" * 50)
                print("GEMINI ANSWER:")
                print("=" * 50)
                print(answer)
            except Exception as e:

                print(f" An error occurred: {e}")

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tiktoken
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
from supadata import Supadata, SupadataError
import os

load_dotenv()

model = ChatNVIDIA(model='openai/gpt-oss-120b')
embeddings = NVIDIAEmbeddings(model='nvidia/nv-embedqa-e5-v5')  # FIXED: was embeddingSs

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("video-rag")
supadata = Supadata(os.getenv("SUPADATA_API_KEY"))

# Rough token counter used only to enforce the embedding model's 512-token
# limit. This isn't NVIDIA's exact tokenizer, but cl100k_base gives a close
# enough (slightly conservative) estimate for safety-net purposes.
_token_encoder = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(_token_encoder.encode(text))

def enforce_token_limit(docs, max_tokens=450):
    """
    Guarantees every doc chunk is under max_tokens (safety margin below the
    real 512 limit). Any chunk that's still too large after character-based
    splitting (common with translated/non-English-origin text, which can
    tokenize denser than plain English) gets recursively halved by
    character length until it fits.
    """
    safe_docs = []
    for doc in docs:
        text = doc.page_content
        if count_tokens(text) <= max_tokens:
            safe_docs.append(doc)
            continue
        # Recursively split oversized chunks in half until each piece fits
        stack = [text]
        while stack:
            chunk = stack.pop()
            if count_tokens(chunk) <= max_tokens:
                if chunk.strip():
                    doc.page_content = chunk
                    from copy import deepcopy
                    new_doc = deepcopy(doc)
                    new_doc.page_content = chunk
                    safe_docs.append(new_doc)
                continue
            mid = len(chunk) // 2
            stack.append(chunk[:mid])
            stack.append(chunk[mid:])
    return safe_docs

# --- 1. INGEST ---
def extract_video_id(url):
    try:
        # shortened youtube url (e.g. https://youtu.be/VIDEO_ID)
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        # full youtube url (e.g. https://www.youtube.com/watch?v=VIDEO_ID)
        return url.split("?v=")[-1].split("&")[0]
    except IndexError:
        return None

def ingest(videoUrl):
    video_id = extract_video_id(videoUrl)

    stats = index.describe_index_stats()
    namespace_stats = stats.namespaces.get(video_id)
    if namespace_stats and namespace_stats.vector_count > 0:
        # Generate questions from existing vectors
        vector_store = PineconeVectorStore(
            index_name="video-rag",
            embedding=embeddings,
            namespace=video_id
        )
        retriever = vector_store.as_retriever(search_kwargs={'k': 3})
        existing_docs = retriever.invoke("video summary")
        suggested_questions = generate_suggested_questions(existing_docs)
        return {"message": "Already ingested", "suggestedQuestions": suggested_questions}

    try:
        # 1. Try YouTubeTranscriptApi first (Direct & Free)
        try:
            yt_api = YouTubeTranscriptApi()  # FIXED: was YoutubeTranscriptApi (wrong casing)
            transcript_list = yt_api.fetch(video_id, languages=['en', 'en-IN'])
            # NOTE: newer versions of youtube-transcript-api return snippet objects
            # with a `.text` attribute rather than dict-style items. If you hit a
            # TypeError here, switch to: " ".join([item.text for item in transcript_list])
            transcript = " ".join([item['text'] for item in transcript_list])
            print(f"[Ingest] Successfully fetched transcript via youtube-transcript-api for {video_id}")
        except Exception as yt_err:
            print(f"[Ingest] YouTubeTranscriptApi failed: {yt_err}. Trying Supadata...")
            # 2. Fallback to Supadata
            try:
                result = supadata.transcript(
                    url=videoUrl,
                    lang="en",
                    text=True,
                    mode="auto"
                )
                transcript = result.content
                print(f"[Ingest] Successfully fetched transcript via Supadata for {video_id}")
            except SupadataError:
                return "No transcript available"

    except Exception as e:
        print(f"Transcript error: {e}")
        return "No transcript available"

    # NOTE: chunk_size is in characters, but the embedding model
    # (nvidia/nv-embedqa-e5-v5) has a hard 512-TOKEN limit per input.
    # Character-based splitting is only an approximation — translated /
    # non-English-origin transcripts (e.g. Hindi auto-captions translated
    # to English) can tokenize denser than plain English prose. So we
    # split on characters first, then run every chunk through a hard
    # token-count safety net (enforce_token_limit) before embedding.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100
    )

    docs = splitter.create_documents([transcript])
    docs = enforce_token_limit(docs, max_tokens=450)

    # store in pinecone
    PineconeVectorStore.from_documents(
        docs,
        embedding=embeddings,
        index_name="video-rag",
        namespace=video_id
    )

    # Generate suggested questions from first few chunks
    suggested_questions = generate_suggested_questions(docs[:3])

    return {"message": "Ingested", "suggestedQuestions": suggested_questions}


def generate_suggested_questions(docs):
    """Generate 3 suggested questions from document chunks using LLM."""
    try:
        context = "\n\n".join([doc.page_content[:500] for doc in docs])

        prompt = PromptTemplate(
            template="""
            Based on the following video transcript excerpts, generate 3 natural questions that a viewer might ask about this video content.

            Context:
            {context}

            Provide exactly 3 questions, one per line. Make them specific to the content.

            Questions:
            """,
            input_variables=["context"]
        )

        chain = prompt | model | StrOutputParser()
        result = chain.invoke({"context": context})

        # Parse questions from result
        questions = [q.strip() for q in result.strip().split('\n') if q.strip() and '?' in q]
        return questions[:3]
    except Exception as e:
        print(f"Error generating questions: {e}")
        return []

def query(video_id, question, history=""):

    vector_store = PineconeVectorStore(
        index_name="video-rag",
        embedding=embeddings,
        namespace=video_id
    )

    retriever = vector_store.as_retriever(
        search_type='similarity',
        search_kwargs={'k': 4}
    )

    prompt = PromptTemplate(
        template = """
        You are an intelligent assistant answering questions based on a YouTube video.
        Your job is to provide accurate, helpful answers using ONLY the provided context.

        ---------------------
        CONVERSATION HISTORY:
        {history}
        ---------------------

        CONTEXT FROM VIDEO:
        {context}
        ---------------------

        QUESTION:
        {question}
        ---------------------

        INSTRUCTIONS:

        1. Use ONLY the provided context to answer.
        2. Do NOT use outside knowledge.
        3. If the answer is not clearly present, say:
           "I don't know based on the video."

        4. If the question is a follow-up, use the conversation history to understand it.

        5. Keep answers:
           - clear
           - concise
           - well-structured

        6. When possible:
           - explain concepts step-by-step
           - include key points from the video

        7. Do NOT hallucinate or guess.

        ---------------------

        ANSWER:
        """,
        input_variables=['context', 'question', 'history']
    )

    def format_docs(docs):
        if not docs:
            print("\n=== WARNING: NO CONTEXT FOUND ===\nRetriever returned 0 documents\n")
            return "No relevant context found in video."
        context = "\n\n".join([doc.page_content for doc in docs])
        print(f"\n=== CONTEXT SENT TO LLM ({len(docs)} chunks) ===\n{context}\n=== END CONTEXT ===\n")
        return context

    parallel_chain = RunnableParallel({
        'context': RunnableLambda(lambda x: x['question']) | retriever | RunnableLambda(format_docs),
        'question': lambda x: x['question'],
        'history': lambda x: x['history']
    })

    parser = StrOutputParser()
    chain = parallel_chain | prompt | model | parser

    return chain.invoke({"question": question, "history": history})

# print(ingest("Ih_4C6DJ0EU"))
# print(query("Ih_4C6DJ0EU", "What is this video about?"))
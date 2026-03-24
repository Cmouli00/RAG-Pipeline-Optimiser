import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from flashrank import Ranker, RerankRequest
import chromadb

# Initialize Ranker once to save memory
ranker = Ranker()
client = chromadb.Client()


def run_rag_strategy(file_path, query, strategy_name):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # Configuration Matrix
    configs = {
        "Small_Chunks": {"size": 256, "overlap": 30, "rerank": False},
        "Large_Chunks": {"size": 1024, "overlap": 100, "rerank": False},
        "Hybrid_Mock": {"size": 512, "overlap": 50, "rerank": False},
        "Advanced_Rerank": {"size": 512, "overlap": 50, "rerank": True}
    }
    
    conf = configs[strategy_name]
    splitter = RecursiveCharacterTextSplitter(chunk_size=conf["size"], chunk_overlap=conf["overlap"])
    split_docs = splitter.split_documents(docs)
    
    vectorstore = Chroma.from_documents(
        documents=split_docs, 
        # embedding=OpenAIEmbeddings(),
        embedding=OllamaEmbeddings(model="nomic-embed-text"), 
        collection_name=f"temp_{strategy_name}_{os.getpid()}",
                client=client)
    
    # Initial Retrieval (Get top 10 candidates)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    initial_docs = retriever.invoke(query)
    
    if conf["rerank"]:
        # Convert docs to FlashRank format
        passages = [{"id": i, "text": d.page_content} for i, d in enumerate(initial_docs)]
        rerank_request = RerankRequest(query=query, passages=passages)
        
        # Re-order based on semantic relevance
        results = ranker.rerank(rerank_request)
        # Take only the top 3 after re-ranking
        context_text = "\n---\n".join([r['text'] for r in results[:3]])
    else:
        # Standard retrieval top 3
        context_text = "\n---\n".join([d.page_content for d in initial_docs[:3]])
    
    # Clean up vectorstore to save RAM
    vectorstore.delete_collection()
    return context_text
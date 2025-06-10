import os
from typing import Dict, List

from dotenv import load_dotenv
from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_tavily import TavilySearch
from supabase.client import create_client

load_dotenv()


@tool
def create_retriever_from_supabase(query: str) -> str:
    """Search for similar documents in the Supabase vector store.

    This tool uses semantic search to find documents that are semantically similar to the provided query.
    It leverages the Supabase vector store and HuggingFace embeddings to perform the search.

    Args:
        query (str): The search query to find similar documents.

    Returns:
        str: A list of documents that are semantically similar to the query.
    """
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )
    supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])

    vector_store = SupabaseVectorStore(
        client=supabase,
        embedding=embeddings,
        table_name="documents",
        query_name="match_documents_langchain",
    )

    return vector_store.similarity_search(query)


@tool
def internet_search(query: str) -> Dict[str, List[Dict[str, str]]]:
    """Perform a web search using Tavily Search API.

    This tool searches the web for relevant information based on the provided query.
    It returns up to 3 most relevant results with their sources, titles, and content.

    Args:
        query (str): The search query to look up on the web.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing a list of search results.
            Each result is a dictionary with keys:
            - Source: URL of the webpage
            - Title: Title of the webpage
            - Content: Main content/text from the webpage
    """
    response = TavilySearch(max_results=3).invoke(query)

    formatted_answer = [
        {
            "Source": result["url"],
            "Title": result["title"],
            "Content": result["content"],
        }
        for result in response["results"]
    ]

    return {"web_results": formatted_answer}


@tool
def wiki_search(query: str) -> Dict[str, List[Dict[str, str]]]:
    """Search Wikipedia articles using the provided query.

    This tool searches Wikipedia for articles matching the query and returns
    up to 3 most relevant results with their sources, titles, and content.

    Args:
        query (str): The search query to look up on Wikipedia.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing a list of Wikipedia results.
            Each result is a dictionary with keys:
            - Source: URL of the Wikipedia article
            - Title: Title of the Wikipedia article
            - Content: Main content/text from the article
    """
    docs = WikipediaLoader(query=query, load_max_docs=3).load()

    formatted_answer = [
        {
            "Source": doc.metadata["source"],
            "Title": doc.metadata["title"],
            "Content": doc.page_content,
        }
        for doc in docs
    ]

    return {"wiki_results": formatted_answer}


@tool
def arxiv_search(query: str) -> Dict[str, List[Dict[str, str]]]:
    """Search academic papers on arXiv using the provided query.

    This tool searches arXiv for academic papers matching the query and returns
    up to 3 most relevant results with their sources, titles, and content.

    Args:
        query (str): The search query to look up on arXiv.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing a list of arXiv results.
            Each result is a dictionary with keys:
            - Source: URL of the arXiv paper
            - Title: Title of the academic paper
            - Content: Main content/abstract of the paper
    """
    docs = ArxivLoader(query=query, load_max_docs=3).load()

    formatted_answer = [
        {
            "Published": doc.metadata["Published"],
            "Authors": doc.metadata["Authors"],
            "Title": doc.metadata["Title"],
            "Content": doc.page_content,
        }
        for doc in docs
    ]

    return {"arxiv_results": formatted_answer}

from typing import Dict, List

from langchain_community.document_loaders import ArxivLoader, WikipediaLoader
from langchain_core.tools import tool
from langchain_tavily import TavilySearch


@tool
def internet_search(query: str) -> Dict[str, List[Dict[str, str]]]:
    """Perform a web search using Tavily Search API.

    This tool searches the web for relevant information based on the provided query.
    It returns up to 2 most relevant results with their sources, titles, and content.

    Args:
        query (str): The search query to look up on the web.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing a list of search results.
            Each result is a dictionary with keys:
            - Source: URL of the webpage
            - Title: Title of the webpage
            - Content: Main content/text from the webpage
    """
    response = TavilySearch(max_results=1).invoke(query)

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
    up to 2 most relevant results with their sources, titles, and content.

    Args:
        query (str): The search query to look up on Wikipedia.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing a list of Wikipedia results.
            Each result is a dictionary with keys:
            - Source: URL of the Wikipedia article
            - Title: Title of the Wikipedia article
            - Content: Main content/text from the article
    """
    docs = WikipediaLoader(query=query, load_max_docs=2).load()

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
    up to 2 most relevant results with their sources, titles, and content.

    Args:
        query (str): The search query to look up on arXiv.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing a list of arXiv results.
            Each result is a dictionary with keys:
            - Source: URL of the arXiv paper
            - Title: Title of the academic paper
            - Content: Main content/abstract of the paper
    """
    docs = ArxivLoader(query=query, load_max_docs=2).load()

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

import os
from typing import List, Literal

from langchain_core.messages import HumanMessage
from langchain_openai.chat_models import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from typing_extensions import TypedDict

from tools import add, arxiv_search, div, internet_search, mod, mult, sub, wiki_search

llm = ChatOpenAI(model="gpt-4.1-mini", api_key=os.getenv("OPENAI_API_KEY"))


class State(MessagesState):
    next: str


def create_supervisor_node(
    members: List[str],
) -> Command[Literal["math", "web_search", "__end__"]]:
    options = ["FINISH"] + members

    class Router(TypedDict):
        next: Literal[*options]

    def supervisor_node(
        state: State,
    ) -> Command[Literal["math", "web_search", "__end__"]]:
        prompt = f"""
        You are a supervisor tasked with managing a conversation between the 
        following workers: {members}. Given the following user request, respond with the worker to act next. 
        Each worker will perform a task and respond with their results and status. When finished, respond with FINISH.

        Guidelines:
            - The final answer must be either a number, a single string, or a comma-separated list of numbers or strings.
            - Do not include units (e.g. %, $, km) or commas inside numbers unless explicitly requested.
            - If you use abbreviations in strings, write out the full expression in parentheses the first time the word appears.
            - Write digits in full words only if asked.
        """

        messages = [{"role": "system", "content": prompt}] + state["messages"]

        response = llm.with_structured_output(Router).invoke(messages)

        goto = response["next"]
        if goto == "FINISH":
            goto = END

        return Command(goto=goto, update={"next": goto})

    return supervisor_node


def math_node(state: State) -> Command[Literal["supervisor"]]:
    math_agent = create_react_agent(model=llm, tools=[add, sub, mult, div, mod])

    result = math_agent.invoke(state)

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="math")
            ]
        },
        goto="supervisor",
    )


def web_search_node(state: State) -> Command[Literal["supervisor"]]:
    search_agent = create_react_agent(
        model=llm, tools=[internet_search, wiki_search, arxiv_search]
    )

    result = search_agent.invoke(state)

    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_search")
            ]
        },
        goto="supervisor",
    )


def build_worflow() -> StateGraph:
    workflow = StateGraph(State)

    workflow.add_node(
        "supervisor", create_supervisor_node(members=["math", "web_search"])
    )
    workflow.add_node("math", math_node)
    workflow.add_node("web_search", web_search_node)

    workflow.add_edge(START, "supervisor")

    return workflow.compile()


class BasicAgent:
    def __init__(self) -> None:
        print("BasicAgent initialized.")
        self.graph = build_worflow()

    def __call__(self, question: str) -> str:
        print(f"Agent received the question: {question[:50]}...")

        messages = [HumanMessage(content=question)]
        messages = self.graph.invoke({"messages": messages})

        return messages["messages"][-1].content

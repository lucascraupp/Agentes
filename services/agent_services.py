from typing import Any, Callable, List, Literal

import yaml
from langchain.agents.agent import AgentExecutor
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import create_react_agent

from tools import (
    add,
    arxiv_search,
    create_handoff_tool,
    div,
    internet_search,
    mod,
    mult,
    retriever_tool,
    sub,
    wiki_search,
)
from utils import pretty_print_messages


def load_prompt(name: str) -> str:
    with open("prompts.yaml", "r") as f:
        prompts = yaml.safe_load(f)
    return prompts[name]


def create_llm(model: Literal["groq", "openai"] = "openai") -> BaseChatModel:
    return (
        ChatGroq(model="qwen-qwq-32b", temperature=0)
        if model == "groq"
        else ChatOpenAI(model="gpt-4.1", temperature=0)
    )


def create_agent(
    llm: BaseChatModel, tools: List[Any], prompt_name: str, name: str
) -> AgentExecutor:
    return create_react_agent(
        model=llm, tools=tools, prompt=load_prompt(prompt_name), name=name
    )


def create_supervisor_agent(llm: BaseChatModel) -> AgentExecutor:
    assign_to_research_agent = create_handoff_tool(
        agent_name="research_agent",
        description="Assign task to a researcher agent.",
    )

    assign_to_math_agent = create_handoff_tool(
        agent_name="math_agent",
        description="Assign task to a math agent.",
    )

    return create_agent(
        llm=llm,
        tools=[assign_to_research_agent, assign_to_math_agent],
        prompt_name="supervisor_prompt",
        name="supervisor",
    )


def create_workflow() -> Callable:
    llm = create_llm()

    research_agent = create_agent(
        llm=llm,
        tools=[retriever_tool, internet_search, wiki_search, arxiv_search],
        prompt_name="web_research_prompt",
        name="research_agent",
    )

    math_agent = create_agent(
        llm=llm,
        tools=[add, sub, mult, div, mod],
        prompt_name="math_prompt",
        name="math_agent",
    )

    supervisor_agent = create_supervisor_agent(llm)

    workflow = StateGraph(MessagesState)

    workflow.add_node(
        supervisor_agent, destinations=("research_agent", "math_agent", END)
    )
    workflow.add_node(research_agent)
    workflow.add_node(math_agent)
    workflow.add_edge(START, "supervisor")
    workflow.add_edge("research_agent", "supervisor")
    workflow.add_edge("math_agent", "supervisor")

    return workflow.compile()


class BasicAgent:
    def __init__(self) -> None:
        print("BasicAgent initialized.")
        self.graph = create_workflow()

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")

        initial_messages = [HumanMessage(content=question)]
        final_messages = None

        for chunk in self.graph.stream({"messages": initial_messages}):
            pretty_print_messages(chunk)
            final_messages = chunk

        if final_messages is None:
            raise RuntimeError("No messages were generated during processing")

        return final_messages["supervisor"]["messages"][-1].content

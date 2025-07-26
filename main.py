from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph, END
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from uipath_langchain.chat.models import UiPathAzureChatOpenAI
import os

class GraphState(BaseModel):
    topic: str

class GraphOutput(BaseModel):
    report: str

async def generate_report(state: GraphState) -> GraphOutput:
    llm_model = UiPathAzureChatOpenAI(
        model="gpt-4o-2024-08-06",
        temperature=0,
        max_tokens=4000,
        timeout=30,
        max_retries=2,
        # other params...
    )
    
    system_prompt = "You are a report generator. Please provide a brief report based on the given topic."
    output = await llm_model.ainvoke([SystemMessage(system_prompt), HumanMessage(state.topic)])
    return GraphOutput(report=output.content)

builder = StateGraph(GraphState, output=GraphOutput)

builder.add_node("generate_report", generate_report)

builder.add_edge(START, "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()

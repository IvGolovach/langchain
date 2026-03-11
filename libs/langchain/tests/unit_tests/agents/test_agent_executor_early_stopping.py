"""Regression tests for runnable-agent early stopping."""

from __future__ import annotations

from textwrap import dedent

import pytest
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from langchain_classic.agents import AgentExecutor, create_react_agent
from tests.unit_tests.agents.test_agent import FakeListLLM


def _build_react_prompt() -> PromptTemplate:
    return PromptTemplate.from_template(
        dedent(
            """
            Answer the following questions as best you can.
            You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}
            """,
        ).strip(),
    )


def _build_executor(responses: list[str]) -> AgentExecutor:
    tool = Tool(
        name="Search",
        func=lambda value: f"Result for {value}",
        description="Useful for searching",
    )
    runnable = create_react_agent(
        FakeListLLM(responses=responses),
        [tool],
        _build_react_prompt(),
    )
    return AgentExecutor(
        agent=runnable,
        tools=[tool],
        max_iterations=1,
        early_stopping_method="generate",
    )


def test_runnable_agent_generate_on_early_stop_returns_final_answer() -> None:
    executor = _build_executor(
        [
            "Thought: I should search\nAction: Search\nAction Input: weather",
            "Thought: I now know the final answer\nFinal Answer: sunny",
        ],
    )

    result = executor.invoke({"input": "What is the weather?"})

    assert result["output"] == "sunny"


@pytest.mark.anyio
async def test_runnable_agent_generate_on_early_stop_returns_final_answer_async() -> (
    None
):
    executor = _build_executor(
        [
            "Thought: I should search\nAction: Search\nAction Input: weather",
            "Thought: I now know the final answer\nFinal Answer: sunny",
        ],
    )

    result = await executor.ainvoke({"input": "What is the weather?"})

    assert result["output"] == "sunny"


def test_runnable_agent_generate_on_early_stop_falls_back_to_raw_output() -> None:
    executor = _build_executor(
        [
            "Thought: I should search\nAction: Search\nAction Input: weather",
            "Thought: I should search again\nAction: Search\nAction Input: forecast",
        ],
    )

    result = executor.invoke({"input": "What is the weather?"})

    assert (
        result["output"]
        == "Thought: I should search again\nAction: Search\nAction Input: forecast"
    )

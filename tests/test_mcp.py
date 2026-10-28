"""The FastMCP server really exposes and runs its tools."""

import asyncio

import pytest

pytest.importorskip("fastmcp", reason="fastmcp is an optional extra")

from fastmcp import Client          # noqa: E402

from mcp_server import build_server  # noqa: E402


def test_tools_are_registered():
    async def scenario():
        async with Client(build_server()) as client:
            return [tool.name for tool in await client.list_tools()]

    names = asyncio.run(scenario())
    assert "search_papers_tool" in names
    assert "ask_research_question_tool" in names
    assert "ingest_arxiv_tool" in names


def test_search_tool_returns_hits():
    async def scenario():
        async with Client(build_server()) as client:
            return await client.call_tool("search_papers_tool",
                                          {"query": "graph rag global questions",
                                           "top_k": 2})

    result = asyncio.run(scenario())
    text = result.content[0].text
    assert "paper_id" in text

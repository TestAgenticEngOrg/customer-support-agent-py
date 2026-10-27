import logging
import os
from contextlib import AsyncExitStack

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

log = logging.getLogger("customer-support-agent")

# Agent Manager injects <PROXY_ID>_URL and <PROXY_ID>_API_KEY for every MCP proxy attached to the agent.
# The LLM provider uses the same naming, so its prefix is skipped.
SKIP_PREFIXES = {"ANTHROPIC"}
TIMEOUT_SECONDS = 30


def discover_proxies() -> dict[str, tuple[list[str], str]]:
    proxies = {}
    for name, value in os.environ.items():
        prefix = name.removesuffix("_URL")
        if prefix == name or prefix in SKIP_PREFIXES:
            continue
        key = os.environ.get(f"{prefix}_API_KEY", "").strip()
        # One variable can hold several comma-separated URLs. Empty values mean "not configured here".
        urls = [u.strip() for u in value.split(",") if u.strip()]
        if key and urls:
            proxies[prefix] = (urls, key)
    return proxies


class McpTools:
    """Connects to every attached MCP proxy for the length of one agent run and exposes their tools."""

    def __init__(self, reserved_names: set[str]):
        self.schemas: list[dict] = []
        self._reserved = set(reserved_names)
        self._routes: dict[str, ClientSession] = {}
        self._stack = AsyncExitStack()

    async def __aenter__(self) -> "McpTools":
        for proxy, (urls, key) in discover_proxies().items():
            for url in urls:
                try:
                    await self._connect(url, key)
                except Exception:
                    # One unreachable proxy should not take the whole agent down.
                    log.exception("Skipping MCP proxy %s at %s", proxy, url)
        return self

    async def __aexit__(self, *exc) -> None:
        await self._stack.aclose()

    async def _connect(self, url: str, key: str) -> None:
        stack = AsyncExitStack()
        try:
            http = await stack.enter_async_context(
                httpx2.AsyncClient(headers={"API-Key": key}, timeout=TIMEOUT_SECONDS)
            )
            read, write = await stack.enter_async_context(streamable_http_client(url, http_client=http))
            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            listed = await session.list_tools()
        except BaseException:
            await stack.aclose()
            raise
        self._stack.push_async_callback(stack.aclose)
        for tool in listed.tools:
            if tool.name in self._reserved:
                log.warning("Ignoring MCP tool %s from %s: name already in use", tool.name, url)
                continue
            self._reserved.add(tool.name)
            self._routes[tool.name] = session
            self.schemas.append(
                {"name": tool.name, "description": tool.description or "", "input_schema": tool.input_schema}
            )

    def has(self, name: str) -> bool:
        return name in self._routes

    async def call(self, name: str, args: dict) -> tuple[str, bool]:
        try:
            result = await self._routes[name].call_tool(name, args)
        except Exception as e:
            log.exception("MCP tool %s failed", name)
            return f"Tool {name} failed: {e}", True
        text = "\n".join(c.text for c in getattr(result, "content", []) if c.type == "text")
        return text or "(no output)", bool(getattr(result, "is_error", False))

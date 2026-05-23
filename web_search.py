import os

import httpx

TAVILY_URL = "https://api.tavily.com/search"
MAX_RESULTS = 5
TIMEOUT_SECONDS = 30.0


class WebSearchError(RuntimeError):
    pass


async def search(query: str) -> dict:
    """Run a Tavily search. Returns {"answer", "results": [{title, url, content}]}.

    Raises WebSearchError on any failure (missing key, network, non-200, bad shape).
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise WebSearchError("TAVILY_API_KEY is not set")

    payload = {
        "api_key": api_key,
        "query": query,
        "search_depth": "basic",
        "include_answer": True,
        "max_results": MAX_RESULTS,
    }

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
            r = await client.post(TAVILY_URL, json=payload)
    except httpx.HTTPError as e:
        raise WebSearchError(f"network error: {e}") from e

    if r.status_code != 200:
        raise WebSearchError(f"HTTP {r.status_code}: {r.text[:200]}")

    try:
        data = r.json()
    except ValueError as e:
        raise WebSearchError(f"invalid JSON from Tavily: {e}") from e

    return {
        "answer": data.get("answer") or "",
        "results": [
            {
                "title": (item.get("title") or "")[:200],
                "url": item.get("url") or "",
                "content": (item.get("content") or "")[:1500],
            }
            for item in (data.get("results") or [])
        ],
    }


def format_for_prompt(query: str, search: dict) -> str:
    """Render the search payload as a context block for the council prompt."""
    parts = [f"\n=== Web search results for: \"{query}\" ===\n"]
    if search.get("answer"):
        parts.append(f"\nSummary of findings:\n{search['answer']}\n")
    if search.get("results"):
        parts.append("\nTop sources:\n")
        for i, r in enumerate(search["results"], 1):
            parts.append(f"\n[{i}] {r['title']}")
            if r.get("url"):
                parts.append(f"\n    URL: {r['url']}")
            if r.get("content"):
                parts.append(f"\n    {r['content']}")
            parts.append("\n")
    parts.append(
        "\n=== End of web search context ===\n"
        "When relevant, draw on this material. Be honest if the search didn't "
        "fully answer the question — don't invent facts to fill gaps.\n"
    )
    return "".join(parts)

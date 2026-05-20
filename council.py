import asyncio
import json
import os
from typing import AsyncIterator

from openai import AsyncOpenAI

MODEL = "deepseek-chat"  # or "deepseek-reasoner" for R1 reasoning model (slower)
MAX_TOKENS = 4096
NUM_CRITIQUE_ROUNDS = 2
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

PERSONAS = {
    "Pragmatist": (
        "You are the Pragmatist on a three-member deliberation council. "
        "You focus on what actually works in practice — concrete examples, real-world "
        "constraints, battle-tested approaches. You're skeptical of theoretical purity "
        "that ignores implementation realities. Keep your answers grounded and specific. "
        "Avoid hedging; commit to a position."
    ),
    "Skeptic": (
        "You are the Skeptic on a three-member deliberation council. "
        "You question assumptions and look for flaws in reasoning. You search for "
        "counterexamples, hidden costs, edge cases, and unstated premises. You push back "
        "when others are too confident — but you're not a contrarian for sport. Your dissent "
        "should illuminate. When you find a real weakness, name it precisely."
    ),
    "Theorist": (
        "You are the Theorist on a three-member deliberation council. "
        "You think in principles, frameworks, and first principles. You're drawn to "
        "underlying structure and connect specific cases to general patterns. Don't get "
        "lost in pure abstraction — ground your principles in the concrete question. "
        "Name the framework you're using; don't smuggle it in."
    ),
}

AGENT_NAMES = list(PERSONAS.keys())


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=DEEPSEEK_BASE_URL,
    )


async def _stream_agent(
    client: AsyncOpenAI,
    agent_name: str,
    system: str,
    user_msg: str,
    round_num: int,
    queue: asyncio.Queue,
) -> tuple[str, str]:
    """Stream one agent's response, pushing deltas to the queue. Returns (name, full_text)."""
    await queue.put({"type": "agent_start", "agent": agent_name, "round": round_num})
    chunks: list[str] = []
    stream = await client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        stream=True,
    )
    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            chunks.append(delta)
            await queue.put(
                {"type": "agent_delta", "agent": agent_name, "round": round_num, "text": delta}
            )
    await queue.put({"type": "agent_done", "agent": agent_name, "round": round_num})
    return agent_name, "".join(chunks)


async def _run_round(
    client: AsyncOpenAI,
    round_num: int,
    prompts: dict[str, tuple[str, str]],
    queue: asyncio.Queue,
) -> dict[str, str]:
    """Run all agents for one round in parallel, streaming to queue."""
    tasks = [
        asyncio.create_task(
            _stream_agent(client, name, system, user_msg, round_num, queue)
        )
        for name, (system, user_msg) in prompts.items()
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)


def _format_transcript(
    question: str,
    rounds: dict[int, dict[str, str]],
    current_agent: str | None,
) -> str:
    parts = [f"Original question:\n\n{question}\n"]
    for r in sorted(rounds.keys()):
        header = "Opening Answers" if r == 0 else f"Critique Round {r}"
        parts.append(f"\n=== {header} ===\n")
        for name, text in rounds[r].items():
            marker = " (YOU — your prior response)" if name == current_agent else ""
            parts.append(f"\n--- {name}{marker} ---\n{text}\n")
    return "".join(parts)


async def run_council(question: str) -> AsyncIterator[str]:
    """Drive the full council deliberation, yielding SSE-formatted events."""
    client = _client()
    queue: asyncio.Queue = asyncio.Queue()

    async def producer() -> None:
        try:
            rounds: dict[int, dict[str, str]] = {}

            opening_prompts = {
                name: (
                    PERSONAS[name],
                    f"Question:\n\n{question}\n\n"
                    "Provide your independent answer. Don't qualify it to death — "
                    "commit to a position you can defend.",
                )
                for name in AGENT_NAMES
            }
            rounds[0] = await _run_round(client, 0, opening_prompts, queue)
            await queue.put({"type": "round_complete", "round": 0})

            for r in range(1, NUM_CRITIQUE_ROUNDS + 1):
                critique_prompts = {}
                for name in AGENT_NAMES:
                    transcript = _format_transcript(question, rounds, current_agent=name)
                    critique_prompts[name] = (
                        PERSONAS[name],
                        f"{transcript}\n\n"
                        "Now: (1) critique the other council members' most recent positions — "
                        "be specific about where they're wrong or weak; (2) state your revised "
                        "position. If their arguments shifted your view, say so and update. "
                        "Don't dig in defensively just to preserve your prior take.",
                    )
                rounds[r] = await _run_round(client, r, critique_prompts, queue)
                await queue.put({"type": "round_complete", "round": r})

            await queue.put({"type": "synthesis_start"})
            full_transcript = _format_transcript(question, rounds, current_agent=None)
            synth_stream = await client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a neutral synthesizer reading a deliberation transcript. "
                            "Your job is to extract signal from the back-and-forth — agreement, "
                            "real disagreement, and a final integrated answer. You did not "
                            "participate in the deliberation; you have no prior position."
                        ),
                    },
                    {
                        "role": "user",
                        "content": (
                            f"{full_transcript}\n\n"
                            "Produce a synthesis with three sections, in this order:\n\n"
                            "**Points of agreement** — where did the council actually converge? "
                            "Skip surface-level platitudes; name the substantive agreements.\n\n"
                            "**Remaining disagreements** — where do real differences persist, "
                            "and what's at stake in each? If a disagreement is purely semantic, "
                            "say so.\n\n"
                            "**Final synthesized answer** — your best integrated answer to the "
                            "original question. Draw on the strongest threads from the deliberation. "
                            "Be direct."
                        ),
                    },
                ],
                stream=True,
            )
            async for chunk in synth_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    await queue.put({"type": "synthesis_delta", "text": delta})

            await queue.put({"type": "done"})
        except Exception as e:
            await queue.put({"type": "error", "message": f"{type(e).__name__}: {e}"})
        finally:
            await queue.put(None)

    task = asyncio.create_task(producer())
    try:
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"
    finally:
        if not task.done():
            task.cancel()

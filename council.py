import asyncio
import json
import os
from typing import AsyncIterator

from openai import AsyncOpenAI

import db

MODEL = "deepseek-chat"
MAX_TOKENS = 4096
NUM_CRITIQUE_ROUNDS = 2
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

DEFAULT_PERSONAS: list[dict] = [
    {
        "name": "Pragmatist",
        "system": (
            "You are the Pragmatist on a three-member deliberation council. "
            "You focus on what actually works in practice — concrete examples, real-world "
            "constraints, battle-tested approaches. You're skeptical of theoretical purity "
            "that ignores implementation realities. Keep your answers grounded and specific. "
            "Avoid hedging; commit to a position."
        ),
    },
    {
        "name": "Skeptic",
        "system": (
            "You are the Skeptic on a three-member deliberation council. "
            "You question assumptions and look for flaws in reasoning. You search for "
            "counterexamples, hidden costs, edge cases, and unstated premises. You push back "
            "when others are too confident — but you're not a contrarian for sport. Your dissent "
            "should illuminate. When you find a real weakness, name it precisely."
        ),
    },
    {
        "name": "Theorist",
        "system": (
            "You are the Theorist on a three-member deliberation council. "
            "You think in principles, frameworks, and first principles. You're drawn to "
            "underlying structure and connect specific cases to general patterns. Don't get "
            "lost in pure abstraction — ground your principles in the concrete question. "
            "Name the framework you're using; don't smuggle it in."
        ),
    },
]


def resolve_personas(stored: list[dict]) -> list[dict]:
    """Use stored personas if 3 are present and well-formed; otherwise fall back to defaults."""
    if (
        isinstance(stored, list)
        and len(stored) == 3
        and all(isinstance(p, dict) and p.get("name") and p.get("system") for p in stored)
    ):
        return stored
    return DEFAULT_PERSONAS

SYNTH_SYSTEM = (
    "You are a neutral synthesizer reading a deliberation transcript. "
    "Your job is to extract signal from the back-and-forth — agreement, real "
    "disagreement, and a final integrated answer. You did not participate in "
    "the deliberation; you have no prior position."
)


def _client() -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url=DEEPSEEK_BASE_URL,
    )


def _format_prior_attachments(history: list[dict]) -> str:
    """Re-share any files attached in earlier turns of this conversation."""
    all_attachments = [a for t in history for a in t.get("attachments", [])]
    if not all_attachments:
        return ""
    parts = ["\n=== Files shared earlier in this conversation ===\n"]
    for a in all_attachments:
        parts.append(f"\n--- {a['filename']} ---\n{a.get('content', '')}\n")
    parts.append("\n=== End of shared files ===\n")
    return "".join(parts)


def _format_history(history: list[dict]) -> str:
    if not history:
        return ""
    parts = ["\n=== Prior turns in this conversation ===\n"]
    for i, t in enumerate(history, 1):
        parts.append(f"\nTurn {i} — User asked: {t['user_question']}\n")
        if t.get("synthesis"):
            parts.append(f"\nCouncil's synthesized answer:\n{t['synthesis']}\n")
    parts.append("\n=== End of prior turns ===\n")
    return "".join(parts)


def _format_attachments(attachments: list[dict]) -> str:
    if not attachments:
        return ""
    parts = ["\n=== Files attached to this question ===\n"]
    for a in attachments:
        parts.append(f"\n--- {a['filename']} ---\n{a['content']}\n")
    parts.append("\n=== End of attached files ===\n")
    return "".join(parts)


def _format_round_transcript(rounds: dict[int, dict[str, str]], current_agent: str | None) -> str:
    parts = []
    for r in sorted(rounds.keys()):
        header = "Opening Answers" if r == 0 else f"Critique Round {r}"
        parts.append(f"\n=== {header} ===\n")
        for name, text in rounds[r].items():
            marker = " (YOU — your prior response in this turn)" if name == current_agent else ""
            parts.append(f"\n--- {name}{marker} ---\n{text}\n")
    return "".join(parts)


def _build_opening_prompt(history: list[dict], attachments: list[dict], question: str) -> str:
    return (
        _format_prior_attachments(history)
        + _format_history(history)
        + _format_attachments(attachments)
        + f"\n=== Current question ===\n\n{question}\n\n"
        + "Provide your independent answer to the current question. "
        + "Don't qualify it to death — commit to a position you can defend. "
        + (
            "Build on (or push back against) the prior turns; don't pretend they didn't happen."
            if history
            else ""
        )
    )


def _build_critique_prompt(
    history: list[dict],
    attachments: list[dict],
    question: str,
    rounds: dict[int, dict[str, str]],
    current_agent: str,
) -> str:
    return (
        _format_prior_attachments(history)
        + _format_history(history)
        + _format_attachments(attachments)
        + f"\n=== Current question ===\n\n{question}\n"
        + _format_round_transcript(rounds, current_agent)
        + "\n\nNow: (1) critique the other council members' most recent positions — "
        + "be specific about where they're wrong or weak; (2) state your revised "
        + "position. If their arguments shifted your view, say so and update. "
        + "Don't dig in defensively just to preserve your prior take."
    )


def _build_synth_prompt(
    history: list[dict],
    attachments: list[dict],
    question: str,
    rounds: dict[int, dict[str, str]],
) -> str:
    return (
        _format_prior_attachments(history)
        + _format_history(history)
        + _format_attachments(attachments)
        + f"\n=== Current question ===\n\n{question}\n"
        + _format_round_transcript(rounds, current_agent=None)
        + "\n\nProduce a synthesis with three sections, in this order:\n\n"
        + "**Points of agreement** — where did the council actually converge? "
        + "Skip surface-level platitudes; name the substantive agreements.\n\n"
        + "**Remaining disagreements** — where do real differences persist, "
        + "and what's at stake in each? If a disagreement is purely semantic, "
        + "say so.\n\n"
        + "**Final synthesized answer** — your best integrated answer to the "
        + "current question. Draw on the strongest threads from the deliberation. "
        + "Be direct."
    )


def _make_title(question: str, max_len: int = 60) -> str:
    title = question.strip().split("\n")[0].strip()
    if len(title) > max_len:
        title = title[: max_len - 1].rstrip() + "…"
    return title or "Untitled"


async def _stream_agent(
    client: AsyncOpenAI,
    agent_name: str,
    slot: int,
    system: str,
    user_msg: str,
    round_num: int,
    queue: asyncio.Queue,
) -> tuple[str, str]:
    await queue.put(
        {"type": "agent_start", "agent": agent_name, "slot": slot, "round": round_num}
    )
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
                {
                    "type": "agent_delta",
                    "agent": agent_name,
                    "slot": slot,
                    "round": round_num,
                    "text": delta,
                }
            )
    await queue.put(
        {"type": "agent_done", "agent": agent_name, "slot": slot, "round": round_num}
    )
    return agent_name, "".join(chunks)


async def _run_round(
    client: AsyncOpenAI,
    round_num: int,
    personas: list[dict],
    user_msg_for: dict[str, str],
    queue: asyncio.Queue,
) -> dict[str, str]:
    tasks = [
        asyncio.create_task(
            _stream_agent(
                client,
                p["name"],
                slot,
                p["system"],
                user_msg_for[p["name"]],
                round_num,
                queue,
            )
        )
        for slot, p in enumerate(personas)
    ]
    results = await asyncio.gather(*tasks)
    return dict(results)


async def run_council(
    conversation_id: int,
    question: str,
    attachments: list[dict],
) -> AsyncIterator[str]:
    """Stream the deliberation, then persist the turn. Yields SSE-formatted events."""
    client = _client()
    queue: asyncio.Queue = asyncio.Queue()
    history = db.get_history(conversation_id)
    is_first_turn = len(history) == 0
    personas = resolve_personas(db.get_personas(conversation_id))

    async def producer() -> None:
        synthesis_chunks: list[str] = []
        try:
            rounds: dict[int, dict[str, str]] = {}

            opening_user_msg = _build_opening_prompt(history, attachments, question)
            opening_msg_by_name = {p["name"]: opening_user_msg for p in personas}
            rounds[0] = await _run_round(client, 0, personas, opening_msg_by_name, queue)
            await queue.put({"type": "round_complete", "round": 0})

            for r in range(1, NUM_CRITIQUE_ROUNDS + 1):
                critique_msg_by_name = {
                    p["name"]: _build_critique_prompt(
                        history, attachments, question, rounds, p["name"]
                    )
                    for p in personas
                }
                rounds[r] = await _run_round(client, r, personas, critique_msg_by_name, queue)
                await queue.put({"type": "round_complete", "round": r})

            await queue.put({"type": "synthesis_start"})
            synth_stream = await client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system", "content": SYNTH_SYSTEM},
                    {
                        "role": "user",
                        "content": _build_synth_prompt(history, attachments, question, rounds),
                    },
                ],
                stream=True,
            )
            async for chunk in synth_stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    synthesis_chunks.append(delta)
                    await queue.put({"type": "synthesis_delta", "text": delta})

            synthesis_text = "".join(synthesis_chunks)
            db.save_turn(
                conversation_id,
                user_question=question,
                attachments=attachments,
                responses={str(r): rounds[r] for r in rounds},
                synthesis=synthesis_text,
            )
            if is_first_turn:
                title = _make_title(question)
                db.update_title(conversation_id, title)
                await queue.put({"type": "title_updated", "title": title})

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

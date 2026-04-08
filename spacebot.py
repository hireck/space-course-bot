"""
Space Missions and Space Technology — Zulip RAG Bot

Each Zulip stream/topic pair gets its own conversation history window.
The bot responds whenever it is @-mentioned in any stream or DM.

Dependencies:
    pip install zulip langchain langchain-openai openai weaviate-client \
                sentence-transformers

Environment variables required:
    OPENAI_API_KEY   – OpenAI API key
    ZULIP_RC         – Path to the zuliprc credentials file
                       (defaults to ~/.zuliprc if unset)
"""

import json
import os
import re
from collections import defaultdict, deque

from openai import OpenAI
import zulip
import weaviate
import weaviate.classes.query as wq
from langchain_openai import ChatOpenAI
from sentence_transformers import CrossEncoder, SentenceTransformer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ZULIP_RC          = os.environ.get("ZULIP_RC", os.path.expanduser("~/.zuliprc"))
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY")
HISTORY_WINDOW    = 4          # number of previous messages kept per topic
RETRIEVAL_LIMIT   = 20         # candidates fetched from Weaviate
RERANK_TOP_K      = 7          # top documents after cross-encoder reranking
RERANK_FALLBACK_K = 2          # fallback if no document scores > 0

#openai.api_key = OPENAI_API_KEY

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

system_prompt = """
You are EuroLLM --- an AI assistant specialized in European languages that provides safe, educational and helpful answers. You operate as a Zulip bot.

You are an expert on Space Missions and Space Technology.
Your task is to assist students of physics and astronomy to learn about the
topic and work towards designing their own space mission.

Write all mathematical equations, symbols, etc. in LaTeX, using Zulip's LaTeX formatting funcionality. Do not include tags.

"""

RAG_TEMPLATE = """

Use the following pieces of retrieved information to answer the user's question.
Be helpful. Volunteer additional information where relevant, but keep it concise.
Do not make up answers that are not supported by the retrieved information.
If the retrieved documents do not contain sufficient information to answer the
question, say so.

Include references in your answer to the documents you used, to indicate where
the information comes from. The documents have a field called "chunk_number".
Use those numbers to refer to them. Wrap the number in <cite>...</cite> tags,
e.g. '<cite>23</cite>'. Put each cited document chunk number between its own
pair of tags. Do not cite other sources than the provided documents. Do not
list the sources below your answer.

Retrieved information:
{context}

Preceding conversation:
{conversation}

Question: {question}
Helpful Answer:"""

CONTEXTUALIZING_TEMPLATE = """
Given a chat history and the latest user question which might reference context
in the chat history, formulate a standalone question in English which can be understood
without the chat history. The overall topic is Space Missions and Space
Technology. Do NOT answer the question, just reformulate it and/or translate it to English if needed and
otherwise return it as is.

Chat history:
{history}

Latest user question:
{question}

Standalone version of the question:
"""

# ---------------------------------------------------------------------------
# Model loading (done once at startup)
# ---------------------------------------------------------------------------

print("Loading language model …")
#gpt4 = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

#client = OpenAI(
#    base_url="http://localhost:8081/v1",
#    api_key="not-needed"  # required by the SDK but ignored by llama.cpp
#)
#In Docker:
client = OpenAI(base_url="http://llamacpp:8081/v1", api_key="not-needed")

print("Loading embedding model …")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2")

print("Loading cross-encoder …")
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# ---------------------------------------------------------------------------
# Weaviate
# ---------------------------------------------------------------------------

print("Connecting to Weaviate …")
#weaviate_client = weaviate.connect_to_local()
#In Docker:
weaviate_client = weaviate.connect_to_local(host="weaviate", port=8080)
assert weaviate_client.is_ready(), "Weaviate is not ready!"
chunks_collection = weaviate_client.collections.get("DocumentChunk")

# ---------------------------------------------------------------------------
# Conversation history
# Per-topic ring buffer: key = (stream_id, topic) or ("dm", user_id)
# ---------------------------------------------------------------------------

history: dict[tuple, deque] = defaultdict(lambda: deque(maxlen=HISTORY_WINDOW))


def history_key(event: dict) -> tuple:
    msg = event["message"]
    if msg["type"] == "stream":
        return ("stream", msg["stream_id"], msg["subject"])
    return ("dm", msg["sender_id"])


def format_history(key: tuple) -> str:
    return "\n".join(
        f"{role}: {text}" for role, text in history[key]
    )


def push_history(key: tuple, role: str, text: str) -> None:
    history[key].append((role, text))


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

META_FIELDS = ["chunk_number", "book_title", "year", "section_headers"]


def vectorize(text: str):
    return embed_model.encode([text])[0]


def retrieve_docs(query: str) -> list:
    query_vector = vectorize(query)
    response = chunks_collection.query.near_vector(
        near_vector=query_vector,
        limit=RETRIEVAL_LIMIT,
        return_metadata=wq.MetadataQuery(distance=True),
    )
    retrieved = response.objects

    cross_inp    = [[query, d.properties["page_content"]] for d in retrieved]
    cross_scores = cross_encoder.predict(cross_inp)

    scored       = list(zip(cross_scores, retrieved))
    positive     = [(s, d) for s, d in scored if s > 0]

    if positive:
        reranked = sorted(positive, key=lambda t: t[0], reverse=True)
        return [d for _, d in reranked[:RERANK_TOP_K]]
    else:
        reranked = sorted(scored, key=lambda t: t[0], reverse=True)
        return [d for _, d in reranked[:RERANK_FALLBACK_K]]


def format_docs(docs: list) -> str:
    if not docs:
        return "No relevant documents were found."
    parts = []
    for d in docs:
        meta = {f: d.properties.get(f) for f in META_FIELDS}
        parts.append(json.dumps(meta, indent=4) + "\n" + d.properties["page_content"])
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Citation / source helpers
# ---------------------------------------------------------------------------

def _dedup(seq: list) -> list:
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def used_sources(answer: str) -> tuple[str, list[str]]:
    """Replace <cite>N</cite> tags with sequential [1], [2], … references."""
    numbers = _dedup(
        s.strip() for s in re.findall(r"<cite>(.*?)</cite>", answer, re.DOTALL)
    )
    for idx, num in enumerate(numbers, start=1):
        answer = re.sub(rf"<cite> ?{num} ?</cite>", f"[{idx}]", answer)
    return answer, numbers


def build_sources_text(docs: list, source_numbers: list[str]) -> str:
    if not docs or not source_numbers:
        return (
            "_The information presented here does not explicitly reference the "
            "retrieved sources. Extra caution with respect to accuracy may be "
            "in order._"
        )
    num2doc = {str(d.properties["chunk_number"]): d for d in docs}
    lines = []
    for idx, num in enumerate(source_numbers, start=1):
        doc = num2doc.get(num)
        if doc is None:
            continue
        title   = doc.properties.get("book_title", "Unknown title")
        headers = doc.properties.get("section_headers")
        section = (
            "Section: " + ", ".join(headers)
            if headers
            else doc.properties["page_content"][:60] + "…"
        )
        lines.append(f"[{idx}] *{title}* — {section}")
    return "\n".join(lines)


def fix_math_formatting(text: str) -> str:
    """
    Post-process LLM output to fix math formatting for Zulip.
    
    Target conventions:
      - Inline math: $$...$$
      - Display math: ```math ... ```
    """

    # 1. Convert ```math blocks first (before other substitutions touch their contents)
    #    Already-correct blocks are left alone; this is mostly a no-op for those.
    #    But normalise ``` math (with a space) -> ```math
    text = re.sub(r'```\s+math', '```math', text)

    # 2. Convert \[...\] display math to ```math blocks
    def replace_display(m):
        inner = m.group(1).strip()
        return f'```math\n{inner}\n```'
    text = re.sub(r'\\\[\s*(.*?)\s*\\\]', replace_display, text, flags=re.DOTALL)

    # 3. Convert $$...$$ on its own line to ```math blocks
    #    (model sometimes does this instead of inline)
    def replace_display_dollars(m):
        inner = m.group(1).strip()
        return f'```math\n{inner}\n```'
    text = re.sub(r'(?m)^\s*\$\$\s*(.*?)\s*\$\$\s*$', replace_display_dollars, text, flags=re.DOTALL)

    # 4. Convert \(...\) inline math to $$...$$
    text = re.sub(r'\\\(\s*(.*?)\s*\\\)', lambda m: f'$${m.group(1)}$$', text, flags=re.DOTALL)

    # 5. Convert single $...$ inline math to $$...$$
    #    Use a negative lookbehind/ahead to avoid touching already-double-dollar signs
    text = re.sub(r'(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)', lambda m: f'$${m.group(1)}$$', text, flags=re.DOTALL)

    # 6. Strip extra whitespace inside $$ ... $$ inline math
    #text = re.sub(r'\$\$\s+(.*?)\s+\$\$', lambda m: f'$${m.group(1)}$$', text, flags=re.DOTALL)

    return text

# ---------------------------------------------------------------------------
# Core RAG pipeline
# ---------------------------------------------------------------------------

def answer_question(user_input: str, conv_key: tuple, message: dict, bot_handler) -> str:
    """Run the full RAG pipeline and return a formatted Zulip message."""
    prev_conv = format_history(conv_key)

    # --- Helper to send or update a status message ---
    status_msg_id = None

    def send_status(text: str) -> None:
        nonlocal status_msg_id
        if status_msg_id is None:
            # Send the first status message and remember its ID
            result = bot_handler.send_reply(message, text)
            if isinstance(result, dict) and "id" in result:
                status_msg_id = result["id"]
        else:
            # Update the existing status message in place
            bot_handler.update_message({
                "message_id": status_msg_id,
                "content": text,
            })

    # 1. Start typing indicator + first status
    bot_handler.send_typing_indicator(message)
    send_status("🧠 Analysing your question…")

    # 2. Contextualise the query against history
    ctx_prompt = CONTEXTUALIZING_TEMPLATE.format(
        history=prev_conv, question=user_input
    )
    ctx_messages = [
        {"role": "system", "content": "You are EuroLLM"},
        {"role": "user", "content": ctx_prompt}
    ]
    ctx_response = client.chat.completions.create(
        model="utter-project/EuroLLM-22B-Instruct-2512",
        messages=ctx_messages
    )
    search_query = ctx_response.choices[0].message.content
    print(f"[search query] {search_query}")

    # 3. Retrieve & rerank
    send_status("📚 Retrieving relevant documents…")
    docs = retrieve_docs(search_query)

    # 4. Generate answer
    send_status("✍️ Generating answer…")
    full_prompt = RAG_TEMPLATE.format(
        context=format_docs(docs),
        question=user_input,
        conversation=prev_conv,
    )
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user", "content": full_prompt
        },
    ]
    response = client.chat.completions.create(
        model="utter-project/EuroLLM-22B-Instruct-2512",
        messages=messages
    )
    raw_answer = response.choices[0].message.content
    ai_answer, source_numbers = used_sources(raw_answer)
    sources_text = build_sources_text(docs, source_numbers)

    # 5. Update history
    push_history(conv_key, "human", user_input)
    push_history(conv_key, "ai", ai_answer)

    # 6. Format final message
    reply = fix_math_formatting(ai_answer)
    if sources_text:
        reply += f"\n\n---\n**Sources**\n{sources_text}"

    # 7. Replace the status message with the final answer
    if status_msg_id is not None:
        bot_handler.update_message({
            "message_id": status_msg_id,
            "content": reply,
        })
        return None  # Signal that the reply has already been sent
    return reply


# ---------------------------------------------------------------------------
# Zulip bot handler
# ---------------------------------------------------------------------------

class SpaceBotHandler:
    """
    Zulip event handler.

    The bot reacts to any message in which it is @-mentioned (stream) or to
    any direct message sent to it.
    """

    def usage(self) -> str:
        return (
            "I am the Space Missions and Space Technology chatbot. "
            "@-mention me with a question and I will answer using the "
            "course textbooks."
        )

    def handle_message(self, message: dict, bot_handler) -> None:
        sender_email = message["sender_email"]

        # Ignore messages sent by the bot itself
        if bot_handler.email == sender_email:
            return

        # Strip the @-mention prefix so it is not part of the query
        content = re.sub(r"@\*\*[^*]+\*\*\s*", "", message["content"]).strip()
        if not content:
            bot_handler.send_reply(
                message,
                "Hi! Ask me anything about Space Missions and Space Technology.",
            )
            return

        # Build the conversation key
        if message["type"] == "stream":
            key = ("stream", message["stream_id"], message["subject"])
        else:
            key = ("dm", message["sender_id"])

        print(f"[message] from={sender_email} key={key} content={content!r}")

        try:
            reply = answer_question(content, key, message, bot_handler)
            # Only send if answer_question didn't already update in place
            if reply is not None:
                bot_handler.send_reply(message, reply)
        except Exception as exc:
            print(f"[error] {exc}")
            bot_handler.send_reply(
                message,
                "Oops — something went wrong while generating an answer. "
                "Please try again."
            )


# ---------------------------------------------------------------------------
# Entry point  (run directly, not via zulip-run-bot)
# ---------------------------------------------------------------------------

def main() -> None:
    zulip_client = zulip.Client(config_file=ZULIP_RC)
    bot_handler  = zulip_client

    def send_reply(message: dict, content: str) -> dict:
        if message["type"] == "stream":
            result = zulip_client.send_message({
                "type":    "stream",
                "to":      message["display_recipient"],
                "topic":   message["subject"],
                "content": content,
            })
        else:
            result = zulip_client.send_message({
                "type":    "private",
                "to":      [message["sender_email"]],
                "content": content,
            })
        return result  # Contains {"id": <message_id>, ...}

    def send_typing_indicator(message: dict) -> None:
        if message["type"] == "stream":
            zulip_client.call_endpoint(
                url="typing",
                method="POST",
                request={
                    "op":    "start",
                    "type":  "stream",
                    "to":    message["stream_id"],
                    "topic": message["subject"],
                },
            )
        else:
            zulip_client.call_endpoint(
                url="typing",
                method="POST",
                request={
                    "op":   "start",
                    "type": "direct",
                    "to":   [message["sender_id"]],
                },
            )

    def update_message(payload: dict) -> None:
        zulip_client.call_endpoint(
            url=f"messages/{payload['message_id']}",
            method="PATCH",
            request={"content": payload["content"]},
        )

    bot_handler.send_reply           = send_reply
    bot_handler.send_typing_indicator = send_typing_indicator
    bot_handler.update_message       = update_message
    bot_handler.email                = zulip_client.get_profile()["email"]

    handler = SpaceBotHandler()
    print(f"Bot running as {bot_handler.email} — waiting for messages …")

    def on_event(event: dict) -> None:
        if event.get("type") == "message":
            msg = event["message"]
            is_dm        = msg["type"] == "private"
            is_mentioned = any(
                flag in msg.get("flags", []) for flag in ("mentioned", "wildcard_mentioned")
            )
            if is_dm or is_mentioned:
                handler.handle_message(msg, bot_handler)

    zulip_client.call_on_each_event(
        on_event,
        event_types=["message"],
    )

if __name__ == "__main__":
    main()

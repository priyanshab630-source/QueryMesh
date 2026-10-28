"""A2A JSON-RPC endpoint.

One endpoint serves every agent: the `agent` field in the request (or the
`X-A2A-Agent` header) selects which card is being addressed. Supported methods:

    message/send   -> run the agent on the text part of the message
    tasks/get      -> read a task back by id
    agent/getCard  -> the card for the addressed agent

This is enough for another agent framework to discover the cards
(`GET /.well-known/agent.json`) and delegate work to QueryMesh.
"""

import uuid

from src.a2a import agent_cards

TASKS = {}


def _text_of(params: dict) -> str:
    message = params.get("message") or {}
    for part in message.get("parts", []):
        if part.get("kind", "text") == "text" and part.get("text"):
            return part["text"]
    return ""


def _run_until(agent: str, question: str) -> dict:
    """Run the pipeline and return the state produced by `agent`."""
    from src.agents import citation, grader, retriever, supervisor, synthesizer

    steps = {
        "supervisor": [supervisor],
        "retriever": [supervisor, retriever],
        "grader": [supervisor, retriever, grader],
        "synthesizer": [supervisor, retriever, grader, synthesizer],
        "citation": [supervisor, retriever, grader, synthesizer, citation],
    }
    state = {"question": question, "loops": 0, "trace": [], "meta": {}}
    for module in steps[agent]:
        state = module.run(state)
    return state


def _handle_agent(agent: str, text: str) -> str:
    if agent in ("querymesh", "orchestrator"):
        from src.agents.workflow import ask

        result = ask(text)
        sources = "; ".join(f"[{c['number']}] {c['title']}" for c in result["citations"])
        return f"{result['answer']}\n\nSources: {sources or 'none verified'}"

    state = _run_until(agent, text)
    if agent == "supervisor":
        return "Sub-questions: " + "; ".join(state.get("sub_questions", []))
    if agent == "retriever":
        return "Retrieved: " + "; ".join(
            f"{h['chunk']['paper_id']} ({h['score']:.4f})" for h in state.get("chunks", []))
    if agent == "grader":
        kept = state.get("chunks", [])
        best = kept[0].get("relevance", 0.0) if kept else 0.0
        return (f"Kept {len(kept)} chunk(s), rejected "
                f"{len(state.get('rejected', []))}; best relevance {best:.2f}")
    if agent == "synthesizer":
        return state.get("answer", "")
    return (f"Verified {len(state.get('citations', []))} citation(s); grounding "
            f"{state.get('meta', {}).get('grounding', 0.0):.2f}")


def handle(payload: dict) -> dict:
    """JSON-RPC 2.0 dispatcher. Returns the response object."""
    request_id = payload.get("id")
    method = payload.get("method", "")
    params = payload.get("params") or {}
    agent = params.get("agent") or payload.get("agent") or "querymesh"
    if agent == "orchestrator":          # friendly alias for the top-level card
        agent = "querymesh"

    def ok(result):
        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    def error(code, message):
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": code, "message": message}}

    if agent not in agent_cards.AGENT_CARDS:
        return error(-32602, f"unknown agent: {agent}")

    if method == "agent/getCard":
        return ok(agent_cards.get(agent))

    if method == "message/send":
        text = _text_of(params)
        if not text:
            return error(-32602, "message.parts must contain a text part")
        task_id = str(uuid.uuid4())
        output = _handle_agent(agent, text)
        task = {
            "id": task_id,
            "contextId": params.get("contextId", task_id),
            "status": {"state": "completed"},
            "artifacts": [{"artifactId": f"{task_id}-out", "name": "response",
                           "parts": [{"kind": "text", "text": output}]}],
        }
        TASKS[task_id] = task
        return ok(task)

    if method == "tasks/get":
        task_id = params.get("id")
        if task_id not in TASKS:
            return error(-32602, f"unknown task: {task_id}")
        return ok(TASKS[task_id])

    return error(-32601, f"method not found: {method}")

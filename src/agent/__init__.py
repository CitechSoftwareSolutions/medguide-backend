"""The agent: LangGraph orchestration over MedGemma (via Ollama).

* ``supervisor/``   routes each request to a capability or declines it
* ``knowledge_qa/`` the capability that runs Tool 01 and Tool 02
* ``memory/``       chat session identity and checkpoints
* ``agent.py``      the entrypoint; ``routes.py`` exposes it over HTTP

Deliberately free of re-exports: the tools import ``src.agent.state``, and the
graphs import the tools, so eager imports here would create a cycle. Import
what you need directly, for example ``from src.agent.agent import answer_question``.
"""

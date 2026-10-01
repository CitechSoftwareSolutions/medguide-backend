"""The agent: LangGraph orchestration over MedGemma (via Ollama).

* ``knowledge_qa/`` the graph that runs Tool 01 and Tool 02 for each question
* ``graph.py``      wraps knowledge_qa with chat memory (the session graph)
* ``memory/``       chat session identity and checkpoints
* ``agent.py``      the entrypoint; ``routes.py`` exposes it over HTTP

Deliberately free of re-exports: the tools import ``src.agent.state``, and the
graphs import the tools, so eager imports here would create a cycle. Import
what you need directly, for example ``from src.agent.agent import answer_question``.
"""

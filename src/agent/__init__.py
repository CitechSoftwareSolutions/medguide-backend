"""The AI layer: LangGraph orchestration over Claude and a local FAISS index.

Deliberately free of re-exports. ``src.services.rag_service`` imports
``src.agent.llm``, while the graph modules import ``src.services``; eagerly
pulling the graph in here would close that loop into a circular import. Import
what you need directly::

    from src.agent.agent import answer_question
"""

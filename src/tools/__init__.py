"""The two retrieval tools the agent orchestrates.

* ``tool_01_knowledge_base`` - the institution's curated guidelines. Answers are
  written from these passages and nothing else.
* ``tool_02_nice_guidance`` - NICE guidance. Used only to check an answer after
  it is written; never answer material.

Each tool has the same layout:

* ``data/``      the PDFs it indexes
* ``service.py`` its vector store, and indexing, search and sync over it
* ``nodes.py``   the LangGraph nodes it adds to the agent's knowledge_qa graph
* ``routes.py``  its document endpoints

The agent wires both tools into one graph in ``src/agent/knowledge_qa/graph.py``.
"""

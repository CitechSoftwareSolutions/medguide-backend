"""Application use cases.

Submodules are deliberately not imported here. ``agent_service`` pulls in the
agent graph, whose nodes import ``rag_service`` from this package; importing it
eagerly would make ``import src.agent.graph`` circular. ``from src.services
import rag_service`` still works, because Python loads a submodule on demand.
"""

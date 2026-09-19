"""Graph node implementations.

Deliberately empty of re-exports. ``supervisor_nodes`` imports the capability
registry from ``src.agent.tools``, and ``src.agent.tools`` imports the subgraph
that depends on ``qa_nodes``; eagerly importing both node modules here would
close that loop into a circular import. Import the specific module instead::

    from src.agent.nodes import qa_nodes
"""

"""Knowledge entry categories."""

from enum import StrEnum


class KnowledgeType(StrEnum):
    """The supported kinds of medical knowledge entries."""

    CONDITION = "condition"
    MEDICATION = "medication"
    PROCEDURE = "procedure"
    SYMPTOM = "symptom"

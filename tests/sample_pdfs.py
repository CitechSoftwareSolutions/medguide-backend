"""The real PDFs shipped with each tool, used as test inputs."""

from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[1] / "src" / "tools"

TOOL_01_DATA = _TOOLS / "tool_01_knowledge_base" / "data"
MIGRAINE_PDF = TOOL_01_DATA / "Migraine_Diagnosis_and_Management.pdf"
TREMOR_PDF = TOOL_01_DATA / "Chronic_Hand_Tremor_Diagnosis_and_Management.pdf"
DIABETES_PDF = TOOL_01_DATA / "Diabetes_Diagnosis_and_Management.pdf"

TOOL_02_DATA = _TOOLS / "tool_02_nice_guidance" / "data"
HEADACHES_PDF = TOOL_02_DATA / "headaches-pdf-243935625157.pdf"
TYPE_2_DIABETES_PDF = TOOL_02_DATA / "type-2-diabetes-in-adults-management-pdf-1837338615493.pdf"

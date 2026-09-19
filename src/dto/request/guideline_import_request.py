"""Request DTO for bulk-importing a structured clinical guideline document."""

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.validations import require_text


class DocumentMetadata(BaseModel):
    """Identifying information for one guideline document."""

    model_config = ConfigDict(extra="ignore")

    title: str
    category: str | None = None
    target_audience: str | None = None


class ClinicalPresentation(BaseModel):
    """How a condition typically presents."""

    model_config = ConfigDict(extra="ignore")

    key_features: list[str] = Field(default_factory=list)


class DiagnosticApproach(BaseModel):
    """What tends to cause or point to a condition."""

    model_config = ConfigDict(extra="ignore")

    common_causes: list[str] = Field(default_factory=list)


class ManagementPlan(BaseModel):
    """How a condition is treated, from first-line to follow-up."""

    model_config = ConfigDict(extra="ignore")

    initial_conservative: list[str] = Field(default_factory=list)
    specific_advanced: list[str] = Field(default_factory=list)
    follow_up_monitoring: list[str] = Field(default_factory=list)


class ConditionEntry(BaseModel):
    """One condition inside a guideline's registry."""

    model_config = ConfigDict(extra="ignore")

    condition_name: str
    clinical_presentation: ClinicalPresentation = Field(default_factory=ClinicalPresentation)
    diagnostic_approach: DiagnosticApproach = Field(default_factory=DiagnosticApproach)
    management_plan: ManagementPlan = Field(default_factory=ManagementPlan)

    @field_validator("condition_name")
    @classmethod
    def validate_condition_name(cls, value: str) -> str:
        """Trim the name and reject an empty one."""
        return require_text(value)


class ImportGuidelineRequest(BaseModel):
    """A full clinical guideline document, ready to become knowledge entries.

    Unknown extra keys are ignored rather than rejected (unlike the other
    request DTOs) because this is meant to accept whole documents pasted as-is;
    guidelines vary in shape and a stray field should not fail the import.
    """

    model_config = ConfigDict(extra="ignore")

    document_metadata: DocumentMetadata
    workflow_steps: list[str] = Field(default_factory=list)
    conditions_registry: list[ConditionEntry] = Field(default_factory=list)
    global_red_flags: list[str] = Field(default_factory=list)
    clinical_best_practices: list[str] = Field(default_factory=list)
    disclaimer: str | None = None

    @model_validator(mode="after")
    def validate_has_content(self) -> "ImportGuidelineRequest":
        """Reject a document with no conditions and no workflow to import."""
        if not self.conditions_registry and not self.workflow_steps:
            raise ValueError(
                "The guideline has neither conditions_registry nor workflow_steps to import."
            )
        return self

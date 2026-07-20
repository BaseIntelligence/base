from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

OverallVerdict = Literal["valid", "invalid", "suspicious", "error"]
RuleStatus = Literal["pass", "fail", "uncertain"]
RecommendedStatus = Literal["accepted", "rejected", "needs_review", "error"]


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    snippet: str
    reason_code: str
    description: str


class HardcodingFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    line_start: int = Field(ge=1)
    line_end: int = Field(ge=1)
    snippet: str
    reason_code: str
    description: str


class RuleResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rule_id: str
    title: str
    status: RuleStatus
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)


class WorkspaceFileContent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    content: str
    truncated: bool = False


class ReviewerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules_version: str
    rule_files: list[str]
    policy_excerpt: str
    workspace_files: list[str]
    static_findings: list[HardcodingFinding]
    file_contents: list[WorkspaceFileContent] = Field(default_factory=list)


class ReviewerResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    verdict: OverallVerdict
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceItem] = Field(default_factory=list)
    notes: str = ""


class AnalyzerPipelineReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rules_version: str
    overall_verdict: OverallVerdict
    recommended_status: RecommendedStatus
    reason_codes: list[str]
    rule_results: list[RuleResult]
    evidence: list[EvidenceItem]
    hardcoding_findings: list[HardcodingFinding]
    rules_files: list[str]
    reviewer_used: bool
    reviewer_notes: str = ""

    def to_json_compatible(self) -> dict[str, object]:
        return self.model_dump(mode="json")

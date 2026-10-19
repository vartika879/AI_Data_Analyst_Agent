"""Shared construction path for guarded LLM clients."""
from analyst.config import Settings
from analyst.config import ROLES
from analyst.llm import LLMClient, LLMError
from analyst.privacy import PrivacyContext, PrivacyGuardClient, tracing_enabled
from analyst.providers import make_client


def build_agent_clients(settings: Settings, df, use_clarifier: bool = True):
    """Return (planner, reviewer, narrator, clarifier) with one local privacy vault."""
    if tracing_enabled():
        raise LLMError("External LangSmith/LangChain tracing is enabled. Disable tracing before using uploaded data.")
    context = PrivacyContext(df, settings.privacy_mode)

    def guarded(role: str):
        return PrivacyGuardClient(make_client(settings, role), context, role)

    strict = settings.privacy_mode == "strict"
    planner = guarded("planner")
    reviewer = guarded("reviewer") if settings.reviewer_enabled and not strict else None
    narrator = guarded("narrator") if settings.narrator_enabled and not strict else None
    clarifier = guarded("clarifier") if settings.clarifier_enabled and use_clarifier else None
    return planner, reviewer, narrator, clarifier


class _DisabledAgentClient:
    def complete_json(self, system_prompt: str, user_prompt: str) -> str:
        raise LLMError("This agent is disabled in strict privacy mode.")


def build_guarded_client(settings: Settings, role: str, context: PrivacyContext) -> LLMClient:
    """Evaluation-runner entry point using the same guard as the application."""
    if tracing_enabled():
        raise LLMError("External LangSmith/LangChain tracing is enabled. Disable tracing before using uploaded data.")
    if role not in ROLES:
        raise ValueError(f"Unknown LLM role: {role}")
    if context.mode == "strict" and role in {"reviewer", "narrator"}:
        return _DisabledAgentClient()
    return PrivacyGuardClient(make_client(settings, role), context, role)

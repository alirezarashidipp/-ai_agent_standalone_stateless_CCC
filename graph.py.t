# graph.py  (stateless version)
import yaml
from typing import Literal
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt

from state import WorkflowState, ExtractorOutput, ValidatorOutput, TechQuestionsOutput
from llm_client import StructuredLLMClient

# Load configurations
with open("prompts.yaml", "r", encoding="utf-8") as f:
    PROMPTS = yaml.safe_load(f)

# Initialize Client
llm = StructuredLLMClient()

# Inline Model for Final Output Synthesis
class FinalStoryOutput(BaseModel):
    story: str = Field(description="The fully synthesized Jira Agile Story")

# ---------------------------------------------------------
# Node Logic  (unchanged)
# ---------------------------------------------------------

def phase0_extract(state: WorkflowState) -> dict:
    sys_prompt = PROMPTS["extractor"]["system"]
    user_prompt = f"Raw input: {state['raw_input']}"

    result = llm.query(sys_prompt, user_prompt, ExtractorOutput)

    missing = []
    if not result.who:  missing.append("who")
    if not result.what: missing.append("what")
    if not result.why:  missing.append("why")

    return {
        "who": result.who,
        "what": result.what,
        "why": result.why,
        "ac_evidence": result.ac_evidence,
        "missing_fields": missing,
        "phase1_retries": 0,
        "feedback_retries": 0,
    }


def phase1_lock(state: WorkflowState) -> dict:
    missing = state.get("missing_fields", [])
    if not missing:
        return {}

    target    = missing[0]
    retries   = state.get("phase1_retries", 0)

    if retries >= 3:
        return {"is_aborted": True, "abort_reason": f"Hard Abort: Failed 3 times on '{target}'."}

    rejection = state.get("last_rejection_reason")
    if rejection:
        prompt_msg = f"[REJECTED]: {rejection}\nMissing '{target}'. Please provide a valid value:"
    else:
        prompt_msg = f"Missing '{target}'. Please provide a value:"

    user_val = interrupt(prompt_msg)

    sys_prompt = PROMPTS["validator"]["system"].format(field=target)
    result     = llm.query(sys_prompt, f"User provided: {user_val}", ValidatorOutput)

    if result.is_valid:
        return {
            target: result.normalized_value,
            "missing_fields": missing[1:],
            "phase1_retries": 0,
            "last_rejection_reason": None,
        }
    else:
        return {
            "phase1_retries": retries + 1,
            "last_rejection_reason": result.rejection_reason,
        }


def phase2_tech_lead(state: WorkflowState) -> dict:
    sys_prompt = PROMPTS["tech_lead"]["system"].format(what=state["what"], why=state["why"])
    result     = llm.query(sys_prompt, "Review and generate technical questions.", TechQuestionsOutput)
    return {"pending_questions": result.questions, "tech_notes": []}


def phase2_ask_questions(state: WorkflowState) -> dict:
    questions = state.get("pending_questions", [])
    if not questions:
        return {}

    current_q   = questions[0]
    user_answer = interrupt(f"[Tech Lead]: {current_q}")

    sys_prompt = PROMPTS["inline_validator"]["system"].format(question=current_q, answer=user_answer)
    result     = llm.query(sys_prompt, f"Evaluate this answer: {user_answer}", ValidatorOutput)

    new_notes = list(state.get("tech_notes", []))
    if result.is_valid and result.normalized_value:
        new_notes.append(f"Constraint derived from '{current_q}': {result.normalized_value}")

    return {
        "pending_questions": questions[1:],
        "tech_notes": new_notes,
    }


def phase3_synthesize(state: WorkflowState) -> dict:
    tech_notes_str = "\n".join(state.get("tech_notes", []))
    feedback       = state.get("feedback_raw", "")

    sys_prompt = PROMPTS["agile_coach"]["system"].format(
        who=state["who"], what=state["what"], why=state["why"],
        tech_notes=tech_notes_str, ac_evidence=state.get("ac_evidence", ""),
        feedback=feedback,
    )

    result = llm.query(sys_prompt, "Generate final Jira story.", FinalStoryOutput)
    return {"final_story": result.story}


def phase3_feedback(state: WorkflowState) -> dict:
    retries = state.get("feedback_retries", 0)
    if retries >= 3:
        return {"is_complete": True}

    user_feedback = interrupt(
        f"\n[Agile Coach Output]:\n{state['final_story']}\n\n"
        "Type 'confirm' to accept, or provide feedback for rewrite:"
    )

    if user_feedback.strip().lower() == "confirm":
        return {"is_complete": True}

    return {
        "feedback_raw": user_feedback,
        "feedback_retries": retries + 1,
    }

# ---------------------------------------------------------
# Routing Logic  (unchanged)
# ---------------------------------------------------------

def route_after_phase0(state: WorkflowState) -> Literal["phase1_lock", "phase2_tech_lead"]:
    return "phase1_lock" if state["missing_fields"] else "phase2_tech_lead"


def route_after_phase1(state: WorkflowState) -> Literal["phase1_lock", "phase2_tech_lead", "__end__"]:
    if state.get("is_aborted"):     return END
    if state.get("missing_fields"): return "phase1_lock"
    return "phase2_tech_lead"


def route_after_phase2(state: WorkflowState) -> Literal["phase2_ask_questions", "phase3_synthesize"]:
    return "phase2_ask_questions" if state.get("pending_questions") else "phase3_synthesize"


def route_after_phase3(state: WorkflowState) -> Literal["phase3_synthesize", "__end__"]:
    return END if state.get("is_complete") else "phase3_synthesize"

# ---------------------------------------------------------
# Graph Compilation  — STATELESS: no checkpointer
# ---------------------------------------------------------

builder = StateGraph(WorkflowState)

builder.add_node("phase0_extract",      phase0_extract)
builder.add_node("phase1_lock",         phase1_lock)
builder.add_node("phase2_tech_lead",    phase2_tech_lead)
builder.add_node("phase2_ask_questions",phase2_ask_questions)
builder.add_node("phase3_synthesize",   phase3_synthesize)
builder.add_node("phase3_feedback",     phase3_feedback)

builder.add_edge(START, "phase0_extract")
builder.add_conditional_edges("phase0_extract",       route_after_phase0)
builder.add_conditional_edges("phase1_lock",          route_after_phase1)
builder.add_edge("phase2_tech_lead",                  "phase2_ask_questions")
builder.add_conditional_edges("phase2_ask_questions", route_after_phase2)
builder.add_edge("phase3_synthesize",                 "phase3_feedback")
builder.add_conditional_edges("phase3_feedback",      route_after_phase3)

# NOTE: graph.compile() is called WITHOUT a checkpointer.
#       All state persistence is the caller's responsibility (see main.py).
graph = builder.compile()

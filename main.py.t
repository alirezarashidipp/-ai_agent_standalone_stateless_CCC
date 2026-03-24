import copy
from langgraph.types import GraphInterrupt
from graph import graph
 
 
# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
 
def _empty_state(raw_text: str) -> dict:
    """Return a fully-initialised WorkflowState dict."""
    return {
        "raw_input":            raw_text,
        "who":                  None,
        "what":                 None,
        "why":                  None,
        "ac_evidence":          None,
        "missing_fields":       [],
        "current_field_target": None,
        "phase1_retries":       0,
        "last_rejection_reason": None,
        "is_aborted":           False,
        "abort_reason":         None,
        "pending_questions":    [],
        "current_question":     None,
        "tech_notes":           [],
        "final_story":          None,
        "feedback_retries":     0,
        "is_complete":          False,
        "feedback_raw":         "",
    }
 
 
def _merge(base: dict, update: dict) -> dict:
    """Shallow-merge `update` into a copy of `base`."""
    merged = copy.copy(base)
    merged.update(update)
    return merged
 
 
# ---------------------------------------------------------
# Main loop
# ---------------------------------------------------------
 
def run_pipeline() -> None:
    print("=== Jira Story Generator Pipeline (stateless) ===")
    raw_text = input("Enter raw unstructured requirement: ").strip()
 
    if not raw_text:
        print("[FATAL] Input cannot be empty. Aborting.")
        return
 
    # `current_state` is the single source of truth between graph calls.
    current_state: dict = _empty_state(raw_text)
    # When a node is interrupted it may have already produced partial state
    # updates *before* the interrupt call.  LangGraph surfaces those as the
    # return value of graph.invoke() when it raises GraphInterrupt.
    # We capture them in `pending_updates` and apply them before re-invoking.
    pending_updates: dict = {}
 
    while True:
        # Build the state we will pass into this invocation.
        invoke_state = _merge(current_state, pending_updates)
        pending_updates = {}
 
        try:
            # Stateless invocation — no config / thread_id needed.
            result: dict = graph.invoke(invoke_state)
 
        except GraphInterrupt as exc:
            # `exc.args[0]` is the Interrupt object emitted by `interrupt()`.
            interrupt_obj   = exc.args[0]
            prompt_msg      = interrupt_obj.value          # the string we asked
            partial_updates = getattr(exc, "state", {})   # any updates before the interrupt
 
            # Merge partial node updates so they survive the next invocation.
            current_state = _merge(invoke_state, partial_updates)
 
            print(f"\n[Action Required] {prompt_msg}")
            user_response = input("> ").strip()
 
            if user_response.lower() == "exit":
                print("[Process Terminated]")
                return
 
            # Inject the human answer as the *resume value* by storing it
            # in a well-known key.  The node that called interrupt() will
            # receive this value when the graph is re-invoked with the same
            # state (LangGraph stateless resume convention).
            pending_updates["__resume__"] = user_response
            continue  # re-invoke graph with updated state
 
        # ── Graph ran to completion (no interrupt raised) ──────────────────
        # `result` is the final WorkflowState dict.
        current_state = result
 
        if current_state.get("is_aborted"):
            print(f"\n[ABORTED] {current_state.get('abort_reason')}")
            return
 
        if current_state.get("is_complete"):
            print("\n=== FINAL AGILE STORY ===")
            print(current_state.get("final_story", "No story generated."))
            print("=========================")
            return
 
        # Execution exhausted without completion or interrupt — shouldn't
        # happen in normal flow, but guard against it.
        print("[WARN] Graph finished without is_complete flag. Check routing logic.")
        break
 
 
if __name__ == "__main__":
    run_pipeline()

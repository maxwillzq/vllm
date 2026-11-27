# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Grammar-as-a-Drafter (GaaD) draft-row planner.

Merges model-proposed draft tokens with deterministic grammar jump-forward
tokens into a single speculative draft row:

1. Model draft tokens that match the grammar's forced jump-forward token are
   promoted to grammar-forced slots (`is_forced=True`).
2. Model draft tokens at unconstrained positions that the grammar accepts stay
   model-draft slots (`is_forced=False`).
3. The model draft is early-cut at the first token that either disagrees with a
   forced grammar token or is rejected by the grammar.
4. Remaining slots up to `max_tokens` are filled with grammar-forced
   jump-forward tokens.
"""

from collections.abc import Sequence

from vllm.v1.structured_output.backend_types import StructuredOutputGrammar


def plan_gaad_draft_tokens(
    grammar: StructuredOutputGrammar,
    model_draft: Sequence[int],
    max_tokens: int,
) -> tuple[list[int], list[bool]]:
    """Plan up to `max_tokens` speculative draft tokens for a constrained request.

    Leaves `grammar` in its original committed state on return.

    Args:
        grammar: Active structured-output grammar positioned at the request's
            committed token state.
        model_draft: Unpadded model-proposed draft token IDs (may be empty).
        max_tokens: Maximum number of draft tokens to emit for this step.

    Returns:
        A `(tokens, is_forced)` pair of equal length `<= max_tokens`, where
        `is_forced[i]` is `True` if `tokens[i]` is deterministically forced by
        the grammar at position `i` (GaaD `GRAMMAR` slot) and `False` if it is
        a model-proposed token (`MODEL` slot).
    """
    if max_tokens <= 0 or grammar.is_terminated():
        return [], []

    tokens: list[int] = []
    is_forced: list[bool] = []
    advanced = 0

    # Cached jump-forward run from the current grammar state so advancing
    # through multi-token forced strings avoids re-querying / re-encoding.
    run: list[int] = []
    pos = 0
    run_valid = False

    def peek_forced(remaining: int) -> int | None:
        nonlocal run, pos, run_valid
        if not run_valid or pos >= len(run):
            run = grammar.find_jump_forward_tokens(remaining)
            pos = 0
            run_valid = True
        return run[pos] if pos < len(run) else None

    try:
        # Phase 1: walk model draft tokens, upgrading forced hits and
        # early-cutting on mismatches or grammar rejections.
        limit = min(len(model_draft), max_tokens)
        for i in range(limit):
            if grammar.is_terminated():
                break
            tok = model_draft[i]
            forced = peek_forced(max_tokens - len(tokens))
            if forced is not None:
                if tok != forced:
                    break
                if not grammar.accept_tokens("", [tok]):
                    break
                advanced += 1
                pos += 1
                tokens.append(tok)
                is_forced.append(True)
            else:
                if not grammar.accept_tokens("", [tok]):
                    break
                advanced += 1
                run_valid = False
                tokens.append(tok)
                is_forced.append(False)

        # Phase 2: fill remaining budget with deterministic jump-forward tokens.
        while len(tokens) < max_tokens and not grammar.is_terminated():
            forced = peek_forced(max_tokens - len(tokens))
            if forced is None:
                break
            if not grammar.accept_tokens("", [forced]):
                break
            advanced += 1
            pos += 1
            tokens.append(forced)
            is_forced.append(True)
    finally:
        if advanced > 0:
            grammar.rollback(advanced)

    return tokens, is_forced

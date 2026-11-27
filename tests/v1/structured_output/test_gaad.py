# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import xgrammar as xgr

from vllm.config.structured_outputs import StructuredOutputsConfig
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.structured_output.backend_xgrammar import XgrammarGrammar
from vllm.v1.structured_output.gaad import plan_gaad_draft_tokens
from vllm.v1.worker.gpu.spec_decode.grammar.config import GrammarSpeculatorConfig
from vllm.v1.worker.gpu.spec_decode.grammar.speculator import GrammarSpeculator
from vllm.v1.worker.gpu.spec_decode.speculator_config import _CONFIG_BY_METHOD


class _GreedyMockTokenizer:
    """Minimal greedy longest-match tokenizer for deterministic XGrammar tests."""

    def __init__(self, vocab: list[str]) -> None:
        self.vocab = vocab
        self.encode_calls = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encode_calls += 1
        ids: list[int] = []
        rest = text
        while rest:
            best_id = -1
            best_len = 0
            for idx, tok in enumerate(self.vocab):
                if tok and rest.startswith(tok) and len(tok) > best_len:
                    best_id = idx
                    best_len = len(tok)
            if best_id < 0:
                return []
            ids.append(best_id)
            rest = rest[best_len:]
        return ids


def _make_xgrammar_regex(
    vocab: list[str], regex: str, max_rollback: int = 16
) -> tuple[XgrammarGrammar, _GreedyMockTokenizer]:
    stop_id = len(vocab) - 1
    tok_info = xgr.TokenizerInfo(
        encoded_vocab=vocab,
        vocab_type=xgr.VocabType.RAW,
        vocab_size=len(vocab),
        stop_token_ids=[stop_id],
    )
    compiler = xgr.GrammarCompiler(tok_info)
    ctx = compiler.compile_regex(regex)
    matcher = xgr.GrammarMatcher(ctx, max_rollback_tokens=max_rollback)
    tokenizer = _GreedyMockTokenizer(vocab)
    grammar = XgrammarGrammar(
        vocab_size=len(vocab),
        matcher=matcher,
        ctx=ctx,
        tokenizer=tokenizer,
    )
    return grammar, tokenizer


def test_plan_gaad_standalone_fills_forced_run_and_caches_encode():
    vocab = ["ab", "cd", "ef", "1", "2", "<stop>"]
    grammar, tokenizer = _make_xgrammar_regex(vocab, r"abcdef[12]+")

    tokens, is_forced = plan_gaad_draft_tokens(grammar, [], max_tokens=6)
    assert tokens == [0, 1, 2]
    assert is_forced == [True, True, True]
    # Multi-token forced run is encoded once, not once per token.
    assert tokenizer.encode_calls == 1
    # Committed grammar state is restored after planning.
    assert grammar.find_jump_forward_tokens(6) == [0, 1, 2]


def test_plan_gaad_hybrid_promotes_cuts_and_fills():
    # Pattern: "ab" "cd" <digit> "ef"
    vocab = ["ab", "cd", "1", "ef", "X", "<stop>"]
    grammar, _ = _make_xgrammar_regex(vocab, r"abcd[1]ef")

    # Model proposes ["ab", "cd", "1"] -> promoted to [GRAMMAR, GRAMMAR, GRAMMAR]
    # and trailing forced "ef" is filled into slot 3.
    tokens, is_forced = plan_gaad_draft_tokens(grammar, [0, 1, 2], max_tokens=4)
    assert tokens == [0, 1, 2, 3]
    assert is_forced == [True, True, True, True]

    # Pattern with free choice in middle: "ab" ("1"|"2") "cd" "ef"
    vocab2 = ["ab", "cd", "ef", "1", "2", "X", "<stop>"]
    grammar2, _ = _make_xgrammar_regex(vocab2, r"ab[12]cdef")

    # Model proposes ["ab", "2"] -> slot 0 promoted (GRAMMAR), slot 1 stays MODEL,
    # slots 2..3 filled with forced ["cd", "ef"] (GRAMMAR).
    tokens2, is_forced2 = plan_gaad_draft_tokens(grammar2, [0, 4], max_tokens=4)
    assert tokens2 == [0, 4, 1, 2]
    assert is_forced2 == [True, False, True, True]

    # Early-cut on grammar-invalid token "X" at slot 1: stops before "X" and
    # does not advance into invalid state.
    tokens3, is_forced3 = plan_gaad_draft_tokens(grammar2, [0, 5, 1], max_tokens=4)
    assert tokens3 == [0]
    assert is_forced3 == [True]

    # Early-cut when model disagrees with forced token at slot 0: replaces with
    # forced token "ab".
    tokens4, is_forced4 = plan_gaad_draft_tokens(grammar2, [3, 1], max_tokens=4)
    assert tokens4 == [0]
    assert is_forced4 == [True]


def test_manager_plan_spec_tokens_and_one_hot_bitmask():
    # Grammar admits both '"' (1) and '"a' (2) at the start, while the greedy
    # jump tokenizer selects [2, 3] ('"a', 'ge').
    vocab = ["{", '"', '"a', "ge", "age", "1", "2", "<stop>"]
    grammar, _ = _make_xgrammar_regex(vocab, r'"age[12]+')

    vllm_config = SimpleNamespace(
        structured_outputs_config=StructuredOutputsConfig(
            backend="xgrammar",
            enable_grammar_drafter=True,
        ),
        parallel_config=SimpleNamespace(distributed_executor_backend="mp"),
        scheduler_config=SimpleNamespace(max_num_seqs=4),
        model_config=SimpleNamespace(
            skip_tokenizer_init=True,
            is_diffusion=False,
            get_vocab_size=lambda: len(vocab),
        ),
        num_speculative_tokens=4,
    )
    manager = StructuredOutputManager(vllm_config)  # type: ignore[arg-type]

    req = SimpleNamespace(
        request_id="req-1",
        use_structured_output=True,
        structured_output_request=SimpleNamespace(
            grammar=grammar,
            reasoning_ended=True,
            masked_reasoning_end_ids=None,
        ),
        prompt_token_ids=[0],
        _all_token_ids=[0],
    )

    planned = manager.plan_spec_tokens(req, [-1, -1, -1, -1], max_num_spec_tokens=4)  # type: ignore[arg-type]
    assert planned == [2, 3]

    bitmask = manager.grammar_bitmask(
        requests={"req-1": req},  # type: ignore[dict-item]
        structured_output_request_ids=["req-1"],
        scheduled_spec_decode_tokens={"req-1": planned},
    )
    assert bitmask is not None
    # Every forced slot i must have a one-hot bitmask allowing ONLY planned[i]
    # (in particular slot 0 must disallow '"' (1) and allow only '"a' (2)).
    for i, tok in enumerate(planned):
        row_u32 = bitmask[i].view(np.uint32)
        expected = np.zeros_like(row_u32)
        expected[tok >> 5] = np.uint32(1 << (tok & 31))
        assert np.array_equal(row_u32, expected), (
            f"slot {i} (token {tok}) should be one-hot masked"
        )

    # Bonus row (after all forced tokens) must allow both digits '1' (5) and '2' (6).
    bonus_u32 = bitmask[len(planned)].view(np.uint32)
    assert (bonus_u32[0] & (1 << 5)) != 0
    assert (bonus_u32[0] & (1 << 6)) != 0

    manager.clear_request_state("req-1")
    assert "req-1" not in manager._gaad_forced_flags


def test_structured_outputs_config_and_grammar_speculator():
    with pytest.raises(ValueError, match="xgrammar"):
        StructuredOutputsConfig(backend="guidance", enable_grammar_drafter=True)

    assert "grammar" in _CONFIG_BY_METHOD
    spec = SimpleNamespace(
        num_speculative_tokens=None,
        prompt_lookup_max=None,
        prompt_lookup_min=None,
        target_model_config="target_mc",
        target_parallel_config="target_pc",
        draft_model_config=None,
        draft_parallel_config=None,
    )
    GrammarSpeculatorConfig.configure(spec)  # type: ignore[arg-type]
    assert spec.num_speculative_tokens == 8
    assert spec.draft_model_config == "target_mc"

    vllm_config = SimpleNamespace(
        speculative_config=SimpleNamespace(num_speculative_tokens=4),
        scheduler_config=SimpleNamespace(max_num_seqs=8),
    )
    speculator = GrammarSpeculator(vllm_config, torch.device("cpu"))  # type: ignore[arg-type]
    speculator.init_cudagraph_manager(MagicMock())
    assert speculator.capture() is None
    input_batch = SimpleNamespace(num_reqs=3)
    drafts = speculator.propose(
        input_batch,  # type: ignore[arg-type]
        {},
        {},
        torch.empty(0),
        None,
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
        torch.empty(0),
    )
    assert drafts.shape == (3, 4)
    assert drafts.dtype == torch.int64
    assert torch.all(drafts == -1)

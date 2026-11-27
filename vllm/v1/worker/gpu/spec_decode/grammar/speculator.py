# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone Grammar-as-a-Drafter (GaaD) speculator for the V2 GPU runner.

Draft tokens are proposed on the host by `StructuredOutputManager` from
deterministic grammar jump-forward strings; this speculator provides the
device-side placeholder tensor without launching any draft model or kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.spec_decode.speculator import BaseSpeculator

if TYPE_CHECKING:
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.dp_utils import DPSyncState


class GrammarSpeculator(BaseSpeculator):
    """Lightweight placeholder speculator for host-proposed grammar drafts."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device) -> None:
        assert vllm_config.speculative_config is not None
        num_spec_tokens = vllm_config.speculative_config.num_speculative_tokens
        assert num_spec_tokens is not None
        max_num_reqs = vllm_config.scheduler_config.max_num_seqs
        self.drafts = torch.full(
            (max_num_reqs, num_spec_tokens),
            -1,
            dtype=torch.int64,
            device=device,
        )

    def init_cudagraph_manager(self, cudagraph_mode: CUDAGraphMode) -> None:
        del cudagraph_mode

    def capture(self) -> None:
        return None

    @torch.inference_mode()
    def propose(
        self,
        input_batch: InputBatch,
        attn_metadata: Any,
        slot_mappings: Any,
        last_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        num_sampled: torch.Tensor,
        num_rejected: torch.Tensor,
        last_sampled: torch.Tensor,
        next_prefill_tokens: torch.Tensor,
        temperature: torch.Tensor,
        seeds: torch.Tensor,
        dp_sync_state: DPSyncState | None = None,
        dummy_run: bool = False,
        skip_attn_for_dummy_run: bool = False,
        mm_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        is_profile: bool = False,
        num_speculative_tokens: int | None = None,
    ) -> torch.Tensor:
        return self.drafts[: input_batch.num_reqs]

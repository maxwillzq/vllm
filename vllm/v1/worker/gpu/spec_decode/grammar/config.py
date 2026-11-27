# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.worker.gpu.spec_decode.speculator_config import SpeculatorConfig

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig

logger = init_logger(__name__)


class GrammarSpeculatorConfig(SpeculatorConfig):
    """Config for standalone Grammar-as-a-Drafter (GaaD) speculative decoding."""

    base_method = "grammar"
    speculator_class = (
        "vllm.v1.worker.gpu.spec_decode.grammar.speculator.GrammarSpeculator"
    )
    adaptive_verification_uses_draft_logits = False

    @classmethod
    def configure(cls, spec: "SpeculativeConfig") -> None:
        super().configure(spec)
        if spec.num_speculative_tokens is None:
            spec.num_speculative_tokens = 8
            logger.info(
                "Defaulted num_speculative_tokens to %s for grammar drafting.",
                spec.num_speculative_tokens,
            )

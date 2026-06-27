# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import time
from functools import total_ordering

import torch
from transformers import AutoTokenizer, OPTForSequenceClassification

from vllm.logger import init_logger
from vllm.v1.request import Request

logger = init_logger(__name__)


class OPTPredictor:
    """
    Wrapper for OPT-based Learning-to-Rank predictor.
    Uses OPTForSequenceClassification with num_labels=1 to output
    a ranking score per request.
    Lower score = shorter predicted output = higher SJF priority.
    """

    def __init__(self, model_path: str, device: str = "cpu") -> None:
        self.device = device
        logger.info("Loading OPT predictor from %s", model_path)

        self.tokenizer = AutoTokenizer.from_pretrained(
            "facebook/opt-125m", use_fast=False
        )
        self.model = OPTForSequenceClassification.from_pretrained(
            model_path,
            dtype=torch.float16,
        )
        self.model.eval()
        self.model.to(self.device)
        logger.info("OPT predictor loaded successfully")

    @torch.no_grad()
    def predict_score(self, prompt_token_ids: list[int]) -> float:
        """
        Predict ranking score for a single request.
        Lower score = shorter predicted output = higher SJF priority.
        """
        input_ids = torch.tensor(
            [prompt_token_ids], dtype=torch.long, device=self.device
        )

        # Truncate if too long (OPT-125m max 2048)
        if input_ids.shape[1] > 2048:
            input_ids = input_ids[:, -2048:]
        # Clamp token ids to vocab size
        input_ids = input_ids.clamp(0, 50271)

        outputs = self.model(input_ids=input_ids)
        score = outputs.logits.squeeze().item()
        return score


@total_ordering
class SJFSorter:
    """
    Wraps a Request with a composite score for heap-based SJF ordering.
    Composite score = predictor_score - AGING_FACTOR * wait_time
    Lower score = higher priority (min-heap).
    Aging decreases score over time to prevent starvation of long requests.
    """

    AGING_FACTOR = 0.02  # 5초 대기시 score 0.1 변화 -> starvation 방지

    def __init__(self, request: Request, predictor: OPTPredictor) -> None:
        self.request = request
        self.request_id = request.request_id
        assert request.prompt_token_ids is not None, (
            "SJF requires prompt_token_ids to be set"
        )
        self._base_score: float = predictor.predict_score(
            request.prompt_token_ids
        )
        self._arrival_time: float = request.arrival_time

    @property
    def score(self) -> float:
        """Lower = higher priority. Aging reduces score over time."""
        wait_time = time.time() - self._arrival_time
        # 높은 score = 짧은 job -> 부호 반전해서 min-heap에서 먼저 나오게
        return -(self._base_score - self.AGING_FACTOR * wait_time)

    def __lt__(self, other: "SJFSorter") -> bool:
        return self.score < other.score

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, SJFSorter):
            return NotImplemented
        return self.request_id == other.request_id

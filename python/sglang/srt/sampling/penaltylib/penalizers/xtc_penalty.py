import typing

import torch

from ..orchestrator import _BatchedPenalizer, _TokenIDs

class BatchedXTCPenalizer(_BatchedPenalizer):
    """
    Exclude Top Choices (XTC) penalizer penalizes tokens based on xtc_thresholds and xtc_probabilities.
    """

    xtc_thresholds: torch.Tensor = None
    xtc_probabilities: torch.Tensor = None

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.xtc_threshold > 0 and req.sampling_params.xtc_probability > 0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        self.xtc_thresholds = torch.tensor(
            data=[req.sampling_params.xtc_threshold for req in self.orchestrator.reqs()],
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

        self.xtc_probabilities = torch.tensor(
            data=[req.sampling_params.xtc_probability for req in self.orchestrator.reqs()],
            dtype=torch.float32,
            device=self.orchestrator.device,
        )

    def _teardown(self):
        del self.xtc_thresholds
        del self.xtc_probabilities
        self.xtc_thresholds = None
        self.xtc_probabilities = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        pass

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        pass

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply Exclude Top Choices (XTC) sampling to the logits.
        Reference: https://github.com/oobabooga/text-generation-webui/pull/6335
        """
        apply_xtc = torch.rand_like(self.xtc_probabilities) < self.xtc_probabilities

        if not apply_xtc.any():
            return logits

        probs = torch.softmax(logits, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

        # Find indices where the next probability is above the threshold
        # Skips the top choice, which later on becomes skipping the last choice.
        above_threshold = sorted_probs[..., 1:] >= self.xtc_thresholds.unsqueeze(-1)

        # Apply XTC only to rows where it should be applied
        for i in range(logits.shape[0]):
            if apply_xtc[i]:
                # Count logits above the threshold (skipping the first)
                indices_to_remove = above_threshold[i].count_nonzero(dim=-1).item()
                if indices_to_remove > 0:
                    # Implies the top logit and at least one other is >= threshold.
                    # Mask out above_thresh logits except the last/lowest one.
                    logits[i].scatter_(
                        0, sorted_indices[i, :indices_to_remove], -float('inf'))
        return logits

    def _filter(
        self, indices_to_keep: typing.List[int], indices_tensor_to_keep: torch.Tensor
    ):
        self.xtc_thresholds = self.xtc_thresholds[indices_tensor_to_keep]
        self.xtc_probabilities = self.xtc_probabilities[indices_tensor_to_keep]

    def _merge(self, their: "BatchedXTCPenalizer"):
        self.xtc_thresholds = torch.cat([self.xtc_thresholds, their.xtc_thresholds], dim=0)
        self.xtc_probabilities = torch.cat([self.xtc_probabilities, their.xtc_probabilities], dim=0)
import typing

import torch

from ..orchestrator import _BatchedPenalizer, _TokenIDs


class BatchedXTCPenalizer(_BatchedPenalizer):
    """
    XTC (eXclusive Threshold Calibration) penalizer that modifies the logits based on a threshold and
    probability per request. With a certain probability, it removes tokens whose probabilities exceed a threshold,
    except for special tokens like newline or EOS tokens.
    """

    thresholds: torch.Tensor = None
    probabilities: torch.Tensor = None
    filter_value: float = -float("inf")
    special_token_ids: typing.List[typing.List[int]] = None
    random_values: torch.Tensor = None

    def _is_required(self) -> bool:
        # Determine if XTC sampling is required for any request
        return any(
            getattr(req.sampling_params, 'xtc_threshold', 0.0) > 0.0 and
            getattr(req.sampling_params, 'xtc_probability', 0.0) > 0.0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        device = self.orchestrator.device

        # Prepare per-request thresholds and probabilities tensors
        self.thresholds = torch.tensor(
            [
                getattr(req.sampling_params, 'xtc_threshold', 0.0)
                for req in self.orchestrator.reqs()
            ],
            dtype=torch.float32,
            device=device
        ).unsqueeze(1)  # Shape: (batch_size, 1)

        self.probabilities = torch.tensor(
            [
                getattr(req.sampling_params, 'xtc_probability', 0.0)
                for req in self.orchestrator.reqs()
            ],
            dtype=torch.float32,
            device=device
        ).unsqueeze(1)  # Shape: (batch_size, 1)

        # Prepare special token IDs per request
        self.special_token_ids = []
        for req in self.orchestrator.reqs():
            special_ids = []
            # Get the ID for "\n"
            newline_token_id = req.tokenizer.encode("\n")[-1]
            if newline_token_id is not None:
                special_ids.append(newline_token_id)
            # Get the EOS token ID
            if req.tokenizer.eos_token_id is not None:
                special_ids.append(req.tokenizer.eos_token_id)
            self.special_token_ids.append(special_ids)

        # Initialize random values tensor
        self.random_values = None  # Will be generated during _apply

    def _teardown(self):
        del self.thresholds
        del self.probabilities
        del self.special_token_ids
        del self.random_values
        self.thresholds = None
        self.probabilities = None
        self.special_token_ids = None
        self.random_values = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        pass  # Not needed for XTC sampling

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        pass  # Not needed for XTC sampling

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = logits.size()
        device = logits.device

        # Generate random values per batch element to decide whether to apply XTC
        self.random_values = torch.rand(batch_size, 1, device=device)

        # Create a mask indicating whether to apply XTC sampling
        apply_xtc_mask = self.random_values < self.probabilities  # Shape: (batch_size, 1)

        # Indices of batch elements where XTC should be applied
        indices_to_apply_xtc = apply_xtc_mask.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)

        if indices_to_apply_xtc.numel() == 0:
            # No batch elements require XTC sampling
            return logits

        # Clone logits to avoid in-place modifications
        results_logits = logits.clone()

        # Extract logits and thresholds for elements that require XTC
        logits_xtc = logits[indices_to_apply_xtc]  # Shape: (num_xtc, vocab_size)
        thresholds_xtc = self.thresholds[indices_to_apply_xtc]  # Shape: (num_xtc, 1)
        special_token_ids_xtc = [self.special_token_ids[i] for i in indices_to_apply_xtc.tolist()]

        # Compute probabilities
        probs_xtc = torch.softmax(logits_xtc, dim=-1)  # Shape: (num_xtc, vocab_size)

        # Sort the probabilities and get sorted indices
        sorted_probs_xtc, sorted_indices_xtc = torch.sort(probs_xtc, descending=True, dim=-1)

        # Initialize boolean mask for indices to remove
        sorted_indices_to_remove = torch.full_like(sorted_probs_xtc, False, dtype=torch.bool)

        # Set indices to True where the next token's probability is above the threshold
        sorted_indices_to_remove[:, :-1] = sorted_probs_xtc[:, 1:] >= thresholds_xtc

        # Convert sorted indices to original indices
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool).scatter(
            dim=1, index=sorted_indices_xtc, src=sorted_indices_to_remove
        )

        # Create a mask for special tokens
        special_tokens_mask = torch.zeros_like(logits_xtc, dtype=torch.bool)
        for idx, token_ids in enumerate(special_token_ids_xtc):
            special_tokens_mask[idx, token_ids] = True

        # Check if any special tokens would be removed
        special_tokens_removed = (indices_to_remove & special_tokens_mask).any(dim=1)  # Shape: (num_xtc,)

        # Indices of batch elements eligible for masking
        eligible_indices = (~special_tokens_removed).nonzero(as_tuple=False).squeeze(-1)

        if eligible_indices.numel() == 0:
            # No eligible batch elements after removing special tokens
            return logits

        # Apply the mask to eligible batch elements
        mask = indices_to_remove[eligible_indices]
        logits_xtc[eligible_indices] = logits_xtc[eligible_indices].masked_fill(mask, self.filter_value)

        # Update the results_logits with modified logits
        results_logits[indices_to_apply_xtc] = logits_xtc

        return results_logits

    def _filter(
        self, indices_to_keep: typing.List[int], indices_tensor_to_keep: torch.Tensor
    ):
        # Filter tensors based on indices to keep
        self.thresholds = self.thresholds[indices_tensor_to_keep]
        self.probabilities = self.probabilities[indices_tensor_to_keep]
        self.special_token_ids = [self.special_token_ids[i] for i in indices_to_keep]
        if self.random_values is not None:
            self.random_values = self.random_values[indices_tensor_to_keep]

    def _merge(self, their: "BatchedXTCPenalizer"):
        # Merge tensors from another BatchedXTCPenalizer
        self.thresholds = torch.cat([self.thresholds, their.thresholds], dim=0)
        self.probabilities = torch.cat([self.probabilities, their.probabilities], dim=0)
        self.special_token_ids.extend(their.special_token_ids)
        if self.random_values is not None and their.random_values is not None:
            self.random_values = torch.cat([self.random_values, their.random_values], dim=0)
        elif self.random_values is None and their.random_values is not None:
            self.random_values = their.random_values.clone()
        # Else, random_values remain None
import typing

import torch

from ..orchestrator import _BatchedPenalizer, _TokenIDs


class BatchedDRYPenalizer(_BatchedPenalizer):
    """
    Batched DRY (Don't Repeat Yourself) Penalizer penalizes tokens based on repetitive patterns
    in the generated sequence to encourage diversity.
    """

    dry_multipliers: torch.Tensor = None
    dry_bases: torch.Tensor = None
    dry_allowed_lengths: torch.Tensor = None
    dry_sequence_breakerss: typing.List[set] = None
    input_ids: typing.List[torch.Tensor] = None

    def _is_required(self) -> bool:
        return any(
            req.sampling_params.dry_multiplier is not None and req.sampling_params.dry_multiplier > 0
            for req in self.orchestrator.reqs()
        )

    def _prepare(self):
        reqs = self.orchestrator.reqs()
        batch_size = self.orchestrator.batch_size()
        device = self.orchestrator.device

        self.dry_multipliers = torch.tensor(
            [req.sampling_params.dry_multiplier for req in reqs],
            dtype=torch.float32,
            device=device,
        )

        self.dry_bases = torch.tensor(
            [req.sampling_params.dry_base for req in reqs],
            dtype=torch.float32,
            device=device,
        )

        self.dry_allowed_lengths = torch.tensor(
            [req.sampling_params.dry_allowed_length for req in reqs],
            dtype=torch.int32,
            device=device,
        )

        self.dry_sequence_breakerss = []
        for req in reqs:
            breakers = set(req.sampling_params.dry_sequence_breakers or [])
            breakers |= set(req.tokenizer.additional_sequence_breakers or [])
            breakers.add(req.tokenizer.eos_token_id)
            self.dry_sequence_breakerss.append(breakers)

        self.input_ids = [torch.tensor([], dtype=torch.int64, device=device) for _ in range(batch_size)]

    def _teardown(self):
        del self.dry_multipliers
        del self.dry_bases
        del self.dry_allowed_lengths
        del self.dry_sequence_breakerss
        del self.input_ids

        self.dry_multipliers = None
        self.dry_bases = None
        self.dry_allowed_lengths = None
        self.dry_sequence_breakerss = None
        self.input_ids = None

    def _cumulate_input_tokens(self, input_ids: _TokenIDs):
        # input_ids is a list of tensors, one per sequence
        for i in range(len(self.input_ids)):
            self.input_ids[i] = torch.cat([self.input_ids[i], input_ids[i]])

    def _cumulate_output_tokens(self, output_ids: _TokenIDs):
        for i in range(len(self.input_ids)):
            self.input_ids[i] = torch.cat([self.input_ids[i], output_ids[i]])

    def _apply(self, logits: torch.Tensor) -> torch.Tensor:
        num_seqs, vocab_size = logits.shape
        device = logits.device
        for i in range(num_seqs):
            input_ids = self.input_ids[i]
            if input_ids.numel() == 0:
                continue
            last_token = input_ids[-1].item()
            if last_token in self.dry_sequence_breakerss[i]:
                continue
            match_indices = (input_ids[:-1] == last_token).nonzero(as_tuple=False).view(-1)
            if match_indices.numel() == 0:
                continue
            match_lengths = {}
            for idx in match_indices.tolist():
                if idx + 1 >= len(input_ids):
                    continue
                next_token = input_ids[idx + 1].item()
                if next_token in self.dry_sequence_breakerss[i]:
                    continue
                # Compute match_length
                match_length = 1
                while (
                    idx - match_length >= 0
                    and len(input_ids) - match_length - 1 >= 0
                    and input_ids[idx - match_length] == input_ids[-(match_length + 1)]
                    and input_ids[idx - match_length].item() not in self.dry_sequence_breakerss[i]
                ):
                    match_length += 1
                match_lengths[next_token] = max(match_lengths.get(next_token, 0), match_length)
            # Apply penalties
            for token, match_length in match_lengths.items():
                if match_length >= self.dry_allowed_lengths[i]:
                    penalty = self.dry_multipliers[i] * (self.dry_bases[i] ** (match_length - self.dry_allowed_lengths[i]))
                    logits[i, token] -= penalty
        return logits

    def _filter(
        self, indices_to_keep: typing.List[int], indices_tensor_to_keep: torch.Tensor
    ):
        self.dry_multipliers = self.dry_multipliers[indices_tensor_to_keep]
        self.dry_bases = self.dry_bases[indices_tensor_to_keep]
        self.dry_allowed_lengths = self.dry_allowed_lengths[indices_tensor_to_keep]
        self.dry_sequence_breakerss = [self.dry_sequence_breakerss[i] for i in indices_to_keep]
        self.input_ids = [self.input_ids[i] for i in indices_to_keep]

    def _merge(self, their: "BatchedDRYPenalizer"):
        self.dry_multipliers = torch.cat([self.dry_multipliers, their.dry_multipliers], dim=0)
        self.dry_bases = torch.cat([self.dry_bases, their.dry_bases], dim=0)
        self.dry_allowed_lengths = torch.cat([self.dry_allowed_lengths, their.dry_allowed_lengths], dim=0)
        self.dry_sequence_breakerss.extend(their.dry_sequence_breakerss)
        self.input_ids.extend(their.input_ids)
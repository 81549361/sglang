"""Per-layer attention backend dispatch based on head_dim.

For models with heterogeneous head dimensions (e.g., Gemma 4 where
sliding-window layers use head_dim=256 and full-attention layers use
head_dim=512), this backend dispatches each layer to the fastest
compatible kernel: layers within the FlashAttention head_dim limit
use the *primary* backend (typically FA3); the rest fall back to a
universally-compatible *fallback* backend (typically Triton).
"""

import logging
from typing import TYPE_CHECKING, Optional, Set

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend

if TYPE_CHECKING:
    from sglang.srt.layers.attention.nsa.nsa_indexer import BaseIndexerMetadata
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
    from sglang.srt.speculative.spec_info import SpecInput

logger = logging.getLogger(__name__)


class HybridHeadDimAttnBackend(AttentionBackend):
    """Dispatches layers to one of two backends based on head_dim.

    Layers whose ``layer_id`` is in *fallback_layer_ids* are served by
    ``fallback_backend``; all other layers use ``primary_backend``.

    When *extend_always_fallback* is ``True`` the fallback backend is used
    for **all** layers during extend / prefill (useful when prefill requires
    features only the fallback backend supports, such as custom attention
    masks for bidirectional attention in Gemma 4 multimodal).
    """

    def __init__(
        self,
        primary_backend: AttentionBackend,
        fallback_backend: AttentionBackend,
        fallback_layer_ids: Set[int],
        extend_always_fallback: bool = False,
    ):
        self.primary_backend = primary_backend
        self.fallback_backend = fallback_backend
        self.fallback_layer_ids = fallback_layer_ids
        self.extend_always_fallback = extend_always_fallback
        self._backends = [primary_backend, fallback_backend]

    # ------------------------------------------------------------------
    # Metadata initialisation – both backends must be kept in sync
    # ------------------------------------------------------------------

    def init_forward_metadata(self, forward_batch: "ForwardBatch"):
        for b in self._backends:
            b.init_forward_metadata(forward_batch)

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        for b in self._backends:
            b.init_cuda_graph_state(max_bs, max_num_tokens)

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
    ):
        for b in self._backends:
            b.init_forward_metadata_capture_cuda_graph(
                bs,
                num_tokens,
                req_pool_indices,
                seq_lens,
                encoder_lens,
                forward_mode,
                spec_info,
            )

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens: Optional[torch.Tensor],
        forward_mode: "ForwardMode",
        spec_info: Optional["SpecInput"],
        seq_lens_cpu: Optional[torch.Tensor],
    ):
        for b in self._backends:
            b.init_forward_metadata_replay_cuda_graph(
                bs,
                req_pool_indices,
                seq_lens,
                seq_lens_sum,
                encoder_lens,
                forward_mode,
                spec_info,
                seq_lens_cpu,
            )

    def get_cuda_graph_seq_len_fill_value(self):
        return self.primary_backend.get_cuda_graph_seq_len_fill_value()

    # ------------------------------------------------------------------
    # Per-layer forward dispatch
    # ------------------------------------------------------------------

    def _select_backend_for_layer(
        self, layer: "RadixAttention", is_extend: bool = False
    ) -> AttentionBackend:
        if is_extend and self.extend_always_fallback:
            return self.fallback_backend
        if layer.layer_id in self.fallback_layer_ids:
            return self.fallback_backend
        return self.primary_backend

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._select_backend_for_layer(layer, is_extend=False).forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: "RadixAttention",
        forward_batch: "ForwardBatch",
        save_kv_cache: bool = True,
        **kwargs,
    ):
        return self._select_backend_for_layer(layer, is_extend=True).forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def get_indexer_metadata(
        self, layer_id: int, forward_batch: "ForwardBatch"
    ) -> Optional["BaseIndexerMetadata"]:
        if layer_id in self.fallback_layer_ids:
            return self.fallback_backend.get_indexer_metadata(layer_id, forward_batch)
        return self.primary_backend.get_indexer_metadata(layer_id, forward_batch)

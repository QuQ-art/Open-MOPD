# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Forward-local reuse of Hugging Face FlashAttention unpadding metadata.

Only identical, versioned mask tensors on the same CUDA stream are reused.
The installed dispatcher is process-wide, but cache state is context-local:
student/teacher threads and unrelated forwards retain the original behavior.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
from threading import Lock

import torch

_state = ContextVar("openmopd_attention_metadata", default=None)
_install_lock = Lock()


def install_unpad_metadata_cache():
    import transformers.modeling_flash_attention_utils as flash

    with _install_lock:
        original = flash._get_unpad_data
        if getattr(original, "_openmopd_metadata_dispatcher", False):
            return

        @wraps(original)
        def get_unpad_data(mask):
            state = _state.get()
            # Inference tensors do not track in-place changes. Do not cache them.
            if state is None or torch.is_inference(mask):
                return original(mask)
            stream = torch.cuda.current_stream(mask.device).cuda_stream if mask.is_cuda else None
            key = (id(mask), mask._version, stream)
            entry = state["entries"].get(key)
            if entry is not None and entry[0] is mask:
                state["hits"] += 1
                return entry[1]
            result = original(mask)
            # Keep a strong reference to prevent Python id reuse in this forward.
            state["entries"][key] = (mask, result)
            state["misses"] += 1
            return result

        get_unpad_data._openmopd_metadata_dispatcher = True
        flash._get_unpad_data = get_unpad_data


@contextmanager
def reuse_unpad_metadata(enabled=True):
    if not enabled:
        token = _state.set(None)
        try:
            yield None
        finally:
            _state.reset(token)
        return
    install_unpad_metadata_cache()
    state = {"entries": {}, "hits": 0, "misses": 0}
    token = _state.set(state)
    try:
        yield state
    finally:
        _state.reset(token)
        state["entries"].clear()

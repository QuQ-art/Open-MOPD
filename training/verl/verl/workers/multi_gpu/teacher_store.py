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

"""Shared teacher storage, independent of RewardModelWorker forward execution."""
import torch
import torch.distributed as dist

from .cpu_state_store import CPUStateStore, distributed_tensor


class SharedTeacherStore:
    def __init__(self, model_id):
        self.model_id = model_id
        self.store = CPUStateStore.from_environment()

    def load_model(self, pretrained, empty):
        """Only rank zero reads checkpoint tensors; peers build meta structure."""
        message = [None]
        model = None
        if dist.get_rank() == 0:
            try:
                model = pretrained()
                message[0] = {"ok": True}
            except Exception as error:
                message[0] = {"error": repr(error)}
        dist.broadcast_object_list(message, src=0)
        if "error" in message[0]:
            raise RuntimeError("owner model load failed: " + message[0]["error"])
        if model is None:
            with torch.device("meta"):
                model = empty()
        # Preserve tied parameter objects (replace their data once, not names).
        parameters = list(model.named_parameters(remove_duplicate=False))
        tensors = parameters + list(model.named_buffers(remove_duplicate=False))
        descriptors = [None]
        if dist.get_rank() == 0:
            try:
                aliases = {}
                entries = []
                for name, value in tensors:
                    identity = id(value)
                    if identity not in aliases:
                        aliases[identity] = self.store.allocate(self.model_id + "/init", name, value.shape,
                                                               value.dtype, source=value)[1]
                    entries.append((name, aliases[identity]))
                descriptors[0] = {"tensors": entries}
            except Exception as error:
                descriptors[0] = {"error": repr(error)}
        dist.broadcast_object_list(descriptors, src=0)
        if "error" in descriptors[0]:
            raise RuntimeError("shared model initialization failed: " + descriptors[0]["error"])
        shared = {name: self.store.attach(desc) for name, desc in descriptors[0]["tensors"]}
        # assign=True swaps meta parameters; tie_weights restores aliasing.
        # Include nonpersistent buffers, which state_dict deliberately omits.
        model.load_state_dict({key: shared[key] for key in model.state_dict()}, assign=True)
        # assign=True can create different Parameter objects for tied names.
        # Restore identity as well as storage aliasing, including non-HF modules.
        aliases = {}
        for name, original in parameters:
            parent, _, leaf = name.rpartition(".")
            module = model.get_submodule(parent)
            if id(original) not in aliases:
                aliases[id(original)] = module._parameters[leaf]
            module._parameters[leaf] = aliases[id(original)]
        for name, _ in model.named_buffers():
            parent, _, leaf = name.rpartition(".")
            model.get_submodule(parent)._buffers[leaf] = shared[name]
        if hasattr(model, "tie_weights"):
            model.tie_weights()
        return model

    @torch.no_grad()
    def bind_fsdp(self, model):
        """Initialize FSDP on GPU, then offload directly to the shared mapping.

        Construct with CPUOffload(False). Lazy initialization then allocates no
        private pinned parameter/gradient replica on any rank. Inference-only
        CPU offload is enabled after replacing the actual H2D source.
        """
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp._runtime_utils import _lazy_init
        _lazy_init(model, model)
        for name, module in model.named_modules():
            if not isinstance(module, FSDP) or module._handle is None:
                continue
            handle = module._handle
            if handle.uses_sharded_strategy:
                raise ValueError("shared teacher storage requires NO_SHARD")
            flat = handle.flat_param
            host = distributed_tensor(self.store, self.model_id, name or "root", flat.shape,
                                      flat.dtype, source=flat.detach())
            handle._offload_params = True
            module.cpu_offload.offload_params = True
            flat.data = host
            flat._local_shard = host
            # Retain an explicit source and detect accidental private copies.
            flat._openmopd_shared_cpu = host
            flat._cpu_grad = torch.empty(0, dtype=flat.dtype)
        model._openmopd_shared_teacher = self
        from verl.utils.fsdp_utils import release_fsdp_cpu_offloaded_param_views
        release_fsdp_cpu_offloaded_param_views(model)
        self.finish_initialization()

    def assert_shared_sources(self, model):
        """Fail before scoring if FSDP has replaced the writable shared H2D source."""
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        for name, module in model.named_modules():
            if not isinstance(module, FSDP) or module._handle is None:
                continue
            flat = module._handle.flat_param
            source = getattr(flat, "_openmopd_shared_cpu", None)
            if (source is None or source.device.type != "cpu"
                    or flat._local_shard.data_ptr() != source.data_ptr()):
                raise RuntimeError(f"{self.model_id}/{name}: teacher H2D source is no longer shared CPU storage")

    def finish_initialization(self):
        dist.barrier()
        if dist.get_rank() == 0:
            self.store.release_model(self.model_id + "/init")
        dist.barrier()
        self.store.close()

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

"""Synchronous replicated-DP components; independent of single-GPU pipelines.

Shared mode uses named CPU mappings and an authoritative optimizer writer.
Separate mode retains independent optimizer states. FSDP owns gradient reduction.
"""
import torch
import torch.distributed as dist


class MultiGPUStateStore:
    """Ownership interface for lazily allocated optimizer masters and moments."""

    def __init__(self, mode, owner_rank):
        self.mode = mode
        self.owner_rank = owner_rank

    @property
    def owns_state(self):
        return self.mode == "separate" or dist.get_rank() == self.owner_rank


class MultiGPUGradientSync:
    """FSDP NO_SHARD reduces gradients during backward, exactly once."""

    @staticmethod
    def sharding_strategy():
        from torch.distributed.fsdp import ShardingStrategy
        return ShardingStrategy.NO_SHARD


class MultiGPUWeightPublisher:
    """Publish one authoritative policy before the next rollout weight copy."""

    def __init__(self, owner_rank, verify=True):
        self.owner_rank = owner_rank
        self.verify = verify
        self.version = 0

    @torch.no_grad()
    def publish(self, parameters):
        parameters = list(parameters)
        for parameter in parameters:
            dist.broadcast(parameter.data, src=self.owner_rank)
        self.version += 1
        self.assert_version(parameters)

    def assert_version(self, parameters):
        parameters = list(parameters)
        if self.verify and parameters:
            version = torch.tensor([self.version, -self.version], device=parameters[0].device, dtype=torch.int64)
            dist.all_reduce(version, op=dist.ReduceOp.MAX)
            if version[0].item() != -version[1].item():
                raise RuntimeError("multi_gpu policy version mismatch before rollout publication")


class MultiGPUOptimizer(torch.optim.Optimizer):
    """Synchronous optimizer with owner-only or per-rank state allocation."""

    def __init__(self, optimizer, store, publisher):
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state
        self.store = store
        self.publisher = publisher

    def step(self, closure=None):
        if closure is not None:
            raise ValueError("MultiGPUOptimizer does not support closures")
        result = self.optimizer.step() if self.store.owns_state else None
        self.publisher.publish(p for group in self.param_groups for p in group["params"])
        return result

    def zero_grad(self, set_to_none=True):
        self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        # FSDP checkpoint code needs the same optimizer schema on every rank.
        state = [self.optimizer.state_dict() if self.store.owns_state else None]
        if self.store.mode == "shared":
            dist.broadcast_object_list(state, src=self.store.owner_rank)
        return state[0]

    def load_state_dict(self, state_dict):
        if self.store.owns_state:
            self.optimizer.load_state_dict(state_dict)
            self.state = self.optimizer.state
            self.param_groups = self.optimizer.param_groups
        else:
            # Restore scheduling options without allocating another copy of moments.
            for current, saved in zip(self.param_groups, state_dict["param_groups"], strict=True):
                current.update({key: value for key, value in saved.items() if key != "params"})


def build_multi_gpu_optimizer(optimizer, config):
    store = MultiGPUStateStore(config.get("cpu_state_mode", "shared"), config.get("optimizer_owner_rank", 0))
    publisher = MultiGPUWeightPublisher(store.owner_rank, config.get("verify_policy_version", True))
    if config.get("cpu_state_mode", "shared") == "shared":
        from .shared_adam import SharedCPUAdamW
        if type(optimizer).__name__ not in ("AdamW", "BF16StochasticAdamW"):
            raise ValueError("shared CPU state requires AdamW or BF16StochasticAdamW")
        return SharedCPUAdamW(optimizer, store.owner_rank, publisher)
    return MultiGPUOptimizer(optimizer, store, publisher)

from .cpu_state_store import CPUStateStore
from .teacher_store import SharedTeacherStore

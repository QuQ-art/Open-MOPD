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

"""Named, writable MAP_SHARED tensors. RPC transports descriptors, never tensors.

One directory per training run, shared by all local Ray processes. Only the
creator writes a new allocation; followers attach after descriptor publication.
Unlink is a driver operation after all distributed work has stopped. Tensor
objects retain their mmap, so dropping a store never invalidates live views.
"""
import contextlib
import fcntl
import hashlib
import json
import mmap
import os
from pathlib import Path
import shutil
import uuid

import torch

ENV_NAME = "OPENMOPD_CPU_STATE_NAMESPACE"


class CPUStateStore:
    def __init__(self, namespace, root="/dev/shm", *, owner=False):
        if not namespace or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in namespace):
            raise ValueError("invalid shared CPU state namespace")
        self.namespace = namespace
        self.root = Path(root)
        self.path = self.root / ("openmopd-" + namespace)
        self.owner = owner
        if owner:
            self.path.mkdir(mode=0o700, exist_ok=False)
        if not self.path.is_dir():
            raise RuntimeError(f"shared CPU state namespace does not exist: {self.path}")
        self._views = {}

    @classmethod
    def create(cls):
        return cls(uuid.uuid4().hex, owner=True)

    @classmethod
    def from_environment(cls):
        namespace = os.environ.get(ENV_NAME)
        if not namespace:
            raise RuntimeError("shared CPU state requires a driver-created " + ENV_NAME)
        return cls(namespace)

    @contextlib.contextmanager
    def lock(self):
        with open(self.path / "lock", "a+b") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def allocate(self, model, name, shape, dtype, *, source=None, fill=0):
        """Allocate once under a lock, initialize fully, then publish metadata."""
        key = hashlib.sha256((model + "\0" + name).encode()).hexdigest()
        shape = list(shape)
        dtype_name = str(dtype).removeprefix("torch.")
        element_size = torch.empty((), dtype=dtype).element_size()
        import math
        nbytes = math.prod(shape) * element_size
        descriptor = dict(namespace=self.namespace, root=str(self.root), key=key,
                          model=model, name=name, shape=shape, dtype=dtype_name,
                          offset=0, nbytes=nbytes)
        with self.lock():
            metadata = self.path / (key + ".json")
            if metadata.exists():
                existing = json.loads(metadata.read_text())
                if existing != descriptor:
                    raise ValueError(f"shared tensor layout mismatch: {model}/{name}")
                return self.attach(existing), existing
            fd = os.open(self.path / key, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                # Reserve tmpfs pages now: fail with ENOSPC rather than SIGBUS
                # on a later tensor access when /dev/shm is exhausted.
                os.posix_fallocate(fd, 0, max(nbytes, 1))
            except BaseException:
                os.unlink(self.path / key)
                raise
            finally:
                os.close(fd)
            try:
                tensor = self.attach(descriptor)
                with torch.no_grad():
                    if source is None:
                        tensor.fill_(fill)
                    else:
                        tensor.copy_(source)
                metadata.write_text(json.dumps(descriptor))
            except BaseException:
                self._views.pop(key, None)
                (self.path / key).unlink(missing_ok=True)
                raise
        return tensor, descriptor

    def attach(self, descriptor):
        if descriptor["namespace"] != self.namespace or descriptor["root"] != str(self.root):
            raise ValueError("shared tensor belongs to another namespace")
        key = descriptor["key"]
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError("invalid shared tensor key")
        if key not in self._views:
            with open(self.path / key, "r+b", buffering=0) as stream:
                mapping = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_WRITE)
            dtype = getattr(torch, descriptor["dtype"])
            if descriptor["nbytes"]:
                tensor = torch.frombuffer(mapping, dtype=dtype, offset=descriptor["offset"]).view(descriptor["shape"])
            else:
                tensor = torch.empty(descriptor["shape"], dtype=dtype)
            self._views[key] = tensor
        return self._views[key]

    def inventory(self):
        with self.lock():
            return [json.loads(path.read_text()) for path in sorted(self.path.glob("*.json"))]

    def release_model(self, model):
        """After a collective barrier, remove temporary initialization tensors."""
        with self.lock():
            for metadata in self.path.glob("*.json"):
                descriptor = json.loads(metadata.read_text())
                if descriptor["model"] == model:
                    (self.path / descriptor["key"]).unlink()
                    metadata.unlink()
                    self._views.pop(descriptor["key"], None)

    def close(self):
        self._views.clear()  # Live torch views retain mappings independently.

    def unlink(self):
        if not self.owner:
            raise PermissionError("only the namespace owner may unlink all state")
        shutil.rmtree(self.path)
        self.close()


def distributed_tensor(store, model, name, shape, dtype, *, source=None, owner_rank=0):
    """Only the owner allocates; broadcast JSON-sized metadata, never storage."""
    import torch.distributed as dist
    message = [None]
    if dist.get_rank() == owner_rank:
        try:
            _, descriptor = store.allocate(model, name, shape, dtype, source=source)
            message[0] = {"descriptor": descriptor}
        except Exception as error:
            message[0] = {"error": repr(error)}
    dist.broadcast_object_list(message, src=owner_rank)
    if "error" in message[0]:
        raise RuntimeError("shared CPU allocation failed: " + message[0]["error"])
    return store.attach(message[0]["descriptor"])

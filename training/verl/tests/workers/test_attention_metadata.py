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
import threading
import unittest

import torch
import transformers.modeling_flash_attention_utils as flash

from verl.workers.attention_metadata import reuse_unpad_metadata


class TestAttentionMetadata(unittest.TestCase):
    def test_masks_mutation_and_forward_scope(self):
        mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        reference = flash._get_unpad_data(mask)
        with reuse_unpad_metadata() as state:
            a = flash._get_unpad_data(mask)
            b = flash._get_unpad_data(mask)
            self.assertIs(a, b)
            for x, y in zip(a[:2], reference[:2]):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            self.assertEqual(a[2], reference[2])
            mask[1, 0] = 0
            c = flash._get_unpad_data(mask)
            self.assertEqual(c[2], 2)
            self.assertIsNot(c, a)
            flash._get_unpad_data(mask.clone())
            self.assertEqual((state["hits"], state["misses"]), (1, 3))
        self.assertFalse(state["entries"])
        with reuse_unpad_metadata() as fresh:
            flash._get_unpad_data(mask)
            self.assertEqual((fresh["hits"], fresh["misses"]), (0, 1))

    def test_thread_nested_context_and_exception_isolation(self):
        mask = torch.tensor([[0, 1, 1]])
        with reuse_unpad_metadata() as outer:
            first = flash._get_unpad_data(mask)
            result = []
            thread = threading.Thread(target=lambda: result.append(flash._get_unpad_data(mask)))
            thread.start()
            thread.join()
            self.assertIsNot(first, result[0])
            with self.assertRaisesRegex(RuntimeError, "injected"):
                with reuse_unpad_metadata() as inner:
                    self.assertIsNot(first, flash._get_unpad_data(mask))
                    raise RuntimeError("injected")
            self.assertFalse(inner["entries"])
            self.assertIs(first, flash._get_unpad_data(mask))
            self.assertEqual((outer["hits"], outer["misses"]), (1, 1))
        self.assertIsNot(first, flash._get_unpad_data(mask))

    def test_unversioned_inference_tensor_falls_back(self):
        with torch.inference_mode():
            mask = torch.ones(1, 3, dtype=torch.long)
            with reuse_unpad_metadata() as state:
                first = flash._get_unpad_data(mask)
                mask[0, 0] = 0
                second = flash._get_unpad_data(mask)
                self.assertEqual((first[2], second[2]), (3, 2))
                self.assertEqual((state["hits"], state["misses"]), (0, 0))

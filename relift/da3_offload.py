# SPDX-License-Identifier: Apache-2.0
# Backbone traversal adapted from Depth Anything 3, Copyright (c) Meta Platforms,
# Inc. and affiliates, and ByteDance Ltd. Apache License 2.0. See
# docs/licenses/DA3-APACHE-2.0.txt and packages/frontends/upstreams.json.
"""Memory placement adapter for the pinned DA3 inference implementation.

The backbone traversal follows DA3's Apache-2.0 DINO implementation (Meta,
ByteDance, revision recorded in packages/frontends/upstreams.json). Operations,
reference selection and temporal context are preserved; inactive block weights
and retained features move to CPU or memory-mapped torch archives.
"""
from pathlib import Path
import tempfile
import shutil
import weakref
from types import MethodType

import torch


class ColdStorage:
    def __init__(self, mode, directory, ram_budget_gb=18, scratch_budget_gb=64):
        self.mode = mode
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="da3-", dir=directory)
        self.root = Path(self.temporary.name)
        self.serial = 0
        self.bytes_written = 0
        self.ram_limit = ram_budget_gb * 2**30
        self.disk_limit = scratch_budget_gb * 2**30
        self.references = []
        self.peak_retained_cpu_bytes = 0

    def put(self, tensor):
        cpu = tensor.detach().to("cpu").contiguous()
        if self.mode == "ram":
            live, sizes = [], {}
            for reference in self.references:
                value = reference()
                if value is not None:
                    live.append(reference)
                    sizes[value.untyped_storage().data_ptr()] = value.untyped_storage().nbytes()
            sizes[cpu.untyped_storage().data_ptr()] = cpu.untyped_storage().nbytes()
            retained = sum(sizes.values())
            if retained > self.ram_limit:
                raise MemoryError("Retained DA3 tensors exceed the RAM budget; select disk offload")
            self.references = live + [weakref.ref(cpu)]
            self.peak_retained_cpu_bytes = max(self.peak_retained_cpu_bytes, retained)
            return cpu
        size = cpu.untyped_storage().nbytes()
        if self.bytes_written + size > self.disk_limit or size > shutil.disk_usage(self.root).free * .9:
            raise MemoryError("DA3 scratch tensor would exceed the disk budget or free disk space")
        path = self.root / f"tensor-{self.serial:05d}.pt"
        self.serial += 1
        torch.save(cpu, path)
        self.bytes_written += path.stat().st_size
        return torch.load(path, mmap=True, map_location="cpu", weights_only=True)

    def close(self):
        self.temporary.cleanup()


def _paged_intermediates(self, x, n=1, export_feat_layers=None, **kwargs):
    """Same backbone traversal as DA3, retaining selected features off CUDA."""
    from depth_anything_3.model.reference_view_selector import reorder_by_reference, restore_original_order, select_reference_view
    from depth_anything_3.utils.constants import THRESH_FOR_REF_SELECTION

    export_feat_layers = export_feat_layers or []
    storage = self._relift_cold_storage
    device = x.device
    batch, sequence, _, height, width = x.shape
    x = self.prepare_tokens_with_masks(x)
    selected = range(len(self.blocks) - n, len(self.blocks)) if isinstance(n, int) else n
    position, position_global = self._prepare_rope(batch, sequence, height, width, device)
    outputs, auxiliary = [], []
    reference = None
    for index, block in enumerate(self.blocks):
        global_position, local_position = (None, None) if index < self.rope_start or self.rope is None else (position_global, position)
        if self.alt_start != -1 and index == self.alt_start - 1 and x.shape[1] >= THRESH_FOR_REF_SELECTION and kwargs.get("cam_token") is None:
            reference = select_reference_view(x, strategy=kwargs.get("ref_view_strategy", "saddle_balanced"))
            x = reorder_by_reference(x, reference)
            local_x = reorder_by_reference(local_x, reference)
        if self.alt_start != -1 and index == self.alt_start:
            camera_token = kwargs.get("cam_token")
            if camera_token is None:
                camera_token = torch.cat([self.camera_token[:, :1].expand(batch, -1, -1),
                                          self.camera_token[:, 1:].expand(batch, sequence - 1, -1)], dim=1)
            x[:, :, 0] = camera_token
        if self.alt_start != -1 and index >= self.alt_start and index % 2 == 1:
            x = self.process_attention(x, block, "global", pos=global_position, attn_mask=kwargs.get("attn_mask"))
        else:
            x = self.process_attention(x, block, "local", pos=local_position)
            local_x = x
        if index in selected:
            feature = torch.cat([local_x, x], dim=-1) if self.cat_token else x
            if reference is not None:
                feature = restore_original_order(feature, reference)
            outputs.append(storage.put(feature))
            del feature
        if index in export_feat_layers:
            auxiliary.append(storage.put(x))
    # Release the active backbone before normalizing retained feature maps.
    del x, local_x
    normalized, camera_tokens = [], []
    for item in range(len(outputs)):
        feature_cpu = outputs[item]
        outputs[item] = None
        camera_tokens.append(feature_cpu[:, :, 0].clone())
        feature = feature_cpu.to(device)
        if feature.shape[-1] == self.embed_dim:
            feature = self.norm(feature)
        elif feature.shape[-1] == self.embed_dim * 2:
            feature = torch.cat([feature[..., :self.embed_dim], self.norm(feature[..., self.embed_dim:])], dim=-1)
        else:
            raise ValueError(f"Unexpected DA3 feature shape {feature.shape}")
        feature = feature[..., 1 + self.num_register_tokens:, :]
        normalized.append(storage.put(feature))
        del feature, feature_cpu
    aux = []
    for feature_cpu in auxiliary:
        feature = self.norm(feature_cpu.to(device))[..., 1 + self.num_register_tokens:, :]
        aux.append(storage.put(feature))
        del feature
    return tuple(zip(normalized, camera_tokens)), aux


class DA3Pager:
    """Page transformer blocks; stage only the head's existing eight-frame slices."""
    def __init__(self, model, mode="ram", device="cuda", scratch_dir="cache/offload", ram_budget_gb=18, scratch_budget_gb=64):
        self.device = torch.device(device)
        self.storage = ColdStorage(mode, scratch_dir, ram_budget_gb, scratch_budget_gb)
        self.handles, self.originals, self.weights = [], [], []
        self.transferred_weight_bytes = 0
        model.to(device)
        model.device = self.device
        for module in model.modules():
            if module.__class__.__name__ == "DinoVisionTransformer":
                module._relift_cold_storage = self.storage
                self.originals.append((module, "get_intermediate_layers", module.__dict__.get("get_intermediate_layers")))
                module.get_intermediate_layers = MethodType(_paged_intermediates, module)
                for block in module.blocks:
                    self.page_block(block)
            elif module.__class__.__name__ in ("DPT", "DualDPT"):
                original = module._forward_impl
                self.originals.append((module, "_forward_impl", module.__dict__.get("_forward_impl")))
                def head_forward(features, *args, _original=original, **kwargs):
                    active = [feature.to(self.device) for feature in features]
                    return _original(active, *args, **kwargs)
                module._forward_impl = head_forward
            elif module.__class__.__name__ == "CameraDec":
                self.handles.append(module.register_forward_pre_hook(lambda mod, inputs: tuple(x.to(self.device) if torch.is_tensor(x) else x for x in inputs)))

    def page_block(self, block):
        block.to("cpu")
        weights = []
        for parameter in block.parameters():
            cpu = self.storage.put(parameter.data)
            parameter.data = cpu
            weights.append((parameter, cpu))
        buffers = []
        for module in block.modules():
            for name, value in module._buffers.items():
                if value is not None:
                    cpu = self.storage.put(value)
                    module._buffers[name] = cpu
                    buffers.append((module, name, cpu))
        self.weights.append(weights)
        def before(module, inputs):
            for parameter, cpu in weights:
                parameter.data = cpu.to(self.device)
                self.transferred_weight_bytes += cpu.numel() * cpu.element_size()
            for owner, name, cpu in buffers:
                owner._buffers[name] = cpu.to(self.device)
        def after(module, inputs, output):
            for parameter, cpu in weights:
                parameter.data = cpu
            for owner, name, cpu in buffers:
                owner._buffers[name] = cpu
            return output
        self.handles.append(block.register_forward_pre_hook(before))
        self.handles.append(block.register_forward_hook(after, always_call=True))

    def close(self):
        for handle in self.handles:
            handle.remove()
        for module, name, original in self.originals:
            if original is None:
                delattr(module, name)
            else:
                setattr(module, name, original)
            if hasattr(module, "_relift_cold_storage"):
                del module._relift_cold_storage
        self.handles.clear()
        self.originals.clear()
        self.weights.clear()
        self.storage.close()

from copy import deepcopy
import os

import numpy as np
import pytest
import torch

from relift.da3_offload import ColdStorage,DA3Pager
from relift.da3_packing import FeatureArchive,pack_tensors
from relift.native_backend import load_cpu
from test_da3_offload import tiny_backbone,assert_nested_equal


@pytest.fixture(params=["python","cpp"])
def backend(request):
    if request.param == "cpp" and not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("explicit native build required")
    return request.param,load_cpu() if request.param == "cpp" else None


@pytest.mark.parametrize("mode,weights,features",[("ram",True,False),("disk",True,False),("disk",False,True),("disk",True,True)])
def test_packed_backbone_matches_traversal_state_and_repeated_calls(tmp_path,backend,mode,weights,features):
    name,native=backend
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    pager=None
    try:
        torch.manual_seed(7)
        original=tiny_backbone(); candidate=deepcopy(original)
        images=torch.randn(1,5,3,28,28)
        state={key:value.clone() for key,value in candidate.state_dict().items()}
        with torch.inference_mode():
            expected=original.get_intermediate_layers(images,n=[1,3],export_feat_layers=[0,3],ref_view_strategy="saddle_balanced")
            pager=DA3Pager(candidate,mode=mode,device="cpu",scratch_dir=tmp_path,
                           weight_backend=name if weights else "current",feature_backend=name if features else "current",native=native)
            for _ in range(2):
                actual=candidate.get_intermediate_layers(images,n=[1,3],export_feat_layers=[0,3],ref_view_strategy="saddle_balanced")
                assert_nested_equal(expected,actual)
            for key,value in candidate.state_dict().items(): assert torch.equal(value,state[key])
            if features: assert pager.features.verify()
        root=pager.storage.root
        pager.close(); pager=None
        assert not root.exists()
        assert all(not block._forward_hooks and not block._forward_pre_hooks for block in candidate.blocks)
    finally:
        if pager: pager.close()
        torch.set_num_threads(previous)


@pytest.mark.parametrize("dtype",[torch.float32,torch.float16,torch.bfloat16,torch.int64])
def test_archive_values_private_mapping_lifetime_and_corruption(tmp_path,backend,dtype):
    name,native=backend
    owner=ColdStorage("disk",tmp_path)
    archive=FeatureArchive(owner,name,native)
    source=torch.arange(24).to(dtype).reshape(3,8).T
    mapped=archive.put(source)
    assert torch.equal(mapped,source) and archive.verify()
    mapped[0,0]+=1
    assert archive.verify()  # private mapped changes never modify archived bytes
    record=archive.records[0]
    with pytest.raises(ValueError): archive.map(dict(record,offset=-1))
    with archive.path.open("r+b") as stream: stream.seek(-1,2); stream.write(b"X")
    with pytest.raises(ValueError,match="checksum"): archive.verify()
    expected=mapped.clone(); owner.close()
    assert torch.equal(mapped,expected)


def test_budget_alias_rejection_and_failed_append(tmp_path):
    value=torch.arange(16,dtype=torch.float32)
    with pytest.raises(ValueError,match="aliases"):
        pack_tensors([value,value[2:]])
    owner=ColdStorage("disk",tmp_path,scratch_budget_gb=1e-9)
    archive=FeatureArchive(owner)
    with pytest.raises(MemoryError): archive.put(value)
    assert not archive.path.exists()
    owner.close()


def test_partial_pager_setup_cleans_hooks_and_scratch(tmp_path):
    count=0
    def fail(tensors,backend,native):
        nonlocal count
        count+=1
        if count == 2: raise RuntimeError("intentional setup failure")
        return pack_tensors(tensors)
    model=tiny_backbone()
    method=model.get_intermediate_layers.__func__
    with pytest.raises(RuntimeError,match="setup failure"):
        DA3Pager(model,device="cpu",scratch_dir=tmp_path,weight_backend="python",block_packer=fail)
    assert model.get_intermediate_layers.__func__ is method
    assert all(not block._forward_hooks and not block._forward_pre_hooks for block in model.blocks)
    assert not list(tmp_path.glob("da3-*"))


def test_failed_append_is_rolled_back(tmp_path):
    value=torch.arange(16,dtype=torch.float32)
    class FailingNative:
        def write_payload(self,fd,*args):
            os.write(fd,b"X")
            raise RuntimeError("intentional")
    owner=ColdStorage("disk",tmp_path)
    archive=FeatureArchive(owner,"cpp",FailingNative())
    with pytest.raises(RuntimeError,match="intentional"): archive.put(value)
    assert archive.path.stat().st_size == 0 and not archive.records
    owner.close()

import hashlib
import os

import numpy as np
import pytest

from relift.shard_packing import ShardPacker
from relift.storage import TrajectoryStore, atomic_npy, file_sha256


@pytest.fixture(params=["python","cpp"])
def packer(request):
    if request.param == "cpp" and not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("explicit native build required")
    return ShardPacker(request.param)


@pytest.mark.parametrize("dtype",[np.float32,np.float16,np.bool_])
def test_packing_and_npy_bytes_match_current(tmp_path,packer,dtype):
    source = np.arange(60).reshape(3,4,5).astype(dtype)
    for index,value in enumerate((source,source.transpose(1,0,2),source[:,:,::-1],np.asfortranarray(source))):
        packed,finite = packer("observed" if dtype == np.bool_ else "xyz",value)
        assert finite
        np.testing.assert_array_equal(packed,value)
        control,candidate = tmp_path/f"control-{index}.npy",tmp_path/f"candidate-{index}.npy"
        atomic_npy(control,value,hash_write=True)
        writer = packer.write if packer.backend == "cpp" else None
        digest = atomic_npy(candidate,packed,hash_write=True,payload_writer=writer)
        assert digest == file_sha256(candidate) == file_sha256(control)
        if packer.backend == "cpp":
            # Writer-only selection also accepts the original strided source.
            digest=atomic_npy(candidate,value,hash_write=True,payload_writer=packer.write)
            assert digest == file_sha256(control)


def test_nonfinite_rejected_and_fsync_failure_never_commits(tmp_path,packer,monkeypatch):
    store = TrajectoryStore.create(tmp_path,(3,1,2,3),{},np.arange(2),{},hash_writes=True,
             packer=packer,payload_writer=packer.write if packer.backend == "cpp" else None)
    xyz = np.ones((2,3,3),np.float32)
    bad = xyz.copy(); bad[0,0,0] = np.nan
    with pytest.raises(ValueError,match="nonfinite"):
        store.write_shard(0,xyz=bad,lift=xyz.astype(np.float16),observed=np.ones((2,3),bool))
    assert not store.manifest["shards"]
    def fail(fd): raise OSError("simulated fsync failure")
    with monkeypatch.context() as patch:
        patch.setattr("relift.storage.os.fsync",fail)
        with pytest.raises(OSError,match="simulated"):
            store.write_shard(0,xyz=xyz,lift=xyz.astype(np.float16),observed=np.ones((2,3),bool))
    assert not store.manifest["shards"]
    store.write_shard(0,xyz=xyz,lift=xyz.astype(np.float16),observed=np.ones((2,3),bool))
    store.finish()
    path = store.root/store.manifest["shards"][0]["arrays"]["xyz"]["file"]
    with path.open("r+b") as stream:
        stream.seek(-1,2); stream.write(b"X")
    with pytest.raises(ValueError,match="checksum"):
        store.verify()


def test_native_write_error_is_propagated():
    if not os.environ.get("RELIFT_NATIVE_BUILD"):
        pytest.skip("explicit native build required")
    from relift.native_backend import load_cpu
    fd = os.open(os.devnull,os.O_WRONLY); os.close(fd)
    with pytest.raises(RuntimeError,match="native NPY write"):
        load_cpu().write_payload(fd,b"header",np.arange(10,dtype=np.float32),True)


@pytest.mark.parametrize("change",["dtype","shape"])
def test_injected_packer_cannot_commit_an_invalid_array(tmp_path,change):
    def bad(name,value):
        return (value.astype(np.float16) if change == "dtype" else value[:0]),True
    store=TrajectoryStore.create(tmp_path,(1,1,1,3),{},np.arange(1),{},packer=bad)
    with pytest.raises(ValueError,match="packer changed"):
        store.write_shard(0,xyz=np.ones((1,1,3),np.float32),lift=np.ones((1,1,3),np.float16),observed=np.ones((1,1),bool))
    assert not store.manifest["shards"]

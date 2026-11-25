import numpy as np
import pytest
import torch
from panmamba import build_model, predict, load_weights
from panmamba.data import read_sample
from panmamba.training import reconstruction_loss, make_optimizer, make_scheduler


def test_state_roundtrip(tmp_path):
    model = build_model(width=32)
    state = model.state_dict()
    path = tmp_path/"weights.pth"
    torch.save({"net": {"module."+k:v for k,v in state.items()}},path)
    clone = build_model(width=32)
    load_weights(clone,path)
    assert set(state)==set(clone.state_dict())
    for k,v in state.items():
        torch.testing.assert_close(v,clone.state_dict()[k])
    state.pop(next(iter(state)))
    torch.save({"net": state},path)
    with pytest.raises(RuntimeError):
        load_weights(clone,path)


def test_contract_and_loss(tmp_path):
    path = tmp_path/"scene.npz"
    np.savez(path,ms=np.zeros((4,2,3),np.float32),pan=np.zeros((1,8,12),np.float32),
             target=np.zeros((4,8,12),np.float32))
    assert read_sample(path,require_target=True)["ms"].shape==(4,2,3)
    model=build_model(width=32)
    with pytest.raises(ValueError):
        predict(model,torch.zeros(1,4,2,3),torch.zeros(1,1,8,11))
    x=torch.ones(2,4,8,12)
    y=torch.zeros_like(x)
    torch.testing.assert_close(reconstruction_loss(x,y,4),
                              torch.tensor(1./8))
    assert make_scheduler(make_optimizer(model)) is not None

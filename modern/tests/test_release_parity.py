"""Compare the refactor with the exact public release, not a hand-transcribed model."""
import hashlib
import importlib
from pathlib import Path
import sys
import types

import pytest
import torch
from panmamba import build_model, predict, load_weights

ROOT = Path(__file__).resolve().parents[2]
RELEASE_BLOBS = {
    "pan-sharpening/model/refine.py": "c1b7f90b320ac0cd3a9183a24effe3c945c76861",
    "pan-sharpening/model/panmamba_baseline_finalversion.py": "ceced049ba2063d1fceb01b84c732630bcdd913d",
    "Mamba-block/mamba_simple.py": "2c7e1c84ad285e3a8255017e0ac6e7e0ac864b62"
}

def git_blob_sha(path):
    b = path.read_bytes()
    return hashlib.sha1(f"blob {len(b)}\0".encode() + b).hexdigest()


def test_published_sources_unchanged():
    for path, sha in RELEASE_BLOBS.items():
        assert git_blob_sha(ROOT/path) == sha


def original_net(height=None, width=None):
    package_name = "_published_panmamba_reference"
    package = types.ModuleType(package_name)
    package.__path__ = [str(ROOT / "pan-sharpening/model")]
    sys.modules[package_name] = package
    for name in ("refine", "panmamba_baseline_finalversion"):
        module = types.ModuleType(package_name + "." + name)
        module.__package__ = package_name
        sys.modules[module.__name__] = module
        p = Path(package.__path__[0])/(name+".py")
        source = p.read_text(encoding="utf-8")
        # Remove only unused optional visualization dependencies.
        source = "\n".join(line for line in source.splitlines()
                         if not line.startswith(("import cv2", "import torchvision")))
        if name == "panmamba_baseline_finalversion":
            # Both networks use the SAME released fork. For small tests the
            # sole reference-side math-neutral adaptation is the fixed view shape.
            source = source.replace("from mamba_ssm.modules.mamba_simple import Mamba",
                                    "from panmamba._mamba_simple import Mamba")
            if (height, width) != (1024, 1024):
                assert source.count("view(B, C, 128*8, 128*8)") == 1
                source = source.replace("view(B, C, 128*8, 128*8)",
                                        f"view(B, C, {height}, {width})")
        exec(compile(source, str(p), "exec"), module.__dict__)
    return sys.modules[package_name+".panmamba_baseline_finalversion"].Net

# The three published checkpoints are Git-tracked binary files, not generated test fixtures.
CHECKPOINTS = {
    "gf2": "e7b0c9eb77203adf9522f79388adf8be9670b804",
    "wv2": "71b3d0688155cbda855045b75ef5c5acdabc21b4",
    "wv3": "06c2e741b57abb1fd7c411de3f81f82bddddd249",
}

def checkpoint(sensor):
    p = ROOT/"pan-sharpening"/"checkpoint"/"pretrained"/"Mamba"/("mamba"+sensor)/"bestPSNR.pth"
    assert p.is_file(), f"Missing published checkpoint {p}"
    assert git_blob_sha(p) == CHECKPOINTS[sensor]
    return p


def test_vendored_mamba_is_release():
    from panmamba import _mamba_simple
    p = ROOT/"Mamba-block"/"mamba_simple.py"
    source = p.read_text(encoding="utf-8")
    fixed = source.replace(
        "selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None, None",
        "selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None"
    )
    assert Path(_mamba_simple.__file__).read_text(encoding="utf-8") == fixed


# This deliberately slow PyTorch kernel exists ONLY in tests. It exercises real
# weights and both model graphs under a shared recurrence, not CUDA kernel parity.
def ref_inner(xz, conv_weight, conv_bias, x_proj_weight, delta_proj_weight,
              A, B=None, C=None, D=None, delta_bias=None, delta_softplus=True, **kwargs):
    import torch.nn.functional as F
    x,z = xz.chunk(2,dim=1)
    L = x.shape[-1]
    d = x.shape[1]
    x = F.silu(F.conv1d(x,conv_weight,conv_bias,padding=conv_weight.shape[-1]-1,groups=d)[...,:L])
    proj = F.linear(x.transpose(1,2),x_proj_weight)
    rank = delta_proj_weight.shape[1]
    state_size = A.shape[1]
    dt,b,c = proj.split([rank,state_size,state_size],dim=-1)
    dt = F.linear(dt,delta_proj_weight,delta_bias)
    if delta_softplus:
        dt = F.softplus(dt)
    assert B is None and C is None  # only the released input-dependent path
    x_seq = x.transpose(1,2)
    state = torch.zeros(x.shape[0],d,state_size,device=x.device,dtype=A.dtype)
    ys = []
    for i in range(L):
        step = dt[:,i,:,None].float()
        state = (step*A[None]).exp()*state + step*b[:,i,None,:].float()*x_seq[:,i,:,None].float()
        y = (state*c[:,i,None,:].float()).sum(-1)
        if D is not None:
            y = y + D*x_seq[:,i,:]
        ys.append(y)
    return (torch.stack(ys,dim=-1).to(z.dtype)*F.silu(z))


def ref_projected(xz,conv_weight,conv_bias,x_proj_weight,delta_proj_weight,
                  out_proj_weight,out_proj_bias,A,B=None,C=None,D=None,**kwargs):
    import torch.nn.functional as F
    y = ref_inner(xz,conv_weight,conv_bias,x_proj_weight,delta_proj_weight,A,B,C,D,**kwargs)
    return F.linear(y.transpose(1,2),out_proj_weight,out_proj_bias)


@pytest.mark.parametrize("sensor", ["gf2","wv2","wv3"])
def test_real_checkpoint_strict_and_cpu_reference(sensor,monkeypatch):
    from panmamba import _mamba_simple
    monkeypatch.setattr(_mamba_simple,"mamba_inner_fn",ref_projected)
    monkeypatch.setattr(_mamba_simple,"mamba_inner_fn_no_out_proj",ref_inner)
    old = original_net(8,12)(num_channels=4,base_filter=32,args=None).eval()
    new = build_model().eval()
    assert list(old.state_dict()) == list(new.state_dict())
    load_weights(old,checkpoint(sensor))
    load_weights(new,checkpoint(sensor))
    for k,v in old.state_dict().items():
        torch.testing.assert_close(v,new.state_dict()[k],atol=0,rtol=0)
    torch.manual_seed(42)
    ms = torch.rand(1,4,2,3)
    pan = torch.rand(1,1,8,12)
    with torch.inference_mode():
        a = old(ms,None,pan)
        b = predict(new,ms,pan)
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    print(f"{sensor}: strict official checkpoint load; CPU reference max error {(a-b).abs().max().item():.8g}")


@pytest.mark.parametrize("sensor", ["gf2","wv2","wv3"])
def test_real_checkpoint_cuda_fused_parity(sensor):
    import os
    if os.getenv("PANMAMBA_TEST_CUDA") != "1":
        pytest.skip("Set PANMAMBA_TEST_CUDA=1 on a Vim-compatible CUDA environment")
    from panmamba import _mamba_simple
    assert torch.cuda.is_available()
    assert callable(_mamba_simple.mamba_inner_fn) and callable(_mamba_simple.mamba_inner_fn_no_out_proj)
    size = int(os.getenv("PANMAMBA_TEST_SIZE","128"))
    assert size >= 4 and size % 4 == 0
    old = original_net(size,size)(num_channels=4,base_filter=32,args=None).cuda().eval()
    new = build_model(device="cuda").eval()
    load_weights(old,checkpoint(sensor))
    load_weights(new,checkpoint(sensor))
    torch.manual_seed(123)
    ms = torch.rand(1,4,size//4,size//4,device="cuda")
    pan = torch.rand(1,1,size,size,device="cuda")
    with torch.inference_mode():
        a = old(ms,None,pan)
        b = predict(new,ms,pan)
    torch.cuda.synchronize()
    assert torch.isfinite(a).all() and torch.isfinite(b).all()
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-6)
    print(f"{sensor}: actual fused CUDA max error {(a-b).abs().max().item():.8g}")

"""Inference, synthetic latency benchmark, and paired-data L1 training."""
import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .api import build_model, load_weights, predict
from .data import read_sample, PairedNPZ
from .training import reconstruction_loss, make_optimizer, make_scheduler


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def model_for(a):
    if torch.device(a.device).type != "cuda":
        raise ValueError("Released fused Mamba inference/training requires CUDA; CPU is only used for structural and reference tests")
    model = build_model(device=a.device, width=a.width)
    if a.checkpoint:
        load_weights(model, a.checkpoint, trusted_pickle=a.trusted_checkpoint)
    return model


def infer(a):
    model = model_for(a).eval()
    sample = read_sample(a.input)
    with torch.inference_mode():
        pred = predict(model, sample["ms"][None].to(a.device), sample["pan"][None].to(a.device))
    output = Path(a.output)
    if output.suffix != ".npz":
        raise ValueError("Output must be .npz to retain all four bands")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, hrms=pred[0].float().cpu().numpy())


def benchmark(a):
    if a.repeats < 1 or a.warmup < 0 or min(a.height, a.image_width) < 4 or a.height % 4 or a.image_width % 4:
        raise ValueError("Invalid repetitions or shape")
    model = model_for(a).eval()
    ms = torch.rand(a.batch,4,a.height//4,a.image_width//4,device=a.device)
    pan = torch.rand(a.batch,1,a.height,a.image_width,device=a.device)
    with torch.inference_mode():
        for _ in range(a.warmup):
            predict(model, ms, pan)
        synchronize(a.device)
        if torch.device(a.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(a.device)
        times = []
        for _ in range(a.repeats):
            synchronize(a.device)
            start = time.perf_counter()
            predict(model, ms, pan)
            synchronize(a.device)
            times.append((time.perf_counter() - start)*1000)
    print(json.dumps({"median_ms": statistics.median(times), "batch": a.batch,
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "peak_allocated_bytes": torch.cuda.max_memory_allocated(a.device)
                          if torch.device(a.device).type == "cuda" else None}, indent=2))


def train(a):
    if a.epochs < 1 or a.batch < 1:
        raise ValueError("Positive epoch and batch counts required")
    import random
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    model = model_for(a)
    opt = make_optimizer(model,a.lr)
    sched = make_scheduler(opt)
    loader = DataLoader(PairedNPZ(a.data),batch_size=a.batch,shuffle=True,num_workers=a.workers)
    folder = Path(a.output)
    folder.mkdir(parents=True,exist_ok=True)
    for epoch in range(a.epochs):
        model.train()
        total = 0.
        for sample in loader:
            batch = {k: v.to(a.device) for k,v in sample.items()}
            opt.zero_grad(set_to_none=True)
            loss = reconstruction_loss(predict(model,batch["ms"],batch["pan"]),batch["target"],a.batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),a.clip)
            opt.step()
            total += loss.detach().item()
        sched.step()
        print(json.dumps({"epoch":epoch+1,"loss":total/len(loader)}))
        tmp = folder/"latest.tmp"
        torch.save({"net": model.state_dict(),"epoch": epoch+1,"optimizer":opt.state_dict(),
                    "scheduler":sched.state_dict()},tmp)
        tmp.replace(folder/"latest.pth")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command",required=True)
    for name,fn in (("infer",infer),("benchmark",benchmark),("train",train)):
        p=sub.add_parser(name)
        p.set_defaults(fn=fn)
        p.add_argument("--device",default="cuda")
        p.add_argument("--width",type=int,default=32)
        p.add_argument("--checkpoint",required=name=="infer")
        p.add_argument("--trusted-checkpoint",action="store_true",help="Allow unsafe pickle only for independently trusted checkpoints")
        if name=="infer":
            p.add_argument("--input",required=True)
            p.add_argument("--output",required=True)
        elif name=="benchmark":
            p.add_argument("--batch",type=int,default=1)
            p.add_argument("--height",type=int,default=128)
            p.add_argument("--image-width",type=int,default=128)
            p.add_argument("--warmup",type=int,default=3)
            p.add_argument("--repeats",type=int,default=10)
        else:
            p.add_argument("--data",required=True)
            p.add_argument("--output",required=True)
            p.add_argument("--batch",type=int,default=4)
            p.add_argument("--epochs",type=int,default=500)
            p.add_argument("--workers",type=int,default=0)
            p.add_argument("--seed",type=int,default=123)
            p.add_argument("--lr",type=float,default=5e-4)
            p.add_argument("--clip",type=float,default=4.0)
    a=parser.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()

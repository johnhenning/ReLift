#!/usr/bin/env python
"""Train the corrector from scratch on the PointOdyssey window cache (the released recipe).

    python scripts/train.py --run my_run                      # the released recipe: 9 h of training on one GPU
    python scripts/train.py --run my_run --max_steps 150436   # step-indexed instead (other hardware; a deviation)
    python scripts/train.py --run smoke --overfit --max_steps 600 --workers 0     # sanity: one batch, loss collapses

Every setting comes from the config (configs/train/corrector.yaml, the released run's arguments) and can be
overridden on the command line.  Outputs in <runs_root>/<run>/: split.json, train.csv (every 50 steps), val.jsonl
(every --val_every steps), last.pt (resumed automatically on restart), step_*.pt every 10 validations, final.pt,
provenance.json.  ``last.pt``/``final.pt`` are training checkpoints (model, EMA, optimiser); run_corrector.py loads
them directly (EMA weights), export_checkpoint.py turns final.pt into the release format.

The schedule of the released run is indexed by elapsed training time (validation excluded), so the step count
depends on the GPU: 150,436 steps in 9 h on an RTX PRO 6000 (4.6 steps/s, 26 GB).  --seed makes the sampling of
tracks, windows, lengths, mirroring and the simulated lift reproducible; the released run was unseeded.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from relift.config import REPO, paths  # noqa: E402
from relift.data.pointodyssey import EXCLUDE, split  # noqa: E402
from relift.model import IN_FEATS, Corrector  # noqa: E402
from relift.provenance import machine  # noqa: E402
from relift.train import (  # noqa: E402
    LENGTHS,
    VAL_PER_SCENE,
    Windows,
    ema_update,
    loss_fn,
    lr_at,
    make_batch,
    validate,
)

CONFIG = REPO / "configs" / "train" / "corrector.yaml"
VAL_KEYS = ("obs", "hid", "hid_dyn", "vel_hid_dyn")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--run", required=True, help="name of the run directory")
    ap.add_argument(
        "--windows_root", default=None, help="default <cache_root>/pointodyssey_windows"
    )
    ap.add_argument("--runs_root", default=None, help="default <outputs_root>/runs")
    for k in ("hours", "lr"):
        ap.add_argument(f"--{k}", type=float, default=None)
    for k in ("batch", "max_steps", "val_every", "workers", "dim", "depth", "chunk"):
        ap.add_argument(f"--{k}", type=int, default=None)
    ap.add_argument(
        "--spike_guard",
        action="store_true",
        default=None,
        help="skip steps whose gradient norm is 20x the running median",
    )
    ap.add_argument(
        "--ckpt_act",
        action="store_true",
        default=None,
        help="activation checkpointing (less memory, slower)",
    )
    ap.add_argument(
        "--overfit",
        action="store_true",
        help="sanity: one fixed batch and fixed noise; the loss must collapse",
    )
    ap.add_argument(
        "--seed", type=int, default=None, help="seed all sampling (the released run was unseeded)"
    )
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for k in (
        "hours",
        "lr",
        "batch",
        "max_steps",
        "val_every",
        "workers",
        "dim",
        "depth",
        "chunk",
        "spike_guard",
        "ckpt_act",
    ):
        v = getattr(a, k)
        if v is not None:
            cfg[k] = v
    cfg.setdefault("chunk", 4)
    assert (
        cfg.get("mode", "regress_clean") == "regress_clean"
        and cfg.get("input", "synthetic") == "synthetic"
    ), "the release trains the corrector only"
    assert not cfg.get("qk_norm", False), "qk_norm is not part of the released model"
    P = paths()
    cache = a.windows_root or str(P["cache_root"] / "pointodyssey_windows")
    out = os.path.join(a.runs_root or str(P["outputs_root"] / "runs"), a.run)
    os.makedirs(out, exist_ok=True)
    dev = a.device
    if a.seed is not None:
        random.seed(a.seed), np.random.seed(a.seed), torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_names, val_names = split(cache)
    sp = os.path.join(out, "split.json")
    if os.path.exists(sp):  # a resumed run keeps ITS split
        s = json.load(open(sp))
        train_names, val_names = s["train"], s["val"]
    json.dump(
        dict(
            train=train_names,
            val=val_names,
            excluded=sorted(EXCLUDE),
            data=cfg.get("data", "pointodyssey_strides_1_2"),
            input="synthetic",
        ),
        open(sp, "w"),
        indent=1,
    )
    if a.overfit:
        train_names = train_names[:2]
    t0 = time.time()
    train = Windows(cache, train_names)
    val = Windows(cache, val_names, fixed=True, per_scene=VAL_PER_SCENE)
    print(
        f"data: {len(train)} train windows ({len(train_names)} scenes), {len(val)} val windows ({len(val_names)} scenes), "
        f"loaded in {time.time() - t0:.0f}s from {cache}",
        flush=True,
    )
    loader = torch.utils.data.DataLoader(
        train,
        batch_size=cfg["batch"],
        shuffle=True,
        drop_last=True,
        num_workers=cfg["workers"],
        pin_memory=True,
        persistent_workers=cfg["workers"] > 0,
    )
    vloader = torch.utils.data.DataLoader(
        val, batch_size=cfg["batch"], shuffle=False, num_workers=0
    )

    arch = dict(
        dim=cfg["dim"],
        depth=cfg["depth"],
        heads=cfg["dim"] // 64,
        in_feats=IN_FEATS,
        chunk=cfg["chunk"],
    )
    model, ema = Corrector(**arch).to(dev), Corrector(**arch).to(dev)
    ema.load_state_dict(model.state_dict())
    ema.requires_grad_(False)
    model.ckpt = bool(cfg["ckpt_act"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01, betas=(0.9, 0.95))
    print(
        f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f} M parameters; config {cfg}",
        flush=True,
    )
    # the training-checkpoint `args` that checkpoint.load / export read
    args = dict(
        cfg,
        run=a.run,
        mode="regress_clean",
        input="synthetic",
        seed=a.seed,
        overfit=a.overfit,
        config=os.path.abspath(a.config),
    )

    step, elapsed, skipped, spikes, gn_hist = 0, 0.0, 0, 0, []
    last = os.path.join(out, "last.pt")
    if os.path.exists(last) and not a.overfit:
        ck = torch.load(
            last, map_location=dev, weights_only=False
        )  # our own resume point (optimiser state, counters)
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        opt.load_state_dict(ck["opt"])
        step, elapsed = ck["step"], ck["elapsed"]
        print(f"resumed at step {step}, {elapsed / 3600:.2f} h", flush=True)
    budget = cfg["hours"] * 3600.0
    max_steps = int(cfg["max_steps"])
    if not a.overfit and (
        (max_steps and step >= max_steps) or (not max_steps and elapsed >= budget)
    ):
        print("budget already used; nothing to do", flush=True)
        return

    def save(path: str | Path) -> None:
        tmp = str(path) + ".tmp"
        torch.save(
            dict(
                model=model.state_dict(),
                ema=ema.state_dict(),
                opt=opt.state_dict(),
                step=step,
                elapsed=elapsed,
                args=args,
            ),
            tmp,
        )
        os.replace(tmp, path)

    prov = machine(dev)
    json.dump(dict(prov, args=args), open(os.path.join(out, "provenance.json"), "w"), indent=1)
    print(f"machine: {prov}", flush=True)
    new_log = not os.path.exists(os.path.join(out, "train.csv"))
    log_f = open(os.path.join(out, "train.csv"), "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(["step", "hours", "loss", "pos", "vel", "grad_norm", "lr", "peak_gb"])
    fixed = None
    tick, win = time.time(), []
    done = False
    while not done:
        for batch in loader:
            if a.overfit:
                fixed = fixed or batch
                gen = torch.Generator(device=dev).manual_seed(0)
                b = make_batch(fixed, dev, gen, length=68, augment=False)
                lr = cfg["lr"] * min(1.0, (step + 1) / 100.0)
            else:
                b = make_batch(batch, dev, None, length=LENGTHS[np.random.randint(len(LENGTHS))])
                prog = min(step / max_steps, 1.0) if max_steps else min(elapsed / budget, 1.0)
                lr = lr_at(step, prog, cfg["lr"])
            for g in opt.param_groups:
                g["lr"] = lr
            loss, parts = loss_fn(model, b)
            opt.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                skipped += 1
                print(f"step {step}: non-finite loss, skipped ({skipped})", flush=True)
                if skipped > 50:
                    raise RuntimeError("too many non-finite losses")
                continue
            loss.backward()
            gn = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
            if not math.isfinite(gn):
                skipped += 1
                print(f"step {step}: non-finite gradient, skipped ({skipped})", flush=True)
                continue
            gn_hist.append(gn)
            del gn_hist[:-200]
            if cfg["spike_guard"] and step > 2000 and gn > 20.0 * float(np.median(gn_hist)):
                spikes += 1
                print(
                    f"step {step}: gradient spike {gn:.1f} (median {np.median(gn_hist):.2f}), step skipped ({spikes})",
                    flush=True,
                )
                continue
            opt.step()
            ema_update(ema, model, step)
            step += 1
            now = time.time()
            elapsed += now - tick
            tick = now
            win.append((float(loss.detach()), parts["pos"].cpu(), parts["vel"].cpu(), gn))
            if step % 50 == 0:
                L = float(np.mean([w[0] for w in win]))
                pos = float(torch.cat([w[1] for w in win]).mean())
                vel = float(torch.cat([w[2] for w in win]).mean())
                row = [
                    step,
                    round(elapsed / 3600, 4),
                    L,
                    pos,
                    vel,
                    float(np.mean([w[3] for w in win])),
                    lr,
                    torch.cuda.max_memory_allocated() / 2**30 if dev.startswith("cuda") else 0.0,
                ]
                log.writerow(row)
                log_f.flush()
                print(
                    f"step {step:7d}  {elapsed / 3600:5.2f}h  loss {L:.4f}  pos {pos:.4f}  vel {vel:.5f}  gnorm {row[5]:.2f}  lr {lr:.2e}  "
                    f"mem {row[7]:.1f}G  {step / max(elapsed, 1e-9):.2f} it/s(avg)",
                    flush=True,
                )
                win = []
            if not a.overfit and step % cfg["val_every"] == 0:
                m = validate(ema, vloader, dev)
                m.update(step=step, hours=elapsed / 3600)
                with open(os.path.join(out, "val.jsonl"), "a") as f:
                    f.write(json.dumps(m) + "\n")
                print(
                    "  VAL (% of depth; lower is better)   "
                    + "   ".join(
                        f"{name}: " + "/".join(f"{m[f'{name}/{k}']:.2f}" for k in VAL_KEYS)
                        for name in ("lift", "interp", "smooth", "model", "model_wrongcond")
                    )
                    + "    [obs/hid/hid_dyn/vel_hid_dyn]",
                    flush=True,
                )
                save(last)
                if step % (10 * cfg["val_every"]) == 0:
                    save(os.path.join(out, f"step_{step:07d}.pt"))
                tick = time.time()  # validation is not training time
            if (max_steps and step >= max_steps) or (
                not a.overfit and not max_steps and elapsed >= budget
            ):
                done = True
                break
    if not a.overfit:
        save(last)
        save(os.path.join(out, "final.pt"))
    print(f"TRAIN_DONE {step} steps, {elapsed / 3600:.2f} h of training", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import hashlib
import json
import math
import os
import pathlib
import pickle
import re
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper
import torch
import torch.nn as nn

FORCE = False
OPSET = 17
TOL = 1e-4
STEPS = 8
Z_DIM = 256
Z_NORM_TOL = 1e-2
SONIC_CTX = 4
SONIC_QPOS = 36
SONIC_HEIGHT = 0.788740
SONIC_ENC_OBS = 1762
ROWS_TOL = 0.5
ROWS_CHECK = 16

BATCH_ELEMENTWISE = {
    "Add",
    "And",
    "Div",
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
    "Max",
    "Min",
    "Mod",
    "Mul",
    "Or",
    "Pow",
    "PRelu",
    "Sub",
    "Where",
    "Xor",
}
BATCH_UNARY = {
    "Abs",
    "Acos",
    "Asin",
    "Atan",
    "Cast",
    "Ceil",
    "Clip",
    "Cos",
    "Dropout",
    "Elu",
    "Erf",
    "Exp",
    "Floor",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "IsInf",
    "IsNaN",
    "LeakyRelu",
    "Log",
    "Mish",
    "Neg",
    "Not",
    "Reciprocal",
    "Relu",
    "Round",
    "Selu",
    "Sigmoid",
    "Sign",
    "Sin",
    "Softplus",
    "Softsign",
    "Sqrt",
    "Tan",
    "Tanh",
    "Trilu",
}
BATCH_REDUCE = {
    "ReduceL1",
    "ReduceL2",
    "ReduceLogSum",
    "ReduceLogSumExp",
    "ReduceMax",
    "ReduceMean",
    "ReduceMin",
    "ReduceProd",
    "ReduceSumSquare",
}
BATCH_AXIS_DEFAULT = {
    "ArgMax": 0,
    "ArgMin": 0,
    "LayerNormalization": -1,
    "LogSoftmax": -1,
    "OneHot": -1,
    "Softmax": -1,
    "Split": 0,
    "TopK": -1,
}
BATCH_EINSUM = re.compile(
    r"\.\.\. (\w) (\w) (\w), \.\.\. \1 \2 (\w) -> \.\.\. \1 \3 \4"
)

BFM_ZERO_LATENTS = [
    "move-ego-0-0",
    "move-ego-0-0.3",
    "move-ego-0-0.7",
    "move-ego-90-0.3",
    "move-ego-90-0.7",
    "move-ego-180-0.3",
    "move-ego--90-0.3",
    "move-ego--90-0.7",
    "rotate-z-5-0.5",
    "rotate-z--5-0.5",
    "move-ego-low0.5-0-0",
    "move-ego-low0.6-0-0.7",
]


class RlGym(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.memory = nn.LSTM(47, 64, 1)
        self.actor = nn.Sequential(nn.Linear(64, 32), nn.ELU(), nn.Linear(32, 12))

    def forward(
        self,
        x,
        h,
        c,
    ):
        out, (hn, cn) = self.memory(x.unsqueeze(0), (h, c))
        return self.actor(out.squeeze(0)), hn, cn


class RoboMimic(nn.Module):
    def __init__(
        self,
        eps,
    ):
        super().__init__()
        self.rnn = nn.LSTM(96, 256, 1)
        self.actor = nn.Sequential(
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 29),
        )
        self.register_buffer("mean", torch.zeros(1, 96))
        self.register_buffer("std", torch.ones(1, 96))
        self.eps = eps

    def forward(
        self,
        x,
        h,
        c,
    ):
        x = (x - self.mean) / (self.std + self.eps)
        out, (hn, cn) = self.rnn(x.unsqueeze(0), (h, c))
        return self.actor(out.squeeze(0)), hn, cn


class G1Gym(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.memory = nn.LSTM(90, 64, 1)
        self.actor = nn.Sequential(
            nn.Linear(64, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, 27),
        )

    def forward(
        self,
        x,
        h,
        c,
    ):
        out, (hn, cn) = self.memory(x.unsqueeze(0), (h, c))
        return self.actor(out.squeeze(0)), hn, cn


class Mturan33Loco(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        layers = []
        prev = 57
        for h in (512, 256, 128):
            layers += [nn.Linear(prev, h), nn.LayerNorm(h), nn.ELU()]
            prev = h
        layers.append(nn.Linear(prev, 12))
        self.actor = nn.Sequential(*layers)

    def forward(
        self,
        x,
    ):
        return self.actor(x)


def supermimic_mlp(
    prev,
    hidden,
    out,
):
    layers = []
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers.append(nn.Linear(prev, out))
    mlp = nn.Module()
    mlp.net = nn.Sequential(*layers)
    return mlp


class SuperMimicEncoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = supermimic_mlp(680, (2048, 1024, 512, 256), 32)
        self.register_buffer("scale", torch.ones(680))

    def forward(
        self,
        goal,
    ):
        return torch.tanh(self.mlp.net(goal * self.scale))


class SuperMimicDecoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = supermimic_mlp(32 + 930, (2048, 1024, 512, 256), 29)
        self.register_buffer("scale", torch.ones(930))

    def forward(
        self,
        z,
        obs,
    ):
        return self.mlp.net(torch.cat([z, obs * self.scale], dim=-1))


class SuperMimic(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict({"motion": SuperMimicEncoder()})
        self.decoder = SuperMimicDecoder()

    def forward(
        self,
        obs,
        goal,
    ):
        return self.decoder(self.encoders["motion"](goal), obs)


class SunnyFolded(nn.Module):
    def __init__(
        self,
        mean,
        std,
        w0,
        b0,
        rest,
    ):
        super().__init__()
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        layers = [nn.Linear(99, 512), nn.ELU()]
        layers[0].weight = nn.Parameter(w0)
        layers[0].bias = nn.Parameter(b0)
        for w, b in rest:
            layers.append(nn.Linear(w.shape[1], w.shape[0]))
            layers[-1].weight = nn.Parameter(w)
            layers[-1].bias = nn.Parameter(b)
            layers.append(nn.ELU())
        self.net = nn.Sequential(*layers[:-1])

    def forward(
        self,
        x,
    ):
        return self.net(torch.clamp((x - self.mean) / self.std, -100.0, 100.0))


def export_sunny(
    src,
    scan_path,
    dst,
):
    model = onnx.load(src)
    init = {
        i.name: torch.from_numpy(onnx.numpy_helper.to_array(i).copy())
        for i in model.graph.initializer
    }
    mean, std = init["obs_mean"], init["onnx::Div_71"]

    scan = torch.from_numpy(np.load(scan_path).astype(np.float32))
    norm = torch.clamp((scan - mean[99:]) / std[99:], -100.0, 100.0)
    if float(norm.abs().max()) > TOL:
        raise RuntimeError(
            "download_export_onnx: sunny's shipped scan is not the graph's own mean, "
            "so the scan branch is not constant and cannot be folded"
        )

    cnn = nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1),
        nn.ELU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ELU(),
        nn.Flatten(),
        nn.Linear(4576, 64),
        nn.ELU(),
        nn.Linear(64, 32),
        nn.ELU(),
    )
    cnn[0].weight = nn.Parameter(init["scan_cnn.0.weight"])
    cnn[0].bias = nn.Parameter(init["scan_cnn.0.bias"])
    cnn[2].weight = nn.Parameter(init["scan_cnn.2.weight"])
    cnn[2].bias = nn.Parameter(init["scan_cnn.2.bias"])
    cnn[5].weight = nn.Parameter(init["scan_cnn.5.weight"])
    cnn[5].bias = nn.Parameter(init["scan_cnn.5.bias"])
    cnn[7].weight = nn.Parameter(init["scan_cnn.7.weight"])
    cnn[7].bias = nn.Parameter(init["scan_cnn.7.bias"])
    cnn.eval()
    with torch.no_grad():
        feat = cnn(norm.view(1, 1, 13, 11))[0]

    w0 = init["policy_net.0.weight"]
    net = SunnyFolded(
        mean[:99],
        std[:99],
        w0[:, :99].clone(),
        init["policy_net.0.bias"] + w0[:, 99:] @ feat,
        [
            (init["policy_net.2.weight"], init["policy_net.2.bias"]),
            (init["policy_net.4.weight"], init["policy_net.4.bias"]),
            (init["action_net.weight"], init["action_net.bias"]),
        ],
    )
    net.eval()

    example = (torch.randn(4, 99),)
    with torch.no_grad():
        ref = net(*example)

    torch.onnx.export(
        net,
        example,
        dst,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )

    full = torch.cat([example[0], scan.expand(example[0].shape[0], -1)], dim=1)
    agree(dst, onnx_run(src, ["obs"], (full,)), onnx_run(dst, ["obs"], example))


def export_morphmimic(
    src,
    robots_path,
    robot,
    dst,
):
    with open(robots_path) as f:
        doc = json.load(f)
    entry = next(r for r in doc["robots"] if r["name"] == robot)
    consts = {
        "obs_mean": entry["inputs"]["obsMean"],
        "obs_var": entry["inputs"]["obsVar"],
        "goal_mean": entry["inputs"]["goalMean"],
        "goal_var": entry["inputs"]["goalVar"],
        "morph": entry["inputs"]["morph"],
    }
    keep = ["obs_history", "goal_motion"]

    model = onnx.load(src)
    model = onnx.utils.Extractor(model).extract_model(
        [i.name for i in model.graph.input], ["action_motion"]
    )
    graph = model.graph
    used = {x for n in graph.node for x in n.input}
    extra = sorted(
        i.name
        for i in graph.input
        if i.name in used and i.name not in keep + list(consts)
    )
    if extra:
        raise RuntimeError(
            f"download_export_onnx: {src} motion head also reads {extra}"
        )
    for name, value in consts.items():
        graph.initializer.append(
            numpy_helper.from_array(np.asarray(value, np.float32), name)
        )
    inputs = [i for i in graph.input if i.name in keep]
    del graph.input[:]
    graph.input.extend(inputs)
    onnx.save(model, dst)

    rng = np.random.default_rng(0)
    feed = {
        "obs_history": rng.standard_normal((STEPS, 930)).astype(np.float32),
        "goal_motion": rng.standard_normal((STEPS, 670)).astype(np.float32),
    }
    full = dict(feed)
    full["goal_keypoint"] = np.zeros((STEPS, 450), np.float32)
    full["kp_point_mask"] = np.ones((STEPS, 5), bool)
    full["kp_frame_mask"] = np.ones((STEPS, 10), bool)
    for name, value in consts.items():
        full[name] = np.asarray(value, np.float32)
    full["kp_mean"] = np.asarray(entry["inputs"]["kpMean"], np.float32)
    full["kp_var"] = np.asarray(entry["inputs"]["kpVar"], np.float32)
    one = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    ref = one.run(["action_motion"], full)
    many = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    agree(dst, ref, many.run(None, feed))


SUPERMIMIC680_OBS_SCALE = (1.0, 5.0, 2.0, 1.0, 1.0)
SUPERMIMIC680_OBS_TERMS = (29, 29, 3, 3, 29)
SUPERMIMIC680_GOAL_SCALE = (1.0, 5.0, 1.0, 2.0, 1.0, 1.0)
SUPERMIMIC680_GOAL_TERMS = (29, 29, 3, 3, 3, 1)


def export_supermimic680(
    src,
    dst,
):
    if src.endswith(".onnx"):
        state = {
            t.name: torch.from_numpy(numpy_helper.to_array(t).copy())
            for t in onnx.load(src).graph.initializer
        }
    else:
        with torch.serialization.safe_globals(
            [pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath]
        ):
            state = torch.load(src, map_location="cpu", weights_only=True)
        state = state["policy"]
    net = SuperMimic()
    net.load_state_dict(state)
    net.eval()

    obs_scale = np.concatenate(
        [
            np.full(10 * n, 1.0 / s)
            for n, s in zip(SUPERMIMIC680_OBS_TERMS, SUPERMIMIC680_OBS_SCALE)
        ]
    )
    goal_scale = np.tile(
        np.concatenate(
            [
                np.full(n, 1.0 / s)
                for n, s in zip(SUPERMIMIC680_GOAL_TERMS, SUPERMIMIC680_GOAL_SCALE)
            ]
        ),
        10,
    )
    agree(
        f"{src} scales",
        [obs_scale, goal_scale],
        [
            net.decoder.scale.numpy(),
            net.encoders["motion"].scale.numpy(),
        ],
    )

    names = ["obs_history", "goal"]
    example = (torch.randn(STEPS, 930), torch.randn(STEPS, 680))
    if src.endswith(".onnx"):
        one = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
        ref = [
            np.concatenate(
                [
                    one.run(
                        None,
                        {n: t[i : i + 1].numpy() for n, t in zip(names, example)},
                    )[0]
                    for i in range(STEPS)
                ]
            )
        ]
    else:
        with torch.no_grad():
            ref = [net(*example).numpy()]

    torch.onnx.export(
        net,
        example,
        dst,
        input_names=names,
        output_names=["action"],
        dynamic_axes={n: {0: "batch"} for n in names + ["action"]},
        opset_version=OPSET,
        dynamo=False,
    )
    agree(dst, ref, onnx_run(dst, names, example))


def require_cuda():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "download_export_onnx: a CUDA device is required; amo was traced on one "
            "and holds constants pinned to cuda:0"
        )


def agree(
    tag,
    ref,
    got,
):
    worst = max(float(np.abs(r - g).max()) for r, g in zip(ref, got))
    if worst > TOL:
        raise RuntimeError(
            f"download_export_onnx: {tag} disagrees with its source by {worst:.3e}, "
            f"over the {TOL:.0e} tolerance"
        )


def onnx_run(
    path,
    names,
    tensors,
):
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return session.run(
        None, {n: t.detach().cpu().numpy() for n, t in zip(names, tensors)}
    )


def export_stateless(
    src,
    dst,
    example,
    in_names,
    out_names,
):
    module = torch.jit.load(src, map_location="cuda")
    module.eval()
    example = [t.to("cuda") for t in example]
    with torch.no_grad():
        ref = module(*example)
    ref = [ref] if isinstance(ref, torch.Tensor) else list(ref)

    torch.onnx.export(
        module,
        tuple(example),
        dst,
        input_names=in_names,
        output_names=out_names,
        opset_version=OPSET,
        dynamo=False,
    )
    agree(
        dst, [r.detach().cpu().numpy() for r in ref], onnx_run(dst, in_names, example)
    )


def export_submodule(
    src,
    dst,
    net,
    prefix,
    obs_dim,
):
    state = torch.load(src, map_location="cpu", weights_only=False)["model"]
    wanted = set(net.state_dict())
    weights = {
        k[len(prefix) :]: v
        for k, v in state.items()
        if k.startswith(prefix) and k[len(prefix) :] in wanted
    }
    net.load_state_dict(weights)
    net.eval()

    example = (torch.randn(1, obs_dim),)
    with torch.no_grad():
        ref = net(*example)

    torch.onnx.export(
        net,
        example,
        dst,
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=OPSET,
        dynamo=False,
    )
    agree(dst, [ref.numpy()], onnx_run(dst, ["obs"], example))


VIBED_OBS = 99
VIBED_HIST = 10
VIBED_GOAL = 730
VIBED_Z = 32
VIBED_HIDDEN = (2048, 1024, 512, 256)


def vibed_mlp(
    prev,
    out,
):
    layers = []
    for h in VIBED_HIDDEN:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out))
    return nn.Sequential(*layers)


class SuperMimicVibed(nn.Module):
    def __init__(
        self,
        z_first,
    ):
        super().__init__()
        self.z_first = z_first
        self.enc = vibed_mlp(VIBED_GOAL, VIBED_Z)
        self.dec = vibed_mlp(VIBED_Z + VIBED_OBS * VIBED_HIST, 29)
        self.register_buffer("obs_mean", torch.zeros(VIBED_OBS))
        self.register_buffer("obs_std", torch.ones(VIBED_OBS))
        self.register_buffer("goal_mean", torch.zeros(VIBED_GOAL))
        self.register_buffer("goal_std", torch.ones(VIBED_GOAL))

    def forward(
        self,
        obs_history,
        goal,
    ):
        z = torch.tanh(self.enc((goal - self.goal_mean) / self.goal_std))
        o = obs_history.reshape(-1, VIBED_HIST, VIBED_OBS)
        o = ((o - self.obs_mean) / self.obs_std).flatten(1)
        return self.dec(torch.cat([z, o] if self.z_first else [o, z], dim=-1))


def export_supermimic_vibed(
    pan_src,
    pan_dst,
    omega_zip,
    omega_member,
    omega_sha256,
    omega_dst,
):
    import zipfile

    names = ["obs_history", "goal"]
    example = (
        torch.randn(STEPS, VIBED_OBS * VIBED_HIST),
        torch.randn(STEPS, VIBED_GOAL),
    )

    def save(
        net,
        dst,
    ):
        torch.onnx.export(
            net,
            example,
            dst,
            input_names=names,
            output_names=["action"],
            dynamic_axes={n: {0: "batch"} for n in names + ["action"]},
            opset_version=OPSET,
            dynamo=False,
        )

    ck = torch.load(pan_src, map_location="cpu", weights_only=True)
    pi = ck["pi"]
    net = SuperMimicVibed(z_first=True)
    for part in ("enc", "dec"):
        for k in range(0, 9, 2):
            layer = getattr(net, part)[k]
            layer.weight.data.copy_(pi[f"{part}.{k}.weight"])
            layer.bias.data.copy_(pi[f"{part}.{k}.bias"])
    for kind in ("obs", "goal"):
        getattr(net, kind + "_mean").copy_(pi[kind + "_norm.mean"])
        getattr(net, kind + "_std").copy_(torch.sqrt(pi[kind + "_norm.var"] + 1e-2**2))
    net.eval()

    def pan_reference(
        hist,
        goal,
    ):
        def mlp(
            prefix,
            x,
        ):
            for k in range(0, 9, 2):
                x = torch.nn.functional.linear(
                    x, pi[f"{prefix}.{k}.weight"], pi[f"{prefix}.{k}.bias"]
                )
                if k < 8:
                    x = torch.nn.functional.elu(x)
            return x

        def norm(
            kind,
            x,
        ):
            return (x - pi[kind + "_norm.mean"]) / torch.sqrt(
                pi[kind + "_norm.var"] + 1e-2**2
            )

        z = torch.tanh(mlp("enc", norm("goal", goal)))
        o = norm("obs", hist.reshape(-1, VIBED_HIST, VIBED_OBS)).flatten(-2)
        return mlp("dec", torch.cat((z, o), -1))

    with torch.no_grad():
        ref = pan_reference(*example)
    save(net, pan_dst)
    agree(pan_dst, [ref.numpy()], onnx_run(pan_dst, names, example))

    raw = zipfile.ZipFile(omega_zip).read(omega_member)
    if hashlib.sha256(raw).hexdigest() != omega_sha256:
        raise RuntimeError(
            f"download_export_onnx: {omega_zip}:{omega_member} sha256 mismatch"
        )
    init = {
        i.name: torch.from_numpy(numpy_helper.to_array(i).copy())
        for i in onnx.load_from_string(raw).graph.initializer
    }
    net = SuperMimicVibed(z_first=False)
    for part, src in (("enc", "encoder"), ("dec", "decoder")):
        for k in range(0, 9, 2):
            layer = getattr(net, part)[k]
            layer.weight.data.copy_(init[f"actor.{src}.{k}.weight"])
            layer.bias.data.copy_(init[f"actor.{src}.{k}.bias"])
    net.obs_mean.copy_(init["actor.obs_norm.mean"])
    net.goal_mean.copy_(init["actor.goal_norm.mean"])
    divs = [v for k, v in init.items() if k.startswith("onnx::Div")]
    net.obs_std.copy_(next(v for v in divs if v.numel() == VIBED_OBS))
    net.goal_std.copy_(next(v for v in divs if v.numel() == VIBED_GOAL))
    net.eval()

    one = ort.InferenceSession(raw, providers=["CPUExecutionProvider"])
    hist, goal = (t.numpy() for t in example)
    ref = np.concatenate(
        [
            one.run(
                None,
                {
                    "history": hist[r : r + 1].reshape(1, VIBED_HIST, VIBED_OBS),
                    "goal_features": goal[r : r + 1],
                },
            )[0]
            for r in range(STEPS)
        ]
    )
    save(net, omega_dst)
    agree(omega_dst, [ref], onnx_run(omega_dst, names, example))


BREEZEWRF_OBS = 93
BREEZEWRF_HIST = 10
BREEZEWRF_GOAL = 2680
BREEZEWRF_HIDDEN = (2048, 1024, 512, 256)


def breezewrf_mlp(
    prev,
    out,
):
    layers = []
    for h in BREEZEWRF_HIDDEN:
        layers += [nn.Linear(prev, h), nn.ELU()]
        prev = h
    layers.append(nn.Linear(prev, out))
    return nn.Sequential(*layers)


class SuperMimicBreezewrf(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.register_buffer("default_q", torch.zeros(29))
        self.register_buffer("observation_scale", torch.ones(BREEZEWRF_OBS))
        self.encoder = breezewrf_mlp(BREEZEWRF_GOAL, 32)
        self.decoder = breezewrf_mlp(32 + BREEZEWRF_HIST * BREEZEWRF_OBS, 29)

    def forward(
        self,
        obs_history,
        goal,
    ):
        z = torch.tanh(self.encoder(goal))
        o = obs_history.reshape(-1, BREEZEWRF_HIST, BREEZEWRF_OBS)
        o = (o / self.observation_scale).flatten(1)
        return self.decoder(torch.cat([z, o], dim=-1))


def export_supermimic_breezewrf(
    src,
    dst,
):
    ck = torch.load(src, map_location="cpu", weights_only=True)
    if ck["reference_schema"] != "full_body_268" or ck["config"].get(
        "actor_input_normalization", False
    ):
        raise RuntimeError(
            f"download_export_onnx: {src} is not the full_body_268 actor"
        )
    pi = ck["actor"]
    net = SuperMimicBreezewrf()
    net.load_state_dict(pi)
    net.eval()

    def reference(
        hist,
        goal,
    ):
        def mlp(
            prefix,
            x,
        ):
            for k in range(0, 9, 2):
                x = torch.nn.functional.linear(
                    x, pi[f"{prefix}.{k}.weight"], pi[f"{prefix}.{k}.bias"]
                )
                if k < 8:
                    x = torch.nn.functional.elu(x)
            return x

        z = mlp("encoder", goal).tanh()
        o = hist.reshape(-1, BREEZEWRF_HIST, BREEZEWRF_OBS) / pi["observation_scale"]
        return mlp("decoder", torch.cat((z, o.flatten(-2)), -1))

    names = ["obs_history", "goal"]
    example = (
        torch.randn(STEPS, BREEZEWRF_OBS * BREEZEWRF_HIST),
        torch.randn(STEPS, BREEZEWRF_GOAL),
    )
    with torch.no_grad():
        ref = reference(*example)
    torch.onnx.export(
        net,
        example,
        dst,
        input_names=names,
        output_names=["action"],
        dynamic_axes={n: {0: "batch"} for n in names + ["action"]},
        opset_version=OPSET,
        dynamo=False,
    )
    agree(dst, [ref.numpy()], onnx_run(dst, names, example))


def export_recurrent(
    src,
    dst,
    net,
    obs_dim,
    hidden,
    drop,
    rename,
    carries_state=True,
):
    original = torch.jit.load(src, map_location="cpu")
    original.eval()
    state = original.state_dict()

    weights = {k: v for k, v in state.items() if k not in drop}
    for source, target in rename.items():
        weights[target] = state[source]
    net.load_state_dict(weights)
    net.eval()

    with torch.no_grad():
        if carries_state:
            original.hidden_state.zero_()
            original.cell_state.zero_()
        h = torch.zeros(1, 1, hidden)
        c = torch.zeros(1, 1, hidden)
        for _ in range(STEPS):
            x = torch.randn(1, obs_dim)
            ref = original(x)
            got, hn, cn = net(x, h, c)
            agree(f"{dst} rebuild", [ref.numpy()], [got.numpy()])
            if carries_state:
                h, c = hn, cn

    example = (
        torch.randn(1, obs_dim),
        torch.zeros(1, 1, hidden),
        torch.zeros(1, 1, hidden),
    )
    in_names = ["obs", "hidden_in", "cell_in"]
    out_names = ["action", "hidden_out", "cell_out"]
    with torch.no_grad():
        ref = list(net(*example))

    torch.onnx.export(
        net,
        example,
        dst,
        input_names=in_names,
        output_names=out_names,
        opset_version=OPSET,
        dynamo=False,
    )
    agree(dst, [r.detach().numpy() for r in ref], onnx_run(dst, in_names, example))


SUPERMIMIC689_GOAL = 689
SUPERMIMIC689_HIDDEN = (2048, 1024, 512, 256)
SUPERMIMIC689_OBS_TERMS = (29, 29, 3, 3, 29)
SUPERMIMIC689_GOAL_TERMS = (29, 29, 3, 3, 3, 1)


def supermimic689_mlp(
    prev,
    out,
):
    layers = []
    for h in SUPERMIMIC689_HIDDEN:
        layers += [nn.Linear(prev, h), nn.SiLU()]
        prev = h
    layers.append(nn.Linear(prev, out))
    mlp = nn.Module()
    mlp.net = nn.Sequential(*layers)
    return mlp


class SuperMimic689Encoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = supermimic689_mlp(SUPERMIMIC689_GOAL, 32)
        self.register_buffer("scale", torch.ones(SUPERMIMIC689_GOAL))

    def forward(
        self,
        goal,
    ):
        return torch.tanh(self.mlp.net(goal * self.scale))


class SuperMimic689Decoder(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.mlp = supermimic689_mlp(32 + 930, 29)
        self.register_buffer("scale", torch.ones(930))

    def forward(
        self,
        z,
        obs,
    ):
        return self.mlp.net(torch.cat([z, obs * self.scale], dim=-1))


class SuperMimic689(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.encoders = nn.ModuleDict({"motion": SuperMimic689Encoder()})
        self.decoder = SuperMimic689Decoder()

    def forward(
        self,
        obs,
        goal,
    ):
        return self.decoder(self.encoders["motion"](goal), obs)


def supermimic689_numpy(
    state,
    obs,
    goal,
):
    def mlp(prefix, x):
        n = 0
        while f"{prefix}.{n + 2}.weight" in state:
            x = x @ state[f"{prefix}.{n}.weight"].T + state[f"{prefix}.{n}.bias"]
            x = x / (1.0 + np.exp(-x))
            n += 2
        return x @ state[f"{prefix}.{n}.weight"].T + state[f"{prefix}.{n}.bias"]

    z = np.tanh(mlp("encoders.motion.mlp.net", goal * state["encoders.motion.scale"]))
    x = np.concatenate([z, obs * state["decoder.scale"]], axis=-1)
    return mlp("decoder.mlp.net", x)


def export_supermimic689(
    src,
    dst,
):
    with torch.serialization.safe_globals(
        [pathlib.Path, pathlib.PosixPath, pathlib.WindowsPath]
    ):
        ck = torch.load(src, map_location="cpu", weights_only=True)
    cfg = ck["config"]
    if (
        cfg["policy"]["goal_dim"] != SUPERMIMIC689_GOAL
        or cfg["policy"]["history"] != 10
        or cfg["env"]["future_frames"] != 10
        or not cfg["env"]["anchor_obs"]
        or cfg["env"].get("anchor_ori_only", False)
    ):
        raise RuntimeError(
            f"download_export_onnx: {src} is not a 10x68+9 anchor_obs actor"
        )
    net = SuperMimic689()
    net.load_state_dict(ck["policy"])
    net.eval()

    obs_scale = np.concatenate(
        [
            np.full(10 * n, 1.0 / s)
            for n, s in zip(SUPERMIMIC689_OBS_TERMS, cfg["policy"]["obs_term_scale"])
        ]
    )
    goal_scale = np.concatenate(
        [
            np.tile(
                np.concatenate(
                    [
                        np.full(n, 1.0 / s)
                        for n, s in zip(
                            SUPERMIMIC689_GOAL_TERMS, cfg["policy"]["goal_term_scale"]
                        )
                    ]
                ),
                10,
            ),
            np.ones(9),
        ]
    )
    agree(
        f"{src} scales",
        [obs_scale, goal_scale],
        [net.decoder.scale.numpy(), net.encoders["motion"].scale.numpy()],
    )

    names = ["obs_history", "goal"]
    example = (torch.randn(STEPS, 930), torch.randn(STEPS, SUPERMIMIC689_GOAL))
    with torch.no_grad():
        ref = [net(*example).numpy()]
    state = {k: v.double().numpy() for k, v in ck["policy"].items()}
    agree(
        f"{src} torch",
        [supermimic689_numpy(state, *(t.double().numpy() for t in example))],
        ref,
    )

    torch.onnx.export(
        net,
        example,
        dst,
        input_names=names,
        output_names=["action"],
        dynamic_axes={n: {0: "batch"} for n in names + ["action"]},
        opset_version=OPSET,
        dynamo=False,
    )
    agree(dst, ref, onnx_run(dst, names, example))


def export_float_io(
    src,
    dst,
    example,
):
    model = onnx.load(src)
    graph = model.graph
    wrapped = {}

    for value in graph.input:
        elem = value.type.tensor_type.elem_type
        if elem == onnx.TensorProto.FLOAT:
            continue
        inner = value.name + "__i"
        for node in graph.node:
            for i, name in enumerate(node.input):
                if name == value.name:
                    node.input[i] = inner
        graph.node.insert(
            0,
            onnx.helper.make_node(
                "Cast", [value.name], [inner], name="wrap_" + value.name, to=elem
            ),
        )
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        wrapped[value.name] = elem

    for value in graph.output:
        elem = value.type.tensor_type.elem_type
        if elem == onnx.TensorProto.FLOAT:
            continue
        inner = value.name + "__i"
        for node in graph.node:
            for i, name in enumerate(node.output):
                if name == value.name:
                    node.output[i] = inner
            for i, name in enumerate(node.input):
                if name == value.name:
                    node.input[i] = inner
        graph.node.append(
            onnx.helper.make_node(
                "Cast",
                [inner],
                [value.name],
                name="wrap_" + value.name,
                to=onnx.TensorProto.FLOAT,
            )
        )
        value.type.tensor_type.elem_type = onnx.TensorProto.FLOAT
        wrapped[value.name] = elem

    if not wrapped:
        raise RuntimeError(
            f"download_export_onnx: {src} is already float32 at the boundary"
        )

    missing = sorted({v.name for v in graph.input} - set(example))
    if missing:
        raise RuntimeError(
            f"download_export_onnx: {src} has no example input for {missing}"
        )

    onnx.save(model, dst)

    source = {
        name: example[name].astype(onnx.helper.tensor_dtype_to_np_dtype(wrapped[name]))
        if name in wrapped
        else example[name]
        for name in example
    }
    session = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    ref = [np.asarray(r, np.float32) for r in session.run(None, source)]
    session = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    agree(dst, ref, session.run(None, example))


def export_batched(
    src,
    dst,
    batch=STEPS,
):
    model = onnx.load(src)
    graph = model.graph

    rows = {}
    for value in list(graph.input) + list(graph.output):
        shape = [d.dim_value for d in value.type.tensor_type.shape.dim]
        if len(shape) != 1 or shape[0] < 1:
            raise RuntimeError(
                f"download_export_onnx: {src} boundary '{value.name}' is {shape}, not one row"
            )
        rows[value.name] = shape
        value.CopyFrom(
            onnx.helper.make_tensor_value_info(
                value.name,
                value.type.tensor_type.elem_type,
                ["batch"] + shape,
            )
        )
    del graph.value_info[:]
    onnx.save(model, dst)

    rng = np.random.default_rng(0)
    feed = {
        value.name: rng.standard_normal((batch, *rows[value.name])).astype(np.float32)
        for value in graph.input
    }
    one = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    many = ort.InferenceSession(dst, providers=["CPUExecutionProvider"])
    per_row = [one.run(None, {n: v[r] for n, v in feed.items()}) for r in range(batch)]
    ref = [
        np.stack([per_row[r][k] for r in range(batch)]) for k in range(len(per_row[0]))
    ]
    agree(dst, ref, many.run(None, feed))


def record_rows(
    src,
    feed,
):
    model = onnx.shape_inference.infer_shapes(onnx.load(src), strict_mode=False)
    graph = model.graph
    elem = {v.name: v.type.tensor_type.elem_type for v in graph.value_info}
    known = {o.name for o in graph.output}
    for node in graph.node:
        for name in node.output:
            if name and name not in known and name in elem:
                graph.output.append(
                    onnx.helper.make_tensor_value_info(name, elem[name], None)
                )
                known.add(name)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(
        model.SerializeToString(), options, providers=["CPUExecutionProvider"]
    )
    names = [o.name for o in session.get_outputs()]
    return dict(zip(names, (np.asarray(v) for v in session.run(None, feed))))


class Batcher:
    def __init__(
        self,
        src,
        feed,
    ):
        self.model = onnx.load(src)
        self.graph = self.model.graph
        self.rec = record_rows(src, feed)
        for init in self.graph.initializer:
            self.rec[init.name] = numpy_helper.to_array(init)
        for value in self.graph.input:
            self.rec[value.name] = np.asarray(feed[value.name])
        self.weights = {init.name for init in self.graph.initializer}
        self.folded = set(self.weights)
        self.varying = {value.name for value in self.graph.input}
        self.emitted = set()
        self.nodes = []
        self.inits = []
        self.count = 0
        self.ref = self.graph.input[0].name

    def fresh(
        self,
        tag,
    ):
        self.count += 1
        return f"rows_{tag}_{self.count}"

    def rank(
        self,
        name,
    ):
        return self.rec[name].ndim

    def shape(
        self,
        name,
    ):
        return list(self.rec[name].shape)

    def add(
        self,
        op,
        inputs,
        outputs=None,
        **attrs,
    ):
        if outputs is None:
            outputs = [self.fresh(op.lower())]
        self.nodes.append(onnx.helper.make_node(op, inputs, outputs, **attrs))
        return outputs[0] if len(outputs) == 1 else outputs

    def tensor(
        self,
        value,
        tag="c",
    ):
        name = self.fresh(tag)
        self.inits.append(numpy_helper.from_array(np.asarray(value), name))
        return name

    def i64(
        self,
        values,
        tag="i",
    ):
        return self.tensor(np.asarray(values, dtype=np.int64), tag)

    def unsqueeze(
        self,
        x,
        axes,
    ):
        if not axes:
            return x
        return self.add("Unsqueeze", [x, self.i64(list(axes), "axes")])

    def pad_rank(
        self,
        x,
        now,
        want,
    ):
        return self.unsqueeze(x, list(range(1, 1 + want - now)))

    def batch_dim(self):
        return self.add(
            "Slice",
            [
                self.add("Shape", [self.ref]),
                self.i64([0], "start"),
                self.i64([1], "end"),
            ],
        )

    def const(
        self,
        name,
    ):
        if name in self.weights or name in self.emitted:
            return name
        self.inits.append(numpy_helper.from_array(self.rec[name], name))
        self.emitted.add(name)
        return name

    def arg(
        self,
        name,
    ):
        return name if name in self.varying else self.const(name)

    def rows(
        self,
        name,
    ):
        if name in self.varying:
            return name
        value = self.rec[name]
        target = self.batch_dim()
        if value.ndim:
            target = self.add(
                "Concat", [target, self.i64(list(value.shape), "shape")], axis=0
            )
        return self.add("Expand", [self.unsqueeze(self.const(name), [0]), target])

    @staticmethod
    def axis(
        axis,
        rank,
    ):
        return axis + rank if axis < 0 else axis

    @staticmethod
    def attr(
        node,
        key,
        default=None,
    ):
        for a in node.attribute:
            if a.name == key:
                return onnx.helper.get_attribute_value(a)
        return default

    @staticmethod
    def attrs(
        node,
        *skip,
    ):
        return {
            a.name: onnx.helper.get_attribute_value(a)
            for a in node.attribute
            if a.name not in skip
        }

    def run(
        self,
        dst,
    ):
        graph = self.graph
        for value in list(graph.input) + list(graph.output):
            if self.shape(value.name)[:1] != [1]:
                raise RuntimeError(f"download_export_onnx: {value.name} is not one row")
        for node in graph.node:
            inputs = [i for i in node.input if i]
            folded = node.op_type in {"Shape", "Size", "Constant"} or all(
                i in self.folded for i in inputs
            )
            if folded:
                self.folded.update(o for o in node.output if o)
                continue
            self.varying.update(o for o in node.output if o)
            self.rewrite(node)
        inputs, prologue, outputs, epilogue = [], [], [], []
        for value in graph.input:
            inner = value.name + "__row"
            for node in self.nodes:
                for k, name in enumerate(node.input):
                    if name == value.name:
                        node.input[k] = inner
            prologue.append(
                onnx.helper.make_node(
                    "Unsqueeze", [value.name, self.i64([1], "axes")], [inner]
                )
            )
            inputs.append(
                onnx.helper.make_tensor_value_info(
                    value.name,
                    value.type.tensor_type.elem_type,
                    ["batch"] + self.shape(value.name)[1:],
                )
            )
        for value in graph.output:
            inner = value.name + "__row"
            for node in self.nodes:
                for k, name in enumerate(node.output):
                    if name == value.name:
                        node.output[k] = inner
            epilogue.append(
                onnx.helper.make_node(
                    "Squeeze", [inner, self.i64([1], "axes")], [value.name]
                )
            )
            outputs.append(
                onnx.helper.make_tensor_value_info(
                    value.name,
                    value.type.tensor_type.elem_type,
                    ["batch"] + self.shape(value.name)[1:],
                )
            )
        used = set()
        for node in self.nodes:
            used.update(node.input)
        model = onnx.helper.make_model(
            onnx.helper.make_graph(
                prologue + self.nodes + epilogue,
                graph.name,
                inputs,
                outputs,
                initializer=[i for i in graph.initializer if i.name in used]
                + self.inits,
            ),
            opset_imports=self.model.opset_import,
        )
        model.ir_version = self.model.ir_version
        onnx.checker.check_model(model)
        onnx.save(model, dst)

    def rewrite(
        self,
        node,
    ):
        op = node.op_type
        if op in BATCH_ELEMENTWISE:
            return self.elementwise(node)
        if op in BATCH_UNARY:
            return self.add(
                op,
                [self.arg(i) if i else "" for i in node.input],
                list(node.output),
                **self.attrs(node),
            )
        if op in BATCH_REDUCE:
            return self.reduce(node)
        if op in BATCH_AXIS_DEFAULT:
            return self.axis_op(node, BATCH_AXIS_DEFAULT[op])
        handler = getattr(self, "op_" + op, None)
        if handler is None:
            raise RuntimeError(
                f"download_export_onnx: no batched form of {op} ({node.name})"
            )
        return handler(node)

    def elementwise(
        self,
        node,
    ):
        out = node.output[0]
        want = self.rank(out)
        args = [
            self.pad_rank(i, self.rank(i), want) if i in self.varying else self.const(i)
            for i in node.input
        ]
        self.add(node.op_type, args, [out], **self.attrs(node))

    def reduce(
        self,
        node,
    ):
        x = node.input[0]
        rank = self.rank(x)
        axes = self.attr(node, "axes")
        if axes is None:
            axes = list(range(rank))
        self.add(
            node.op_type,
            [self.arg(x)],
            list(node.output),
            axes=[self.axis(a, rank) + 1 for a in axes],
            **self.attrs(node, "axes"),
        )

    def axis_op(
        self,
        node,
        default,
    ):
        anchor = node.output[0] if node.op_type == "OneHot" else node.input[0]
        axis = self.axis(self.attr(node, "axis", default), self.rank(anchor)) + 1
        self.add(
            node.op_type,
            [self.arg(i) if i else "" for i in node.input],
            list(node.output),
            axis=axis,
            **self.attrs(node, "axis"),
        )

    def op_ReduceSum(
        self,
        node,
    ):
        x = node.input[0]
        rank = self.rank(x)
        if len(node.input) > 1 and node.input[1]:
            axes = self.rec[node.input[1]].tolist()
        else:
            axes = list(range(rank))
        self.add(
            "ReduceSum",
            [
                self.arg(x),
                self.i64([self.axis(int(a), rank) + 1 for a in axes], "axes"),
            ],
            list(node.output),
            **self.attrs(node),
        )

    def op_MatMul(
        self,
        node,
    ):
        a, b = node.input
        out = node.output[0]
        ra, rb = self.rank(a), self.rank(b)
        big = max(self.rank(out), 2)
        squeeze = []
        if a in self.varying:
            if ra == 1:
                a = self.pad_rank(self.unsqueeze(a, [1]), 2, big)
                squeeze.append(big - 1)
            else:
                a = self.pad_rank(a, ra, big)
        else:
            value = self.rec[a]
            if ra == 1:
                value = value.reshape(1, -1)
                squeeze.append(big - 1)
            a = self.tensor(np.ascontiguousarray(value), "w")
        if b in self.varying:
            if rb == 1:
                b = self.pad_rank(self.unsqueeze(b, [2]), 2, big)
                squeeze.append(big)
            else:
                b = self.pad_rank(b, rb, big)
        else:
            value = self.rec[b]
            if rb == 1:
                value = value.reshape(-1, 1)
                squeeze.append(big)
            b = self.tensor(np.ascontiguousarray(value), "w")
        y = self.add("MatMul", [a, b])
        if squeeze:
            self.add("Squeeze", [y, self.i64(squeeze, "axes")], [out])
        else:
            self.add("Identity", [y], [out])

    def op_Gemm(
        self,
        node,
    ):
        a, b = node.input[0], node.input[1]
        c = node.input[2] if len(node.input) > 2 else ""
        if b in self.varying or c in self.varying or self.attr(node, "transA", 0):
            raise RuntimeError(
                f"download_export_onnx: Gemm {node.name} is not a row times weights"
            )
        w = self.rec[b]
        if self.attr(node, "transB", 0):
            w = w.T
        w = w * self.attr(node, "alpha", 1.0)
        y = self.add("MatMul", [a, self.tensor(np.ascontiguousarray(w), "w")])
        if c:
            bias = self.rec[c] * self.attr(node, "beta", 1.0)
            self.add(
                "Add",
                [y, self.tensor(np.ascontiguousarray(bias), "b")],
                [node.output[0]],
            )
        else:
            self.add("Identity", [y], [node.output[0]])

    def op_Einsum(
        self,
        node,
    ):
        equation = self.attr(node, "equation").decode()
        if BATCH_EINSUM.fullmatch(equation) is None:
            raise RuntimeError(
                f"download_export_onnx: Einsum {equation} has no batched form"
            )
        a, b = node.input
        want = self.rank(node.output[0])
        ra = want + 1 if a in self.varying else self.rank(a)
        a = self.pad_rank(a, self.rank(a), want) if a in self.varying else self.const(a)
        b = self.pad_rank(b, self.rank(b), want) if b in self.varying else self.const(b)
        at = self.add("Transpose", [a], perm=list(range(ra - 2)) + [ra - 1, ra - 2])
        self.add("MatMul", [at, b], [node.output[0]])

    def op_Reshape(
        self,
        node,
    ):
        data, shape = node.input
        if shape in self.varying:
            raise RuntimeError(
                f"download_export_onnx: Reshape {node.name} takes a shape per row"
            )
        out = node.output[0]
        self.add(
            "Reshape",
            [self.arg(data), self.i64([-1] + self.shape(out), "shape")],
            [out],
        )

    def op_Flatten(
        self,
        node,
    ):
        out = node.output[0]
        self.add(
            "Reshape",
            [self.arg(node.input[0]), self.i64([-1] + self.shape(out), "shape")],
            [out],
        )

    def op_Transpose(
        self,
        node,
    ):
        x = node.input[0]
        perm = self.attr(node, "perm") or list(range(self.rank(x)))[::-1]
        self.add(
            "Transpose",
            [self.arg(x)],
            list(node.output),
            perm=[0] + [p + 1 for p in perm],
        )

    def op_Unsqueeze(
        self,
        node,
    ):
        x, axes = node.input
        rank = self.rank(node.output[0])
        axes = [self.axis(int(a), rank) + 1 for a in self.rec[axes].tolist()]
        self.add("Unsqueeze", [self.arg(x), self.i64(axes, "axes")], list(node.output))

    def op_Squeeze(
        self,
        node,
    ):
        x = node.input[0]
        rank = self.rank(x)
        if len(node.input) > 1 and node.input[1]:
            axes = [self.axis(int(a), rank) for a in self.rec[node.input[1]].tolist()]
        else:
            axes = [k for k, d in enumerate(self.shape(x)) if d == 1]
        self.add(
            "Squeeze",
            [self.arg(x), self.i64([a + 1 for a in axes], "axes")],
            list(node.output),
        )

    def op_Concat(
        self,
        node,
    ):
        out = node.output[0]
        axis = self.axis(self.attr(node, "axis"), self.rank(out)) + 1
        self.add("Concat", [self.rows(i) for i in node.input], [out], axis=axis)

    def op_Slice(
        self,
        node,
    ):
        data, *rest = node.input
        if any(r in self.varying for r in rest):
            raise RuntimeError(
                f"download_export_onnx: Slice {node.name} takes bounds per row"
            )
        rank = self.rank(data)
        if len(rest) > 2 and rest[2]:
            axes = [self.axis(int(a), rank) + 1 for a in self.rec[rest[2]].tolist()]
        else:
            axes = list(range(1, 1 + self.rec[rest[0]].size))
        args = [
            self.arg(data),
            self.const(rest[0]),
            self.const(rest[1]),
            self.i64(axes, "axes"),
        ]
        if len(rest) > 3 and rest[3]:
            args.append(self.const(rest[3]))
        self.add("Slice", args, list(node.output))

    def op_Gather(
        self,
        node,
    ):
        data, idx = node.input
        out = node.output[0]
        rd, ri = self.rank(data), self.rank(idx)
        axis = self.axis(self.attr(node, "axis", 0), rd)
        if idx not in self.varying:
            self.add("Gather", [data, self.const(idx)], [out], axis=axis + 1)
            return
        if data not in self.varying:
            value = self.rec[data]
            if axis:
                value = np.transpose(
                    value, [axis] + [k for k in range(rd) if k != axis]
                )
            y = self.add(
                "Gather", [self.tensor(np.ascontiguousarray(value), "g"), idx], axis=0
            )
        else:
            if axis:
                data = self.add(
                    "Transpose",
                    [data],
                    perm=[0, axis + 1] + [k + 1 for k in range(rd) if k != axis],
                )
            y = self.add(
                "GatherND", [data, self.unsqueeze(idx, [ri + 1])], batch_dims=1
            )
        if not axis:
            self.add("Identity", [y], [out])
            return
        perm = (
            [0]
            + [1 + ri + j for j in range(axis)]
            + list(range(1, 1 + ri))
            + [1 + ri + j for j in range(axis, rd - 1)]
        )
        self.add("Transpose", [y], [out], perm=perm)

    def op_GatherElements(
        self,
        node,
    ):
        data, idx = node.input
        axis = self.axis(self.attr(node, "axis", 0), self.rank(data)) + 1
        self.add(
            "GatherElements",
            [self.rows(data), self.rows(idx)],
            list(node.output),
            axis=axis,
        )

    def op_ScatterElements(
        self,
        node,
    ):
        axis = self.axis(self.attr(node, "axis", 0), self.rank(node.input[0])) + 1
        self.add(
            "ScatterElements",
            [self.rows(i) for i in node.input],
            list(node.output),
            axis=axis,
            **self.attrs(node, "axis"),
        )

    def op_ScatterND(
        self,
        node,
    ):
        data, idx, upd = node.input
        lead = self.shape(idx)[:-1]
        batch = self.batch_dim()
        span = self.add(
            "Range",
            [
                self.i64(0, "zero"),
                self.add("Squeeze", [batch, self.i64([0], "axes")]),
                self.i64(1, "one"),
            ],
        )
        span = self.add(
            "Reshape", [span, self.i64([-1] + [1] * (len(lead) + 1), "shape")]
        )
        row = self.add(
            "Expand",
            [span, self.add("Concat", [batch, self.i64(lead + [1], "shape")], axis=0)],
        )
        full = self.add("Concat", [row, self.rows(idx)], axis=len(lead) + 1)
        self.add(
            "ScatterND",
            [self.rows(data), full, self.rows(upd)],
            list(node.output),
            **self.attrs(node),
        )

    def op_Expand(
        self,
        node,
    ):
        x, shape = node.input
        if shape in self.varying:
            raise RuntimeError(
                f"download_export_onnx: Expand {node.name} takes a shape per row"
            )
        out = node.output[0]
        want = self.rank(out)
        target = [int(s) for s in self.rec[shape].tolist()]
        target = [1] * (1 + want - len(target)) + target
        x = self.pad_rank(x, self.rank(x), want) if x in self.varying else self.rows(x)
        self.add("Expand", [x, self.i64(target, "shape")], [out])

    def op_Tile(
        self,
        node,
    ):
        x, reps = node.input
        if reps in self.varying:
            raise RuntimeError(
                f"download_export_onnx: Tile {node.name} takes repeats per row"
            )
        reps = [1] + [int(v) for v in self.rec[reps].tolist()]
        self.add("Tile", [self.arg(x), self.i64(reps, "reps")], list(node.output))

    def op_CumSum(
        self,
        node,
    ):
        x, axis = node.input
        axis = self.axis(int(self.rec[axis]), self.rank(x)) + 1
        self.add(
            "CumSum",
            [self.arg(x), self.i64(axis, "axis")],
            list(node.output),
            **self.attrs(node),
        )

    def op_Conv(
        self,
        node,
    ):
        x = node.input[0]
        out = node.output[0]
        flat = self.add(
            "Reshape", [self.arg(x), self.i64([-1] + self.shape(x)[1:], "shape")]
        )
        y = self.add(
            "Conv", [flat] + [self.const(i) for i in node.input[1:]], **self.attrs(node)
        )
        self.add("Reshape", [y, self.i64([-1] + self.shape(out), "shape")], [out])

    def op_Pad(
        self,
        node,
    ):
        x, pads, *rest = node.input
        if pads in self.varying:
            raise RuntimeError(
                f"download_export_onnx: Pad {node.name} takes pads per row"
            )
        pads = [int(v) for v in self.rec[pads].tolist()]
        half = len(pads) // 2
        args = [self.arg(x), self.i64([0] + pads[:half] + [0] + pads[half:], "pads")]
        if rest and rest[0]:
            args.append(self.const(rest[0]))
        self.add("Pad", args, list(node.output), **self.attrs(node))

    def op_Resize(
        self,
        node,
    ):
        x, _, scales, *rest = list(node.input) + ["", ""]
        if not scales or scales in self.varying or rest[0]:
            raise RuntimeError(
                f"download_export_onnx: Resize {node.name} without constant scales"
            )
        scales = [1.0] + [float(v) for v in self.rec[scales].tolist()]
        self.add(
            "Resize",
            [self.arg(x), "", self.tensor(np.asarray(scales, np.float32), "scales")],
            list(node.output),
            **self.attrs(node),
        )


def export_rows(
    src,
    dst,
    feed,
    rows=ROWS_CHECK,
):
    Batcher(src, feed(1, 0)).run(dst)
    options = ort.SessionOptions()
    options.log_severity_level = 3
    one = ort.InferenceSession(src, options, providers=["CPUExecutionProvider"])
    many = ort.InferenceSession(dst, options, providers=["CPUExecutionProvider"])
    mixed = feed(rows, 1)
    got = many.run(None, mixed)
    for r in range(rows):
        alone = {k: np.repeat(v[r : r + 1], rows, axis=0) for k, v in mixed.items()}
        same = many.run(None, alone)
        for k, (a, b) in enumerate(zip(got, same)):
            if not all(np.array_equal(b[0], b[j]) for j in range(rows)):
                raise RuntimeError(
                    f"download_export_onnx: {dst} output {k} differs between equal rows"
                )
            if not np.array_equal(a[r], b[r]):
                raise RuntimeError(
                    f"download_export_onnx: {dst} row {r} depends on its neighbours"
                )
        ref = one.run(None, {k: v[r : r + 1] for k, v in mixed.items()})
        worst = max(float(np.abs(x[0] - y[r]).max()) for x, y in zip(ref, got))
        if worst > ROWS_TOL:
            raise RuntimeError(
                f"download_export_onnx: {dst} row {r} is off by {worst:.3e}"
            )


def sonic_planner_feed(
    rows,
    seed,
):
    rng = np.random.default_rng(seed)
    context = np.zeros((rows, SONIC_CTX, SONIC_QPOS), np.float32)
    context[:, :, :2] = rng.standard_normal((rows, SONIC_CTX, 2)) * 0.3
    context[:, :, 2] = SONIC_HEIGHT
    context[:, :, 3] = 1.0
    context[:, :, 7:] = rng.standard_normal((rows, SONIC_CTX, 29)) * 0.1
    allowed = np.zeros((rows, 11), np.float32)
    allowed[:, :6] = 1.0
    return {
        "context_mujoco_qpos": context,
        "mode": rng.integers(0, 2, rows).astype(np.float32),
        "target_vel": rng.uniform(0.0, 1.0, rows).astype(np.float32),
        "movement_direction": rng.standard_normal((rows, 3)).astype(np.float32),
        "facing_direction": rng.standard_normal((rows, 3)).astype(np.float32),
        "random_seed": rng.integers(0, 5000, rows).astype(np.float32),
        "height": np.full(rows, -1.0, np.float32),
        "has_specific_target": np.zeros((rows, 1), np.float32),
        "specific_target_positions": np.zeros((rows, SONIC_CTX, 3), np.float32),
        "specific_target_headings": np.zeros((rows, SONIC_CTX), np.float32),
        "allowed_pred_num_tokens": allowed,
    }


def sonic_encoder_feed(
    rows,
    seed,
):
    rng = np.random.default_rng(seed)
    return {"obs_dict": rng.standard_normal((rows, SONIC_ENC_OBS)).astype(np.float32)}


def export_latents(
    src,
    dst,
    names,
):
    with open(src, "rb") as handle:
        table = pickle.load(handle)

    want = math.sqrt(Z_DIM)
    ref, rows = [], []
    for name in names:
        if name not in table:
            raise RuntimeError(
                f"download_export_onnx: {src} has no latent named '{name}'"
            )
        z = table[name]
        while isinstance(z, (list, tuple)):
            z = z[0]
        z = torch.as_tensor(z).detach().reshape(-1).float()
        if z.numel() != Z_DIM:
            raise RuntimeError(
                f"download_export_onnx: latent '{name}' is {z.numel()} long, expected {Z_DIM}"
            )
        norm = float(z.norm())
        if abs(norm - want) > Z_NORM_TOL:
            raise RuntimeError(
                f"download_export_onnx: latent '{name}' has norm {norm:.6f}, expected {want:.6f}"
            )
        ref.append(z.numpy())
        rows.append(name + "," + ",".join(f"{v:.9g}" for v in z.tolist()))

    header = "name," + ",".join(f"z{i}" for i in range(Z_DIM))
    with open(dst, "w") as handle:
        handle.write(header + "\n" + "\n".join(rows) + "\n")

    got, back = [], []
    with open(dst) as handle:
        for line in handle:
            cells = line.strip().split(",")
            if not cells or cells[0] == "name":
                continue
            back.append(cells[0])
            got.append(np.asarray(cells[1:], dtype=np.float32))
    if back != list(names):
        raise RuntimeError(f"download_export_onnx: {dst} did not round-trip its names")
    agree(dst, ref, got)


HF = "https://huggingface.co"
GH = "https://raw.githubusercontent.com"
DRIVE = "https://drive.usercontent.google.com/download?export=download&confirm=t&id="

FILES = [
    (
        "policies/homie/model.onnx",
        "afe6363c83c8b9ab8b1a2fed598fce7de35ff045eee1a95b336fd6720e3d3d49",
        f"{GH}/InternRobotics/OpenHomie/cefcd85fcf81f529e8be065795fb2a7273e69435/HomieDeploy/deploy.onnx",
    ),
    (
        "policies/huru/model.onnx",
        "9955ef11795b34e640518e6b81f427e7415a90069be39c8d27363433dd463173",
        f"{HF}/fishy233/huru-models/resolve/bc4a0aeadd51a8c99e3aa701e6535004b67cdcd2/g1_walk_mjlab.onnx",
    ),
    (
        "policies/josabb/model.onnx",
        "08106fd7515900f4822939acbece2fef3393608428c1aeb010c2e0b4ebc6b8ea",
        f"{HF}/josabb/G1-humanoid-6dof-hands-locomotion-rl/resolve/83e06e1277b30f9a694678b711c8c753c03107a6/onnx/policy.onnx",
    ),
    (
        "policies/zealot/model_v26.onnx",
        "1e21412a09f3af7fa2dbdec58de4d4600e2679862a1b24c502c0a02916bd440f",
        f"{HF}/haixuantao/zealot-g1-locomotion/resolve/774250dd362e3341d8fe3b69b44d2ff34b0d37ac/g1_v26_iter42290.onnx",
    ),
    (
        "policies/run_residual/model_cmg.onnx",
        "e7d1ee2e67e8d2c4e0fb179ddc328e96bb10bfbd8d636b4bdd635182b34fb97e",
        f"{GH}/PMY9527/RUN_DEPLOY/d6dba9c560c1a201dde244d1e2e600a399d8a607/robots/g1_29dof/config/policy/velocity/cmg/exported/cmg_exported_new.onnx",
    ),
    (
        "policies/run_residual/model_residual.onnx",
        "5124422b4deb64cbf20cd9b7fb9f3bde9f1f0269fff9d4d4a93512cdcfdf9089",
        f"{GH}/PMY9527/RUN_DEPLOY/d6dba9c560c1a201dde244d1e2e600a399d8a607/robots/g1_29dof/config/policy/velocity/residual/exported/point25.onnx",
    ),
    (
        "policies/bfm_zero/model.onnx",
        "209097902c45621eebab2edb81070c31895fcdd558f0cf6f7f5a360fd747ab74",
        f"{HF}/LeCAR-Lab/BFM-Zero/resolve/62b4206d68e026de5e5dc7efb1529bccfb95164c/model/exported/FBcprAuxModel.onnx",
    ),
    (
        "policies/dm_agile/model.onnx",
        "94a95b934e34b4c2682127cd66eacbadaa634db8a27d67b47ad15ff3a814dad9",
        f"{HF}/datamentorshf/dm-g1-agile-locomotion-rl/resolve/4a32a3d8d1979880e2fa6a37760e6cf0017f400f/exported/policy.onnx",
    ),
    (
        "policies/dm_march/model.onnx",
        "c82f23413644ed317b11ce92bf1390e8f35370b8a9c56d6fa6255ef3086567bf",
        f"{HF}/datamentorshf/dm-g1-military-march-rl/resolve/86f48548ec595aa2b277390432619fd7c2c2c841/exported/policy.onnx",
    ),
    (
        "policies/handoff/model.onnx",
        "0c0f7975a50234a2d20935bfec6fbc7aa88b6401d8654bb4feb3aff7d2329333",
        f"{GH}/lzyang2000/HANDOFF/6454ae8811f31ed722e561cb0ca7c1e432ac7ca8/deploy/ckpt/policy.onnx",
    ),
    (
        "policies/legged_rl_lab/model.onnx",
        "26679e18977a3c5d26b4e653aaa46ef4f30fe862e7df719a2d5b77f1ebfab746",
        f"{GH}/zihanwang0422/legged_rl_lab/386ee24bc62c8033641d65b940dc2cf1dec56545/deploy/g1_deploy/exported_policy/g1_flat_1.onnx",
    ),
    (
        "policies/sonic/model_decoder.onnx",
        "c7241a123eaa36b5d64bad19540efde93cac1ad443bd4572fd12ca99898118ed",
        f"{HF}/nvidia/GEAR-SONIC/resolve/6733128a3d8a523b1418b06bca3cdf61c8b0987f/model_decoder.onnx",
    ),
    (
        "policies/wcompton/model.onnx",
        "134ae9450c41e4b282766825ea0295bda2cd7eda83f59781053194b87bad15e1",
        f"{HF}/wcompton/legged_locomotion_rl/resolve/844b125833faa820214546f16bc2e770db1471de/g1_blind_flat_end2end/2026-07-03_00-27-14_flat_loco/policy.onnx",
    ),
]


def sha256_of(
    path,
):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(
    dest,
    want,
    url,
):
    if os.path.exists(dest) and sha256_of(dest) == want:
        print(f"have  {dest}")
        return
    print(f"fetch {dest}")
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    request = urllib.request.Request(
        url, headers={"User-Agent": "teleop-walking-benchmark"}
    )
    if url.startswith(HF) and os.environ.get("HF_TOKEN"):
        # Unredirected: HF resolve URLs redirect to a CDN, which must not get
        # the token.
        request.add_unredirected_header(
            "Authorization", f"Bearer {os.environ['HF_TOKEN']}"
        )
    part = dest + ".part"
    try:
        download(request, part, dest, url)
    except BaseException:
        if os.path.exists(part):
            os.remove(part)
        raise
    got = sha256_of(part)
    if got != want:
        os.remove(part)
        raise SystemExit(
            f"sha256 mismatch for {dest}\n  expected {want}\n  got      {got}"
        )
    os.replace(part, dest)
    print(f"ok    {dest} {got}")


def download(
    request,
    part,
    dest,
    url,
):
    for attempt in range(6):
        try:
            with (
                urllib.request.urlopen(request, timeout=60) as r,
                open(part, "wb") as f,
            ):
                shutil.copyfileobj(r, f, 1 << 20)
            break
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and url.startswith(HF):
                raise SystemExit(
                    f"{dest}: HTTP {e.code} -- this repository is gated.\n"
                    "  Set HF_TOKEN to a token whose account has accepted the gate:\n"
                    '  curl -X POST -H "Authorization: Bearer $HF_TOKEN" \\\n'
                    "       -H 'Content-Type: application/json' -d '{}' \\\n"
                    f"       {url.split('/resolve/')[0]}/ask-access"
                )
            if (e.code < 500 and e.code != 429) or attempt == 5:
                raise SystemExit(f"{dest}: HTTP {e.code}")
        except (urllib.error.URLError, OSError) as e:
            if attempt == 5:
                raise SystemExit(f"{dest}: {e}")
        time.sleep(5)


def export(
    dsts,
    inputs,
    build,
):
    if not FORCE and all(os.path.exists(d) for d in dsts):
        print(f"have  {', '.join(dsts)}")
        return
    require_cuda()
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory(prefix="download_export_onnx.") as tmp:
        paths = {}
        for name, want, url in inputs:
            paths[name] = os.path.join(tmp, name)
            fetch(paths[name], want, url)
        print(f"export {', '.join(dsts)}")
        try:
            build(paths, tmp)
        except BaseException:
            # A half-written or unverified output would pass as done next run.
            for d in dsts:
                if os.path.exists(d):
                    os.remove(d)
            raise


def main():
    global FORCE
    FORCE = "--force" in sys.argv[1:]

    for dest, want, url in FILES:
        fetch(dest, want, url)

    amo = f"{GH}/OpenTeleVision/AMO/34caaf943660e6f9420e35f64e86dd56fb51dd0e"
    export(
        ["policies/amo/model_adapter.onnx"],
        [
            (
                "adapter_jit.pt",
                "159c5f691e55f68c68e2d98e287f4f87baa54e79759c04e06af3bdbc8f8edc98",
                f"{amo}/adapter_jit.pt",
            )
        ],
        lambda p, _: export_stateless(
            p["adapter_jit.pt"],
            "policies/amo/model_adapter.onnx",
            [torch.randn(1, 12)],
            ["input"],
            ["output"],
        ),
    )
    export(
        ["policies/amo/model.onnx"],
        [
            (
                "amo_jit.pt",
                "6d867ed2dd2261d0f02a5e81d2b7f92802be30f3d36570fb7c4b18707649ef3f",
                f"{amo}/amo_jit.pt",
            )
        ],
        lambda p, _: export_stateless(
            p["amo_jit.pt"],
            "policies/amo/model.onnx",
            [torch.randn(1, 1043), torch.randn(1, 2325)],
            ["obs_teacher", "extra_hist"],
            ["output"],
        ),
    )

    export(
        ["policies/rl_gym/model.onnx"],
        [
            (
                "motion.pt",
                "cf668f75b90d1abf73d2b87612a6e76bccc61ff7e083b63582d3f6aaa3c1759d",
                f"{GH}/unitreerobotics/unitree_rl_gym/276801e46c5d433564f24658bac64f254b7d2d4b/deploy/pre_train/g1/motion.pt",
            )
        ],
        lambda p, _: export_recurrent(
            p["motion.pt"],
            "policies/rl_gym/model.onnx",
            RlGym(),
            47,
            64,
            {"hidden_state", "cell_state"},
            {},
        ),
    )

    def robomimic(
        p,
        _,
    ):
        src = p["policy_29dof.pt"]
        normalizer = torch.jit.load(src, map_location="cpu").normalizer
        export_recurrent(
            src,
            "policies/robomimic/model.onnx",
            RoboMimic(float(normalizer.eps)),
            96,
            256,
            {
                "hidden_state",
                "cell_state",
                "normalizer._mean",
                "normalizer._var",
                "normalizer._std",
                "normalizer.count",
            },
            {"normalizer._mean": "mean", "normalizer._std": "std"},
        )

    export(
        ["policies/robomimic/model.onnx"],
        [
            (
                "policy_29dof.pt",
                "d1d91b0201beeb649a4624ba40052d10fe4aebe98bf6f4847decf75dd1fee2da",
                f"{GH}/ccrpRepo/RoboMimic_Deploy/3a72ec1a55dcf155be88c9dcc4a32ecb5c11e313/policy/loco_mode/model/policy_29dof.pt",
            )
        ],
        robomimic,
    )

    export(
        ["policies/g1_gym/model.onnx"],
        [
            (
                "4_26_policy_lstm_16000.pt",
                "aaef00b10c001612638551670e3ecacbad11c3116c2cd2dfdee1b37add02e600",
                f"{GH}/IlikeSukiyaki/G1_GYM/1e5aa629187d7da31ce1f832071bd0a478b0f9aa/legged_gym/sim2mujoco/pre_train/g1_full/4_26_policy_lstm_16000.pt",
            )
        ],
        lambda p, _: export_recurrent(
            p["4_26_policy_lstm_16000.pt"],
            "policies/g1_gym/model.onnx",
            G1Gym(),
            90,
            64,
            {"hidden_state", "cell_state"},
            {},
        ),
    )

    export(
        ["policies/stepdown/model.onnx"],
        [
            (
                "safety_motion.pt",
                "3c3ddea79f3010640493bd65c39c7bdf8784131d923f1f1401c7957ba1adade6",
                f"{HF}/arushbisht12/unitree-g1-stepdown-safety/resolve/fd97d9ac2eccda849e6f7440a4c400a6cd3697ae/safety_motion.pt",
            )
        ],
        lambda p, _: export_recurrent(
            p["safety_motion.pt"],
            "policies/stepdown/model.onnx",
            RlGym(),
            47,
            64,
            set(),
            {},
            carries_state=False,
        ),
    )

    sunny = f"{HF}/jasonsfmeitian/g1-sunny-locomotion/resolve/3a8b206d42a9e3ce015513cb2fffe2498cb2e92c/baseline"
    export(
        ["policies/sunny/model.onnx"],
        [
            (
                "g1_deploy_walk_policy.onnx",
                "6cc6c71dc1d900bc240ade207cc7987b786d90503b9c899edc12ab44fcc56d62",
                f"{sunny}/g1_deploy_walk_policy.onnx",
            ),
            (
                "g1_deploy_scan_mean.npy",
                "0ab35d51d848dea3c7d36557a611c85ed03e0ed87bedc9618560d68d4ccdd686",
                f"{sunny}/g1_deploy_scan_mean.npy",
            ),
        ],
        lambda p, _: export_sunny(
            p["g1_deploy_walk_policy.onnx"],
            p["g1_deploy_scan_mean.npy"],
            "policies/sunny/model.onnx",
        ),
    )

    export(
        ["policies/mturan33/model.onnx"],
        [
            (
                "unified_critic_s6u.pt",
                "449728a9e5e48210012d74db2c6bd689b53e3c298646b7b2e3b1f2b27490b2c5",
                f"{HF}/mturan33/g1-dual-critic-locomanip/resolve/07a034a67155652f2fd6d906bf95d19314a83716/unified_critic_s6u.pt",
            )
        ],
        lambda p, _: export_submodule(
            p["unified_critic_s6u.pt"],
            "policies/mturan33/model.onnx",
            Mturan33Loco(),
            "loco_actor.",
            57,
        ),
    )

    export(
        ["policies/bfm_zero/latents.csv"],
        [
            (
                "reward_locomotion.pkl",
                "1b974e7278ab6183d809d11fbd77cca817f3714a4808737cd69f96606ab77718",
                f"{HF}/LeCAR-Lab/BFM-Zero/resolve/62b4206d68e026de5e5dc7efb1529bccfb95164c/model/reward_inference/reward_locomotion.pkl",
            )
        ],
        lambda p, _: export_latents(
            p["reward_locomotion.pkl"],
            "policies/bfm_zero/latents.csv",
            BFM_ZERO_LATENTS,
        ),
    )

    export(
        ["policies/mimic_lite/model.onnx"],
        [
            (
                "model_roa.onnx",
                "78aec8b2738f3940fb47a58021b9584faa3f69b53476551c52a2026909db33c1",
                f"{DRIVE}1xtHn7tu2s-A8RQ84AuT1EqiXQBYRe9pt",
            )
        ],
        lambda p, _: export_batched(
            p["model_roa.onnx"],
            "policies/mimic_lite/model.onnx",
        ),
    )

    morphmimic = (
        f"{GH}/lzyang2000/morphmimic/0128ef6afc85391094075538c933259ac960aaa0/demo/live"
    )
    export(
        ["policies/morphmimic/model.onnx"],
        [
            (
                "policy.onnx",
                "081af6210d0ba3a6b3f5773532957d27d25d46c1e0a86a5b1c5e74fb15d1b243",
                f"{morphmimic}/policy.onnx",
            ),
            (
                "robots.json",
                "3588df11f9fe891f18914806e578283e583d2fcf96994b4578cd10fe4352952d",
                f"{morphmimic}/robots.json",
            ),
        ],
        lambda p, _: export_morphmimic(
            p["policy.onnx"],
            p["robots.json"],
            "g1",
            "policies/morphmimic/model.onnx",
        ),
    )

    for dst, name, want, url, fn in (
        (
            "policies/supermimic/model_rayzhao.onnx",
            "policy.pt",
            "a12bac4f93cf653dbf755ead058b6a1837a2db6363aea4ff51e9c4bc817bfbf7",
            f"{HF}/RayZhao/supermimic_getup/resolve/4e7a558a4a08bcf8f700bff0fa408c0ebe82f5d6/policy.pt",
            export_supermimic680,
        ),
        (
            "policies/supermimic/model_kangzejian.onnx",
            "supermimic_g1_168000_policy.pt",
            "7d8e9536ae58251911f4c2b5d464c8f573cb4eb9e9b51d1cafd79786b4993ef3",
            "https://github.com/kangzejian1896/supermimic-g1-reproduction/releases/download/v0.2.0-168k/supermimic_g1_168000_policy.pt",
            export_supermimic680,
        ),
        (
            "policies/supermimic/model_kyrieyoung.onnx",
            "kyrieyoung.onnx",
            "1df68386d18dfc7539952264fdd02de92afe02d69cb99be54724d789cd29f31f",
            f"{DRIVE}1EOrVvgyc10NvSu-QtVG4rNbV5BnwvV_P",
            export_supermimic680,
        ),
        (
            "policies/supermimic_breezewrf/model.onnx",
            "policy_10000.pt",
            "eddabf5ebcbcf81d637d9ad320daeda192587f23ab527b58e4dcd5c992a6cdef",
            f"{GH}/Breezewrf/supermimic_mjlab/aab90addd0e783b007691442907bc8c9ce145970/checkpoints/preview/policy_10000.pt",
            export_supermimic_breezewrf,
        ),
        (
            "policies/supermimic/model_dyn.onnx",
            "ckpt_0040000.pt",
            "182b7a5d05b6c3e4364d6223d1cf52ba18e6573563e961f6e019d6d445340c3c",
            f"{HF}/unpackableorange/SuperMimic_Dyn/resolve/669542a05cbef8debd6d008348286db253e3ffe2/ckpt_0040000.pt",
            export_supermimic689,
        ),
        (
            "policies/supermimic/model_m5.onnx",
            "m5_lafan1_21000.pt",
            "4bc2bd71e4809997b704293d5a8c8e8bf5715fe0c3a27a053a895a1eff418832",
            "https://github.com/WXyuany/supermimic-m5/releases/download/v0.1.0/m5_lafan1_21000.pt",
            export_supermimic689,
        ),
    ):
        export([dst], [(name, want, url)], lambda p, _: fn(p[name], dst))

    export(
        [
            "policies/supermimic_vibed/model_panhainan.onnx",
            "policies/supermimic_vibed/model_omegai.onnx",
        ],
        [
            (
                "policy_098000.pt",
                "9e4b9a075382b38006824954327776e63732280b62df881ad7596ee60023cddb",
                f"{GH}/panhainan2020/SuperMimic-MuJoCo/db045afa9c52458f213e2c66037387e9c109a03a/checkpoints/g1_29dof/policy_098000.pt",
            ),
            (
                "SuperMimicKit-g1-55k5-onnx-fp32.zip",
                "1e68e45a2b8c1214746fcb28301aa3c3710050df1f92332dbfb713b03f71030f",
                "https://github.com/OMEGA-i/SuperMimicKit/releases/download/demo-preview-20261024/SuperMimicKit-g1-55k5-onnx-fp32.zip",
            ),
        ],
        lambda p, _: export_supermimic_vibed(
            p["policy_098000.pt"],
            "policies/supermimic_vibed/model_panhainan.onnx",
            p["SuperMimicKit-g1-55k5-onnx-fp32.zip"],
            "g1/checkpoints/55500-20261023T202120655356/policy.onnx",
            "fcc99d13c3ac6d56942ccfc0fa045982672cbd4e12366d36fa90072489c1b215",
            "policies/supermimic_vibed/model_omegai.onnx",
        ),
    )

    sonic = f"{HF}/nvidia/GEAR-SONIC/resolve/6733128a3d8a523b1418b06bca3cdf61c8b0987f"

    def sonic_planner(
        p,
        tmp,
    ):
        context = np.zeros((1, SONIC_CTX, SONIC_QPOS), np.float32)
        context[:, :, 2] = SONIC_HEIGHT
        context[:, :, 3] = 1.0
        allowed = np.zeros((1, 11), np.float32)
        allowed[0, :6] = 1.0
        f32 = os.path.join(tmp, "planner_sonic_f32.onnx")
        export_float_io(
            p["planner_sonic.onnx"],
            f32,
            {
                "context_mujoco_qpos": context,
                "mode": np.zeros(1, np.float32),
                "target_vel": np.full(1, 0.8, np.float32),
                "movement_direction": np.array([[1.0, 0.0, 0.0]], np.float32),
                "facing_direction": np.array([[1.0, 0.0, 0.0]], np.float32),
                "random_seed": np.full(1, 1234.0, np.float32),
                "height": np.full(1, -1.0, np.float32),
                "has_specific_target": np.zeros((1, 1), np.float32),
                "specific_target_positions": np.zeros((1, SONIC_CTX, 3), np.float32),
                "specific_target_headings": np.zeros((1, SONIC_CTX), np.float32),
                "allowed_pred_num_tokens": allowed,
            },
        )
        export_rows(f32, "policies/sonic/model_planner_rows.onnx", sonic_planner_feed)

    export(
        ["policies/sonic/model_planner_rows.onnx"],
        [
            (
                "planner_sonic.onnx",
                "39b553e197f62f077975ba38512bc04781a3fc37c2af7c6756e04629f760edea",
                f"{sonic}/planner_sonic.onnx",
            )
        ],
        sonic_planner,
    )
    export(
        ["policies/sonic/model_encoder_rows.onnx"],
        [
            (
                "model_encoder.onnx",
                "013ab0287236aa2721e13f1e936d699db982302d0de0bfcdae76d5c3245362d3",
                f"{sonic}/model_encoder.onnx",
            )
        ],
        lambda p, _: export_rows(
            p["model_encoder.onnx"],
            "policies/sonic/model_encoder_rows.onnx",
            sonic_encoder_feed,
        ),
    )


if __name__ == "__main__":
    main()

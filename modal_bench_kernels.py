from __future__ import annotations

import os
import time
from pathlib import Path

import modal


APP = modal.App("klein4b-kernel-bench")


def _repo_root() -> str:
    return str(Path(__file__).resolve().parent)


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch")
    .pip_install("numpy")
    .pip_install("ninja", "setuptools", "wheel")
    .add_local_dir(_repo_root(), remote_path="/root/klein4B")
)


@APP.function(
    image=image,
    gpu="A100",
    timeout=60 * 30,
)
def benchmark() -> None:
    import torch

    repo = Path("/root/klein4B")
    cuda_dir = repo / "cuda_kernels"
    os.chdir(cuda_dir)
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "8.0")

    from torch.utils.cpp_extension import load

    ext = load(
        name="klein_cuda_ext",
        sources=["src/ops.cpp", "src/ops_cuda.cu"],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        is_python_module=False,
        verbose=True,
    )

    ns = torch.ops.klein_cuda
    print(f"joint_packed_attention_present={hasattr(ns, 'joint_packed_attention_')}")

    device = "cuda"
    torch.manual_seed(0)

    def bench(fn, iters: int = 200, warmup: int = 50) -> float:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / iters

    # silu_mul
    gate = torch.randn(4096, 3072, device=device, dtype=torch.float32)
    up = torch.randn_like(gate)
    gate2 = gate.clone()
    up2 = up.clone()

    def ref_silu():
        gate2.copy_(gate)
        torch.mul(torch.nn.functional.silu(gate2), up2, out=gate2)

    def ker_silu():
        gate2.copy_(gate)
        ns.silu_mul_(gate2, up2)

    # adaln_norm
    x = torch.randn(864, 3072, device=device, dtype=torch.float32)
    shift = torch.randn(3072, device=device, dtype=torch.float32)
    scale = torch.randn(3072, device=device, dtype=torch.float32)

    def ref_adaln():
        y = x - x.mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return (1.0 + scale) * y + shift

    def ker_adaln():
        return ns.adaln_norm(x, shift, scale, 1e-6)

    # qk_rms_norm
    q = torch.randn(864, 24, 128, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    qw = torch.randn(128, device=device, dtype=torch.float32)
    kw = torch.randn(128, device=device, dtype=torch.float32)

    def ref_qk():
        qn = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * qw
        kn = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * kw
        return qn, kn

    def ker_qk():
        q2 = q.clone()
        k2 = k.clone()
        return ns.qk_rms_norm_(q2, k2, qw, kw, 1e-6)

    # rope_2d_offset
    xrope = torch.randn(864, 24, 128, device=device, dtype=torch.float32)
    cos = torch.randn(864, 128, device=device, dtype=torch.float32)
    sin = torch.randn_like(cos)

    def ref_rope():
        y = xrope.clone()
        part = y[:864]
        a = part[..., 0::2]
        b = part[..., 1::2]
        c = cos[:864, ::2]
        s = sin[:864, ::2]
        a2 = a * c.unsqueeze(1) - b * s.unsqueeze(1)
        b2 = b * c.unsqueeze(1) + a * s.unsqueeze(1)
        part[..., 0::2] = a2
        part[..., 1::2] = b2
        return y

    def ker_rope():
        return ns.rope_2d_offset_(xrope, cos, sin, 0, 864)

    # Composite proxy for a fused transformer inner path.
    seq = 864
    hidden = 3072
    heads = 24
    head_dim = 128
    x = torch.randn(seq, hidden, device=device, dtype=torch.float32)
    shift2 = torch.randn(hidden, device=device, dtype=torch.float32)
    scale2 = torch.randn(hidden, device=device, dtype=torch.float32)
    q = torch.randn(seq, heads, head_dim, device=device, dtype=torch.float32)
    k = torch.randn_like(q)
    qw2 = torch.randn(head_dim, device=device, dtype=torch.float32)
    kw2 = torch.randn(head_dim, device=device, dtype=torch.float32)
    cos2 = torch.randn(seq, head_dim, device=device, dtype=torch.float32)
    sin2 = torch.randn_like(cos2)
    v = torch.randn_like(q)
    gate3 = torch.randn(seq, hidden, device=device, dtype=torch.float32)
    up3 = torch.randn_like(gate3)
    proj = torch.randn_like(x)

    def ref_fused_proxy():
        y = x - x.mean(dim=-1, keepdim=True)
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        y = (1.0 + scale2) * y + shift2
        qn = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * qw2
        kn = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * kw2
        qn = qn.clone()
        kn = kn.clone()
        a = qn[..., 0::2]
        b = qn[..., 1::2]
        c = cos2[:, ::2]
        s = sin2[:, ::2]
        qn[..., 0::2] = a * c.unsqueeze(1) - b * s.unsqueeze(1)
        qn[..., 1::2] = b * c.unsqueeze(1) + a * s.unsqueeze(1)
        a = kn[..., 0::2]
        b = kn[..., 1::2]
        kn[..., 0::2] = a * c.unsqueeze(1) - b * s.unsqueeze(1)
        kn[..., 1::2] = b * c.unsqueeze(1) + a * s.unsqueeze(1)
        attn = torch.nn.functional.scaled_dot_product_attention(
            qn.transpose(0, 1),
            kn.transpose(0, 1),
            v.transpose(0, 1),
            is_causal=False,
        ).transpose(0, 1)
        gate = torch.nn.functional.silu(gate3)
        mlp = gate * up3
        return y + attn.transpose(0, 1).reshape(seq, hidden) + mlp + proj

    def ker_fused_proxy():
        y = ns.adaln_norm(x, shift2, scale2, 1e-6)
        q2 = q.clone()
        k2 = k.clone()
        ns.qk_rms_norm_(q2, k2, qw2, kw2, 1e-6)
        ns.rope_2d_offset_(q2, cos2, sin2, 0, seq)
        ns.rope_2d_offset_(k2, cos2, sin2, 0, seq)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q2.transpose(0, 1),
            k2.transpose(0, 1),
            v.transpose(0, 1),
            is_causal=False,
        ).transpose(0, 1)
        gate4 = gate3.clone()
        ns.silu_mul_(gate4, up3)
        return y + attn.transpose(0, 1).reshape(seq, hidden) + gate4 + proj

    # Double-stream joint attention smoke check.
    q_hidden = torch.randn(128, heads, head_dim, device=device, dtype=torch.float32)
    k_hidden = torch.randn_like(q_hidden)
    v_hidden = torch.randn_like(q_hidden)
    q_context = torch.randn(64, heads, head_dim, device=device, dtype=torch.float32)
    k_context = torch.randn_like(q_context)
    v_context = torch.randn_like(q_context)

    def ker_joint_attention():
        if hasattr(ns, "joint_packed_attention_"):
            return ns.joint_packed_attention_(
                q_hidden,
                k_hidden,
                v_hidden,
                q_context,
                k_context,
                v_context,
                1.0 / (head_dim ** 0.5),
            )
        raise RuntimeError("joint_packed_attention_ is not registered")

    print("joint_packed_attention_smoke_shape=", [tuple(t.shape) for t in ker_joint_attention()])

    def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max().item())

    print("fused_proxy_diff=", max_abs_diff(ref_fused_proxy(), ker_fused_proxy()))

    # Realistic double-block proxy.
    img_seq = 864
    txt_seq = 512
    hidden = 3072
    heads = 24
    head_dim = 128
    mlp_hidden = 9216
    img_hidden = torch.randn(img_seq, hidden, device=device, dtype=torch.float32)
    txt_hidden = torch.randn(txt_seq, hidden, device=device, dtype=torch.float32)
    img_shift1 = torch.randn(hidden, device=device, dtype=torch.float32)
    img_scale1 = torch.randn(hidden, device=device, dtype=torch.float32)
    txt_shift1 = torch.randn(hidden, device=device, dtype=torch.float32)
    txt_scale1 = torch.randn(hidden, device=device, dtype=torch.float32)
    img_gate1 = torch.randn(hidden, device=device, dtype=torch.float32)
    txt_gate1 = torch.randn(hidden, device=device, dtype=torch.float32)
    img_qw = torch.randn(head_dim, device=device, dtype=torch.float32)
    img_kw = torch.randn(head_dim, device=device, dtype=torch.float32)
    txt_qw = torch.randn(head_dim, device=device, dtype=torch.float32)
    txt_kw = torch.randn(head_dim, device=device, dtype=torch.float32)
    img_q_w = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    img_k_w = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    img_v_w = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    txt_q_w = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    txt_k_w = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    txt_v_w = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    img_proj_w = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    txt_proj_w = torch.randn(hidden, hidden, device=device, dtype=torch.float32)
    img_cos = torch.randn(img_seq, head_dim, device=device, dtype=torch.float32)
    img_sin = torch.randn_like(img_cos)
    txt_cos = torch.randn(txt_seq, head_dim, device=device, dtype=torch.float32)
    txt_sin = torch.randn_like(txt_cos)
    img_v = torch.randn(img_seq, heads, head_dim, device=device, dtype=torch.float32)
    txt_v = torch.randn(txt_seq, heads, head_dim, device=device, dtype=torch.float32)

    def ref_block_style():
        ih = img_hidden
        th = txt_hidden
        img_norm = (1.0 + img_scale1) * (
            (ih - ih.mean(dim=-1, keepdim=True))
            * torch.rsqrt((ih - ih.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ) + img_shift1
        txt_norm = (1.0 + txt_scale1) * (
            (th - th.mean(dim=-1, keepdim=True))
            * torch.rsqrt((th - th.mean(dim=-1, keepdim=True)).pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        ) + txt_shift1
        img_q = img_norm @ img_q_w
        img_k = img_norm @ img_k_w
        img_v_local = img_norm @ img_v_w
        txt_q = txt_norm @ txt_q_w
        txt_k = txt_norm @ txt_k_w
        txt_v_local = txt_norm @ txt_v_w
        img_q = img_q.view(img_seq, heads, head_dim)
        img_k = img_k.view(img_seq, heads, head_dim)
        txt_q = txt_q.view(txt_seq, heads, head_dim)
        txt_k = txt_k.view(txt_seq, heads, head_dim)
        img_q = img_q * torch.rsqrt(img_q.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * img_qw
        img_k = img_k * torch.rsqrt(img_k.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * img_kw
        txt_q = txt_q * torch.rsqrt(txt_q.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * txt_qw
        txt_k = txt_k * torch.rsqrt(txt_k.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * txt_kw
        img_q = img_q.clone()
        img_k = img_k.clone()
        txt_q = txt_q.clone()
        txt_k = txt_k.clone()
        img_q[..., 0::2] = img_q[..., 0::2] * img_cos[:, ::2].unsqueeze(1) - img_q[..., 1::2] * img_sin[:, ::2].unsqueeze(1)
        img_q[..., 1::2] = img_q[..., 1::2] * img_cos[:, ::2].unsqueeze(1) + img_q[..., 0::2] * img_sin[:, ::2].unsqueeze(1)
        img_k[..., 0::2] = img_k[..., 0::2] * img_cos[:, ::2].unsqueeze(1) - img_k[..., 1::2] * img_sin[:, ::2].unsqueeze(1)
        img_k[..., 1::2] = img_k[..., 1::2] * img_cos[:, ::2].unsqueeze(1) + img_k[..., 0::2] * img_sin[:, ::2].unsqueeze(1)
        txt_q[..., 0::2] = txt_q[..., 0::2] * txt_cos[:, ::2].unsqueeze(1) - txt_q[..., 1::2] * txt_sin[:, ::2].unsqueeze(1)
        txt_q[..., 1::2] = txt_q[..., 1::2] * txt_cos[:, ::2].unsqueeze(1) + txt_q[..., 0::2] * txt_sin[:, ::2].unsqueeze(1)
        txt_k[..., 0::2] = txt_k[..., 0::2] * txt_cos[:, ::2].unsqueeze(1) - txt_k[..., 1::2] * txt_sin[:, ::2].unsqueeze(1)
        txt_k[..., 1::2] = txt_k[..., 1::2] * txt_cos[:, ::2].unsqueeze(1) + txt_k[..., 0::2] * txt_sin[:, ::2].unsqueeze(1)
        img_attn = torch.nn.functional.scaled_dot_product_attention(img_q.transpose(0, 1), torch.cat([txt_k, img_k], dim=0).transpose(0, 1), torch.cat([txt_v_local.view(txt_seq, heads, head_dim), img_v_local.view(img_seq, heads, head_dim)], dim=0).transpose(0, 1), is_causal=False).transpose(0, 1)
        txt_attn = torch.nn.functional.scaled_dot_product_attention(txt_q.transpose(0, 1), torch.cat([txt_k, img_k], dim=0).transpose(0, 1), torch.cat([txt_v_local.view(txt_seq, heads, head_dim), img_v_local.view(img_seq, heads, head_dim)], dim=0).transpose(0, 1), is_causal=False).transpose(0, 1)
        img_proj = img_attn.reshape(img_seq, hidden) @ img_proj_w
        txt_proj = txt_attn.reshape(txt_seq, hidden) @ txt_proj_w
        img_out = img_hidden + img_proj * img_gate1
        txt_out = txt_hidden + txt_proj * txt_gate1
        return img_out + txt_out.mean()

    def ker_block_style():
        img_norm = ns.adaln_norm(img_hidden, img_shift1, img_scale1, 1e-6)
        txt_norm = ns.adaln_norm(txt_hidden, txt_shift1, txt_scale1, 1e-6)
        img_q = (img_norm @ img_q_w).view(img_seq, heads, head_dim)
        img_k = (img_norm @ img_k_w).view(img_seq, heads, head_dim)
        img_v_local = (img_norm @ img_v_w).view(img_seq, heads, head_dim)
        txt_q = (txt_norm @ txt_q_w).view(txt_seq, heads, head_dim)
        txt_k = (txt_norm @ txt_k_w).view(txt_seq, heads, head_dim)
        txt_v_local = (txt_norm @ txt_v_w).view(txt_seq, heads, head_dim)
        img_q = img_q.clone()
        img_k = img_k.clone()
        txt_q = txt_q.clone()
        txt_k = txt_k.clone()
        ns.qk_rms_norm_(img_q, img_k, img_qw, img_kw, 1e-6)
        ns.qk_rms_norm_(txt_q, txt_k, txt_qw, txt_kw, 1e-6)
        ns.rope_2d_offset_(img_q, img_cos, img_sin, 0, img_seq)
        ns.rope_2d_offset_(img_k, img_cos, img_sin, 0, img_seq)
        ns.rope_2d_offset_(txt_q, txt_cos, txt_sin, 0, txt_seq)
        ns.rope_2d_offset_(txt_k, txt_cos, txt_sin, 0, txt_seq)
        kv = torch.cat([txt_k, img_k], dim=0)
        vv = torch.cat([txt_v_local, img_v_local], dim=0)
        img_attn = torch.nn.functional.scaled_dot_product_attention(img_q.transpose(0, 1), kv.transpose(0, 1), vv.transpose(0, 1), is_causal=False).transpose(0, 1)
        txt_attn = torch.nn.functional.scaled_dot_product_attention(txt_q.transpose(0, 1), kv.transpose(0, 1), vv.transpose(0, 1), is_causal=False).transpose(0, 1)
        img_proj = img_attn.reshape(img_seq, hidden) @ img_proj_w
        txt_proj = txt_attn.reshape(txt_seq, hidden) @ txt_proj_w
        img_out = img_hidden + img_proj * img_gate1
        txt_out = txt_hidden + txt_proj * txt_gate1
        return img_out + txt_out.mean()

    print("block_style_diff=", max_abs_diff(ref_block_style(), ker_block_style()))

    # Lower-materialization proxy for the deeper live-path cleanup:
    # keep the attention stream shapes closer to the model path and avoid
    # unnecessary clones/transposes in the kernel side.
    def ref_deep_fused_proxy():
        img_norm = (img_hidden - img_hidden.mean(dim=-1, keepdim=True))
        img_norm = img_norm * torch.rsqrt(img_norm.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        img_norm = (1.0 + img_scale1) * img_norm + img_shift1
        txt_norm = (txt_hidden - txt_hidden.mean(dim=-1, keepdim=True))
        txt_norm = txt_norm * torch.rsqrt(txt_norm.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        txt_norm = (1.0 + txt_scale1) * txt_norm + txt_shift1

        img_q = (img_norm @ img_q_w).view(img_seq, heads, head_dim)
        img_k = (img_norm @ img_k_w).view(img_seq, heads, head_dim)
        img_v_local = (img_norm @ img_v_w).view(img_seq, heads, head_dim)
        txt_q = (txt_norm @ txt_q_w).view(txt_seq, heads, head_dim)
        txt_k = (txt_norm @ txt_k_w).view(txt_seq, heads, head_dim)
        txt_v_local = (txt_norm @ txt_v_w).view(txt_seq, heads, head_dim)

        q = torch.cat((txt_q, img_q), dim=0).transpose(0, 1)
        k = torch.cat((txt_k, img_k), dim=0).transpose(0, 1)
        v = torch.cat((txt_v_local, img_v_local), dim=0).transpose(0, 1)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(0, 1)

        img_attn = attn[txt_seq:]
        txt_attn = attn[:txt_seq]
        img_proj = img_attn.reshape(img_seq, hidden) @ img_proj_w
        txt_proj = txt_attn.reshape(txt_seq, hidden) @ txt_proj_w
        gate = torch.nn.functional.silu(gate3)
        mlp = gate * up3
        return img_proj.mean() + txt_proj.mean() + mlp.mean()

    def ker_deep_fused_proxy():
        img_norm = ns.adaln_norm(img_hidden, img_shift1, img_scale1, 1e-6)
        txt_norm = ns.adaln_norm(txt_hidden, txt_shift1, txt_scale1, 1e-6)
        img_q = (img_norm @ img_q_w).view(img_seq, heads, head_dim)
        img_k = (img_norm @ img_k_w).view(img_seq, heads, head_dim)
        img_v_local = (img_norm @ img_v_w).view(img_seq, heads, head_dim)
        txt_q = (txt_norm @ txt_q_w).view(txt_seq, heads, head_dim)
        txt_k = (txt_norm @ txt_k_w).view(txt_seq, heads, head_dim)
        txt_v_local = (txt_norm @ txt_v_w).view(txt_seq, heads, head_dim)

        q = torch.cat((txt_q, img_q), dim=0).transpose(0, 1)
        k = torch.cat((txt_k, img_k), dim=0).transpose(0, 1)
        v = torch.cat((txt_v_local, img_v_local), dim=0).transpose(0, 1)
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False).transpose(0, 1)

        img_attn = attn[txt_seq:]
        txt_attn = attn[:txt_seq]
        img_proj = img_attn.reshape(img_seq, hidden) @ img_proj_w
        txt_proj = txt_attn.reshape(txt_seq, hidden) @ txt_proj_w
        gate = gate3.clone()
        ns.silu_mul_(gate, up3)
        return img_proj.mean() + txt_proj.mean() + gate.mean()

    print("deep_fused_diff=", abs(ref_deep_fused_proxy().item() - ker_deep_fused_proxy().item()))

    # Exact shape/layout proxy for the Flux2 hooks we wired into model.py.
    hook_batch = 1
    hook_seq = 864
    hook_heads = 24
    hook_head_dim = 128
    q_hook = torch.randn(hook_batch, hook_heads, hook_seq, hook_head_dim, device=device, dtype=torch.float32)
    k_hook = torch.randn_like(q_hook)
    gate_hook = torch.randn(4096, 3072, device=device, dtype=torch.float32)
    up_hook = torch.randn_like(gate_hook)
    pe_hook = torch.randn(hook_batch, 1, hook_seq, hook_head_dim, 2, 2, device=device, dtype=torch.float32)

    def ref_model_hook():
        q = q_hook.permute(0, 2, 1, 3).reshape(-1, hook_heads, hook_head_dim).contiguous()
        k = k_hook.permute(0, 2, 1, 3).reshape(-1, hook_heads, hook_head_dim).contiguous()
        q = q * torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * qw
        k = k * torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + 1e-6) * kw
        q = q.reshape(hook_batch, hook_seq, hook_heads, hook_head_dim).permute(0, 2, 1, 3).contiguous()
        k = k.reshape(hook_batch, hook_seq, hook_heads, hook_head_dim).permute(0, 2, 1, 3).contiguous()
        q3 = q.squeeze(0).permute(1, 0, 2).contiguous()
        k3 = k.squeeze(0).permute(1, 0, 2).contiguous()
        c = pe_hook[0, 0, :, :, 0, 0]
        s = pe_hook[0, 0, :, :, 1, 0]
        q3 = q3.clone()
        k3 = k3.clone()
        q3[..., 0::2] = q3[..., 0::2] * c[:, ::2].unsqueeze(1) - q3[..., 1::2] * s[:, ::2].unsqueeze(1)
        q3[..., 1::2] = q3[..., 1::2] * c[:, ::2].unsqueeze(1) + q3[..., 0::2] * s[:, ::2].unsqueeze(1)
        k3[..., 0::2] = k3[..., 0::2] * c[:, ::2].unsqueeze(1) - k3[..., 1::2] * s[:, ::2].unsqueeze(1)
        k3[..., 1::2] = k3[..., 1::2] * c[:, ::2].unsqueeze(1) + k3[..., 0::2] * s[:, ::2].unsqueeze(1)
        gate = torch.nn.functional.silu(gate_hook)
        return q3.mean() + k3.mean() + gate.mul(up_hook).mean()

    def ker_model_hook():
        q = q_hook.clone()
        k = k_hook.clone()
        qn = q.permute(0, 2, 1, 3).reshape(-1, hook_heads, hook_head_dim).contiguous()
        kn = k.permute(0, 2, 1, 3).reshape(-1, hook_heads, hook_head_dim).contiguous()
        ns.qk_rms_norm_(qn, kn, qw, kw, 1e-6)
        q = qn.reshape(hook_batch, hook_seq, hook_heads, hook_head_dim).permute(0, 2, 1, 3)
        k = kn.reshape(hook_batch, hook_seq, hook_heads, hook_head_dim).permute(0, 2, 1, 3)
        q3 = q.squeeze(0).permute(1, 0, 2).contiguous()
        k3 = k.squeeze(0).permute(1, 0, 2).contiguous()
        c = pe_hook[0, 0, :, :, 0, 0].contiguous()
        s = pe_hook[0, 0, :, :, 1, 0].contiguous()
        ns.rope_2d_offset_(q3, c, s, 0, hook_seq)
        ns.rope_2d_offset_(k3, c, s, 0, hook_seq)
        gate = gate_hook.clone()
        ns.silu_mul_(gate, up_hook)
        return q3.mean() + k3.mean() + gate.mean()

    print("model_hook_diff=", abs(ref_model_hook().item() - ker_model_hook().item()))

    # GPU residency / reuse: keep tensors on device across repeated calls.
    persistent_gate = torch.randn(16384, 1024, device=device, dtype=torch.float32)
    persistent_up = torch.randn_like(persistent_gate)

    def ref_resident():
        y = persistent_gate.clone()
        torch.mul(torch.nn.functional.silu(y), persistent_up, out=y)
        return y

    def ker_resident():
        y = persistent_gate.clone()
        ns.silu_mul_(y, persistent_up)
        return y

    print("resident_diff=", abs(ref_resident().mean().item() - ker_resident().mean().item()))

    # Attention backend sweep on a representative model-shaped tensor layout.
    attn_q = torch.randn(1, 24, 864, 128, device=device, dtype=torch.bfloat16)
    attn_k = torch.randn_like(attn_q)
    attn_v = torch.randn_like(attn_q)

    def ref_attn_default():
        return torch.nn.functional.scaled_dot_product_attention(attn_q, attn_k, attn_v)

    def ref_attn_flash_ctx():
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_mem_efficient=False, enable_math=False):
            return torch.nn.functional.scaled_dot_product_attention(attn_q, attn_k, attn_v)

    def ref_attn_mem_ctx():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=True, enable_math=False):
            return torch.nn.functional.scaled_dot_product_attention(attn_q, attn_k, attn_v)

    def ref_attn_math_ctx():
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True):
            return torch.nn.functional.scaled_dot_product_attention(attn_q, attn_k, attn_v)

    results = {
        "silu_ref_ms": bench(ref_silu),
        "silu_ker_ms": bench(ker_silu),
        "adaln_ref_ms": bench(ref_adaln),
        "adaln_ker_ms": bench(ker_adaln),
        "qk_ref_ms": bench(ref_qk),
        "qk_ker_ms": bench(ker_qk),
        "rope_ref_ms": bench(ref_rope),
        "rope_ker_ms": bench(ker_rope),
        "fused_proxy_ref_ms": bench(ref_fused_proxy),
        "fused_proxy_ker_ms": bench(ker_fused_proxy),
        "block_style_ref_ms": bench(ref_block_style),
        "block_style_ker_ms": bench(ker_block_style),
        "deep_fused_ref_ms": bench(ref_deep_fused_proxy),
        "deep_fused_ker_ms": bench(ker_deep_fused_proxy),
        "model_hook_ref_ms": bench(ref_model_hook),
        "model_hook_ker_ms": bench(ker_model_hook),
        "resident_ref_ms": bench(ref_resident),
        "resident_ker_ms": bench(ker_resident),
        "attn_default_ms": bench(ref_attn_default),
        "attn_flash_ctx_ms": bench(ref_attn_flash_ctx),
        "attn_mem_ctx_ms": bench(ref_attn_mem_ctx),
        "attn_math_ctx_ms": bench(ref_attn_math_ctx),
    }

    for k, v in results.items():
        print(f"{k}={v:.4f}")
    print(
        "fused_proxy_speedup=%.3fx"
        % (results["fused_proxy_ref_ms"] / max(results["fused_proxy_ker_ms"], 1e-9))
    )
    print(
        "resident_speedup=%.3fx"
        % (results["resident_ref_ms"] / max(results["resident_ker_ms"], 1e-9))
    )
    print(
        "deep_fused_speedup=%.3fx"
        % (results["deep_fused_ref_ms"] / max(results["deep_fused_ker_ms"], 1e-9))
    )


@APP.local_entrypoint()
def main() -> None:
    benchmark.remote()

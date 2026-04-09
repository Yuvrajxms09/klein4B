from __future__ import annotations

import ctypes
from pathlib import Path

import torch

_FLUX_TEXT_DIM = 7680


class KleinCUDABridge:
    def __init__(self, model_dir: str, lib_path: str, use_mmap: bool = True) -> None:
        self._lib = ctypes.CDLL(str(lib_path))
        self._lib.klein_bridge_load.argtypes = [ctypes.c_char_p, ctypes.c_int]
        self._lib.klein_bridge_load.restype = ctypes.c_void_p
        self._lib.klein_bridge_free.argtypes = [ctypes.c_void_p]
        self._lib.klein_bridge_free.restype = None
        self._lib.klein_bridge_denoise_with_refs.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.klein_bridge_denoise_with_refs.restype = ctypes.c_int
        self._lib.klein_bridge_denoise_with_multi_refs.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        self._lib.klein_bridge_denoise_with_multi_refs.restype = ctypes.c_int

        self._ctx = self._lib.klein_bridge_load(model_dir.encode("utf-8"), 1 if use_mmap else 0)
        if not self._ctx:
            raise RuntimeError(f"Failed to load klein-cuda-c context from {model_dir}")

    def close(self) -> None:
        if self._ctx:
            self._lib.klein_bridge_free(self._ctx)
            self._ctx = None

    def __del__(self) -> None:
        self.close()

    @staticmethod
    def _to_ptr(t: torch.Tensor) -> ctypes.POINTER(ctypes.c_float):
        if t.dtype != torch.float32 or not t.is_contiguous() or t.device.type != "cpu":
            raise ValueError("tensor must be contiguous float32 on CPU")
        return ctypes.cast(int(t.data_ptr()), ctypes.POINTER(ctypes.c_float))

    def denoise_with_refs(
        self,
        target_nchw: torch.Tensor,
        refs_nchw: list[torch.Tensor],
        ref_t_offsets: list[int],
        text_emb: torch.Tensor,
        timestep_0_to_1: float,
    ) -> torch.Tensor:
        if target_nchw.dtype != torch.float32:
            raise TypeError("target_nchw must be float32")
        if text_emb.dtype != torch.float32:
            raise TypeError("text_emb must be float32")
        if target_nchw.ndim != 3 or target_nchw.shape[0] != 128:
            raise ValueError("target_nchw must have shape [128, H, W]")
        if text_emb.ndim != 2:
            raise ValueError("text_emb must have shape [text_seq, text_dim]")
        if int(text_emb.shape[1]) != _FLUX_TEXT_DIM:
            raise ValueError(f"text_emb second dimension must be {_FLUX_TEXT_DIM}, got {int(text_emb.shape[1])}")

        latent_h, latent_w = int(target_nchw.shape[1]), int(target_nchw.shape[2])
        out = torch.empty_like(target_nchw, device="cpu", dtype=torch.float32)

        target_c = target_nchw.contiguous()
        text_c = text_emb.contiguous()

        if len(refs_nchw) == 0:
            rc = self._lib.klein_bridge_denoise_with_refs(
                self._ctx,
                self._to_ptr(target_c),
                latent_h,
                latent_w,
                ctypes.POINTER(ctypes.c_float)(),
                0,
                0,
                0,
                self._to_ptr(text_c),
                int(text_c.shape[0]),
                ctypes.c_float(float(timestep_0_to_1)),
                self._to_ptr(out),
            )
        elif len(refs_nchw) == 1:
            if len(ref_t_offsets) != 1:
                raise ValueError("ref_t_offsets must have length 1 for single-ref path")
            ref_c = refs_nchw[0].contiguous()
            if ref_c.shape[0] != 128:
                raise ValueError("ref tensor must be [128, H, W]")
            rc = self._lib.klein_bridge_denoise_with_refs(
                self._ctx,
                self._to_ptr(target_c),
                latent_h,
                latent_w,
                self._to_ptr(ref_c),
                int(ref_c.shape[1]),
                int(ref_c.shape[2]),
                int(ref_t_offsets[0]),
                self._to_ptr(text_c),
                int(text_c.shape[0]),
                ctypes.c_float(float(timestep_0_to_1)),
                self._to_ptr(out),
            )
        else:
            if len(refs_nchw) != len(ref_t_offsets):
                raise ValueError("refs_nchw and ref_t_offsets length mismatch")
            num_refs = len(refs_nchw)
            ref_ptr_arr = (ctypes.POINTER(ctypes.c_float) * num_refs)()
            ref_h_arr = (ctypes.c_int * num_refs)()
            ref_w_arr = (ctypes.c_int * num_refs)()
            ref_t_arr = (ctypes.c_int * num_refs)()
            ref_cpu = []
            for i, (ref, t_off) in enumerate(zip(refs_nchw, ref_t_offsets)):
                ref_c = ref.contiguous()
                if ref_c.shape[0] != 128:
                    raise ValueError("each ref tensor must be [128, H, W]")
                ref_cpu.append(ref_c)
                ref_ptr_arr[i] = self._to_ptr(ref_c)
                ref_h_arr[i] = int(ref_c.shape[1])
                ref_w_arr[i] = int(ref_c.shape[2])
                ref_t_arr[i] = int(t_off)
            rc = self._lib.klein_bridge_denoise_with_multi_refs(
                self._ctx,
                self._to_ptr(target_c),
                latent_h,
                latent_w,
                ref_ptr_arr,
                ref_h_arr,
                ref_w_arr,
                ref_t_arr,
                ctypes.c_int(num_refs),
                self._to_ptr(text_c),
                int(text_c.shape[0]),
                ctypes.c_float(float(timestep_0_to_1)),
                self._to_ptr(out),
            )
        if rc != 1:
            raise RuntimeError("klein-cuda-c denoise call failed")
        return out


def _tokens_to_nchw(pipe: object, tokens: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 2:
        raise ValueError("tokens must be [seq, channels]")
    if ids.ndim != 2:
        raise ValueError("ids must be [seq, 4]")
    return pipe._unpack_latents_with_ids(tokens.unsqueeze(0), ids.unsqueeze(0))[0]


def make_klein_c_denoiser(
    pipe: object,
    *,
    model_dir: str,
    bridge_lib_path: str | None = None,
    enforce_single_ref: bool = False,
) -> object:
    if bridge_lib_path is None:
        bridge_lib_path = str(Path(__file__).with_name("klein_c_bridge") / "libklein_bridge.so")
    bridge = KleinCUDABridge(model_dir=model_dir, lib_path=bridge_lib_path, use_mmap=True)

    def _denoiser(
        *,
        transformer: object,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        txt_ids: torch.Tensor,
        img_ids: torch.Tensor,
        joint_attention_kwargs: dict | None,
        context: str,
    ) -> torch.Tensor:
        del transformer, txt_ids, joint_attention_kwargs, context
        if hidden_states.device.type != "cuda":
            raise RuntimeError("klein-cuda-c bridge expects CUDA pipeline inputs")
        if hidden_states.shape[0] != 1:
            raise RuntimeError("klein-cuda-c bridge currently supports batch size 1")

        ids = img_ids[0]
        hs = hidden_states[0]
        t_values = ids[:, 0].to(torch.int64)
        target_mask = t_values == 0
        ref_mask = t_values > 0
        if target_mask.sum().item() == 0:
            raise RuntimeError("No target tokens found (t=0)")

        target_tokens = hs[target_mask]
        target_ids = ids[target_mask]
        target_nchw = _tokens_to_nchw(pipe, target_tokens, target_ids).contiguous()

        refs_nchw: list[torch.Tensor] = []
        ref_t_offsets: list[int] = []
        if ref_mask.any():
            ref_t_unique = torch.unique(t_values[ref_mask]).tolist()
            ref_t_unique = sorted(int(v) for v in ref_t_unique)
            if enforce_single_ref and len(ref_t_unique) != 1:
                raise RuntimeError("klein-cuda-c bridge configured for single ref, but multiple refs were provided")
            for ref_t in ref_t_unique:
                this_ref_mask = t_values == ref_t
                ref_tokens = hs[this_ref_mask]
                ref_ids = ids[this_ref_mask]
                refs_nchw.append(_tokens_to_nchw(pipe, ref_tokens, ref_ids).contiguous())
                ref_t_offsets.append(ref_t)

        target_cpu = target_nchw.detach().to(dtype=torch.float32, device="cpu").contiguous()
        refs_cpu = [r.detach().to(dtype=torch.float32, device="cpu").contiguous() for r in refs_nchw]
        text_cpu = encoder_hidden_states[0].detach().to(dtype=torch.float32, device="cpu").contiguous()
        t_01 = float(timestep[0].detach().to(dtype=torch.float32).item() / 1000.0)

        out_cpu = bridge.denoise_with_refs(
            target_nchw=target_cpu,
            refs_nchw=refs_cpu,
            ref_t_offsets=ref_t_offsets,
            text_emb=text_cpu,
            timestep_0_to_1=t_01,
        )

        out_t = out_cpu.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(0)
        return pipe._pack_latents(out_t)

    _denoiser._bridge = bridge
    return _denoiser

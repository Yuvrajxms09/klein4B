from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


class FluxImage(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("channels", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_uint8)),
    ]


class FluxParams(ctypes.Structure):
    _fields_ = [
        ("width", ctypes.c_int),
        ("height", ctypes.c_int),
        ("num_steps", ctypes.c_int),
        ("seed", ctypes.c_int64),
        ("guidance", ctypes.c_float),
        ("linear_schedule", ctypes.c_int),
        ("power_schedule", ctypes.c_int),
        ("power_alpha", ctypes.c_float),
    ]


@dataclass
class KleinGenerateConfig:
    width: int
    height: int
    num_steps: int = 4
    seed: int = -1
    guidance: float = 1.0
    linear_schedule: bool = False
    power_schedule: bool = False
    power_alpha: float = 2.0

    def to_c(self) -> FluxParams:
        return FluxParams(
            width=int(self.width),
            height=int(self.height),
            num_steps=int(self.num_steps),
            seed=int(self.seed),
            guidance=float(self.guidance),
            linear_schedule=1 if self.linear_schedule else 0,
            power_schedule=1 if self.power_schedule else 0,
            power_alpha=float(self.power_alpha),
        )


class KleinCFullBackend:
    """
    Full native klein-cuda-c backend (denoising + sampling in C/CUDA runtime).
    """

    def __init__(self, model_dir: str, lib_path: str | None = None, use_mmap: bool = True) -> None:
        if lib_path is None:
            lib_path = str(Path(__file__).with_name("klein_c_bridge") / "libklein_bridge.so")
        self._lib = ctypes.CDLL(str(lib_path))
        self._ctx = self._init_ctx(model_dir=model_dir, use_mmap=use_mmap)

    def _init_ctx(self, model_dir: str, use_mmap: bool) -> ctypes.c_void_p:
        self._lib.flux_cuda_init.argtypes = []
        self._lib.flux_cuda_init.restype = ctypes.c_int
        self._lib.flux_cuda_available.argtypes = []
        self._lib.flux_cuda_available.restype = ctypes.c_int
        self._lib.flux_load_dir.argtypes = [ctypes.c_char_p]
        self._lib.flux_load_dir.restype = ctypes.c_void_p
        self._lib.flux_free.argtypes = [ctypes.c_void_p]
        self._lib.flux_free.restype = None
        self._lib.flux_set_mmap.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self._lib.flux_set_mmap.restype = None
        self._lib.flux_get_error.argtypes = []
        self._lib.flux_get_error.restype = ctypes.c_char_p

        self._lib.flux_img2img.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(FluxImage),
            ctypes.POINTER(FluxParams),
        ]
        self._lib.flux_img2img.restype = ctypes.POINTER(FluxImage)

        self._lib.flux_multiref.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.POINTER(FluxImage)),
            ctypes.c_int,
            ctypes.POINTER(FluxParams),
        ]
        self._lib.flux_multiref.restype = ctypes.POINTER(FluxImage)

        self._lib.flux_image_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
        self._lib.flux_image_create.restype = ctypes.POINTER(FluxImage)
        self._lib.flux_image_free.argtypes = [ctypes.POINTER(FluxImage)]
        self._lib.flux_image_free.restype = None

        cuda_ok = int(self._lib.flux_cuda_init())
        if cuda_ok != 1 or int(self._lib.flux_cuda_available()) != 1:
            raise RuntimeError("CUDA backend not available in loaded klein-cuda-c library")

        ctx = self._lib.flux_load_dir(model_dir.encode("utf-8"))
        if not ctx:
            msg = self.last_error() or "flux_load_dir failed"
            raise RuntimeError(msg)
        self._lib.flux_set_mmap(ctx, 1 if use_mmap else 0)
        return ctx

    def close(self) -> None:
        if getattr(self, "_ctx", None):
            self._lib.flux_free(self._ctx)
            self._ctx = None

    def __del__(self) -> None:
        self.close()

    def last_error(self) -> str:
        msg = self._lib.flux_get_error()
        if not msg:
            return ""
        return msg.decode("utf-8", errors="ignore")

    def _image_to_flux(self, img: Image.Image) -> ctypes.POINTER(FluxImage):
        rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        h, w, c = arr.shape
        if c != 3:
            raise ValueError("Expected RGB image")
        flux_img = self._lib.flux_image_create(int(w), int(h), 3)
        if not flux_img:
            raise RuntimeError("flux_image_create failed")
        n = int(w) * int(h) * 3
        dst = np.ctypeslib.as_array(flux_img.contents.data, shape=(n,))
        dst[:] = arr.reshape(-1)
        return flux_img

    def _flux_to_image(self, flux_img: ctypes.POINTER(FluxImage)) -> Image.Image:
        if not flux_img:
            raise RuntimeError("flux image pointer is null")
        w = int(flux_img.contents.width)
        h = int(flux_img.contents.height)
        c = int(flux_img.contents.channels)
        if c not in (3, 4):
            raise RuntimeError(f"Unsupported channels={c}")
        n = w * h * c
        src = np.ctypeslib.as_array(flux_img.contents.data, shape=(n,))
        arr = src.reshape((h, w, c)).copy()
        if c == 4:
            return Image.fromarray(arr, mode="RGBA")
        return Image.fromarray(arr, mode="RGB")

    def img2img(
        self,
        prompt: str,
        image: Image.Image,
        config: KleinGenerateConfig,
    ) -> Image.Image:
        in_img = self._image_to_flux(image)
        out_img = None
        try:
            params = config.to_c()
            out_img = self._lib.flux_img2img(
                self._ctx,
                prompt.encode("utf-8"),
                in_img,
                ctypes.byref(params),
            )
            if not out_img:
                raise RuntimeError(self.last_error() or "flux_img2img failed")
            return self._flux_to_image(out_img)
        finally:
            if out_img:
                self._lib.flux_image_free(out_img)
            if in_img:
                self._lib.flux_image_free(in_img)

    def multiref(
        self,
        prompt: str,
        refs: list[Image.Image],
        config: KleinGenerateConfig,
    ) -> Image.Image:
        if not refs:
            raise ValueError("refs must not be empty")
        if len(refs) > 4:
            raise ValueError("klein-cuda-c supports up to 4 reference images")

        ref_ptrs: list[ctypes.POINTER(FluxImage)] = []
        out_img = None
        try:
            for img in refs:
                ref_ptrs.append(self._image_to_flux(img))
            arr_type = ctypes.POINTER(FluxImage) * len(ref_ptrs)
            refs_arr = arr_type(*ref_ptrs)
            params = config.to_c()
            out_img = self._lib.flux_multiref(
                self._ctx,
                prompt.encode("utf-8"),
                refs_arr,
                ctypes.c_int(len(ref_ptrs)),
                ctypes.byref(params),
            )
            if not out_img:
                raise RuntimeError(self.last_error() or "flux_multiref failed")
            return self._flux_to_image(out_img)
        finally:
            if out_img:
                self._lib.flux_image_free(out_img)
            for p in ref_ptrs:
                self._lib.flux_image_free(p)

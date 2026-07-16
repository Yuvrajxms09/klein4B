"""Modal benchmark using pre-quantized FLUX.2 Klein 4B NVFP4 weights.

The transformer and reduced 27-layer Qwen encoder are loaded from the pinned
Hugging Face artifact below. Their weights are already packed as NVFP4, while
TorchAO still quantizes request-dependent activations at inference time. The
remaining inference configuration matches ``fastest_script_115ms.py``.

Weights: https://huggingface.co/Yuvrajxms09/klein-torchao-artifacts
Run: ``modal run fastest_script_prequantized_nvfp4.py``
"""


from __future__ import annotations

import os
import re
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import modal


APP = modal.App("klein4b-inference-bench-prequantized-nvfp4")
HF_SECRET = modal.Secret.from_name("huggingface-secret")


def _workspace_root() -> Path:
    return Path(__file__).resolve().parent


image = (
    modal.Image.from_registry("nvidia/cuda:12.4.1-devel-ubuntu22.04", add_python="3.11")
    .apt_install("git", "libopenblas-dev")
    .env({"TORCH_CUDA_ARCH_LIST": "12.0", "USE_HUB_KERNELS": "NO"})
    .run_commands(
        "pip install --index-url https://download.pytorch.org/whl/cu130 "
        "torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0"
    )
    .pip_install("numpy")
    .pip_install("pillow")
    .pip_install("fastapi")
    .pip_install("uvicorn")
    .pip_install("accelerate")
    .pip_install("safetensors")
    .pip_install("huggingface_hub")
    .pip_install("importlib-metadata", "filelock", "httpx", "regex", "requests")
    .pip_install("einops")
    .pip_install("pybase64")
    .pip_install("cache-dit")
    .pip_install("kernels==0.16.0")
    .pip_install("ninja", "setuptools", "wheel")
    .run_commands(
        "pip install -U git+https://github.com/huggingface/transformers.git@"
        "63f32a8782cb70da3365acab16f2b67947737985"
    )
    .run_commands("pip install torchao==0.17.0")
    .run_commands("pip install mslk-cuda==1.1.0")
    .add_local_dir(
        str(_workspace_root()),
        remote_path="/root/klein4B",
        copy=True,
    )
    .add_local_dir(
        str(_workspace_root().parent / "diffusers"),
        remote_path="/root/diffusers",
        copy=True,
    )
)


MODEL_VOLUME = modal.Volume.from_name("klein4B-assets", create_if_missing=False)
VOLUME_MOUNT = "/mnt/klein4B-assets"
PREQUANTIZED_REPO_ID = "Yuvrajxms09/klein-torchao-artifacts"
PREQUANTIZED_REVISION = "7fd5321202831de6c833bd55f318c2f5e92a1bdd"

DEFAULT_BENCHMARK_PROMPTS: tuple[str, ...] = (
    "from top angle",
    "on a bike",
    "on a mountain road",
    "in a rainy city at night",
    "under golden sunset light",
    "in a snowy forest",
    "with a cinematic teal and orange look",
    "as a watercolor painting",
    "as a realistic product photo",
    "with dramatic studio lighting",
    "in a futuristic cyberpunk street",
    "as a vintage film photograph",
    "with soft pastel colors",
    "as a high-detail oil painting",
    "in a desert at noon",
    "with neon reflections on wet pavement",
    "as a comic book illustration",
    "with a minimalist background",
    "in a lush tropical jungle",
    "as a high contrast black and white photo",
    "from top angle",
    "on a bike",
    "on a mountain road",
    "in a rainy city at night",
)


def _require_nvfp4_linears(module, *, expected: int, component: str) -> int:
    import torch
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

    count = sum(
        1
        for child in module.modules()
        if isinstance(child, torch.nn.Linear)
        and isinstance(child.weight, NVFP4Tensor)
    )
    if count != expected:
        raise RuntimeError(
            f"{component} is not fully quantized to NVFP4: expected {expected} "
            f"linear layers, found {count}"
        )

    invalid_activation_configs = []
    for name, child in module.named_modules():
        weight = getattr(child, "weight", None)
        if weight is None or not isinstance(weight, NVFP4Tensor):
            continue
        activation_config = getattr(weight, "act_quant_kwargs", None)
        if activation_config is None or not getattr(
            activation_config,
            "use_dynamic_per_tensor_scale",
            False,
        ):
            invalid_activation_configs.append(name)
    if invalid_activation_configs:
        raise RuntimeError(
            f"{component} contains NVFP4 weights without dynamic activation "
            f"quantization: {invalid_activation_configs[:5]}"
        )
    return count


def _require_prequantized_loader(model, *, component: str) -> None:
    quantizer = getattr(model, "hf_quantizer", None)
    if quantizer is None or not getattr(quantizer, "pre_quantized", False):
        raise RuntimeError(
            f"{component} was not loaded as a pre-quantized checkpoint. "
            "Refusing to benchmark a runtime-quantized fallback."
        )


def _load_prequantized_components(
    *,
    repo_id: str,
    revision: str,
    dtype,
    token: str | None,
):
    import torchao.prototype.mx_formats  # noqa: F401
    from diffusers import Flux2Transformer2DModel
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM

    manifest_path = hf_hub_download(
        repo_id=repo_id,
        filename="manifest.json",
        revision=revision,
        token=token,
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("quantization", {}).get("format") != "torchao-nvfp4":
        raise RuntimeError(f"Unexpected artifact format in {repo_id}@{revision}")

    transformer = Flux2Transformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer",
        revision=revision,
        token=token,
        torch_dtype=dtype,
        local_files_only=False,
        use_safetensors=False,
    )
    text_encoder = AutoModelForCausalLM.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        revision=revision,
        token=token,
        torch_dtype=dtype,
        local_files_only=False,
    )

    _require_prequantized_loader(transformer, component="Transformer")
    _require_prequantized_loader(text_encoder, component="Qwen text encoder")
    transformer_count = _require_nvfp4_linears(
        transformer,
        expected=109,
        component="Transformer",
    )

    decoder = text_encoder.model
    required_layer_count = 27
    if len(decoder.layers) != required_layer_count:
        raise RuntimeError(
            f"Pre-quantized Qwen artifact must contain exactly "
            f"{required_layer_count} layers, found {len(decoder.layers)}"
        )
    text_encoder_count = _require_nvfp4_linears(
        decoder,
        expected=required_layer_count * 7,
        component="Qwen decoder",
    )

    report = {
        "artifact_repo_id": repo_id,
        "artifact_revision": revision,
        "weights_loaded_prequantized": True,
        "transformer_nvfp4_linear_count": transformer_count,
        "text_encoder_nvfp4_linear_count": text_encoder_count,
        "text_encoder_decoder_layers": required_layer_count,
    }
    return transformer, text_encoder, report


def _safe_prompt_slug(prompt: str, max_len: int = 48) -> str:
    slug = re.sub(r"[^\w\s-]", "", prompt.lower())
    slug = re.sub(r"[-\s]+", "-", slug).strip("-")[:max_len]
    return slug or "prompt"


def _save_benchmark_images_after_timing(
    *,
    images: list,
    prompts: list[str],
    base_dir: Path,
    volume: modal.Volume,
    volume_mount: str,
    benchmark_result: dict | None = None,
) -> Path:
    base_dir = base_dir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)
    run_dir = base_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    for i, (prompt, img) in enumerate(zip(prompts, images), start=1):
        slug = _safe_prompt_slug(prompt)
        out_path = run_dir / f"{i:02d}_{slug}.png"
        img.save(out_path, format="PNG")

    lines = [f"{i:02d}\t{p}" for i, p in enumerate(prompts, start=1)]
    (run_dir / "prompts.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    if benchmark_result is not None:
        (run_dir / "benchmark.json").write_text(
            json.dumps(benchmark_result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    if run_dir.resolve().is_relative_to(Path(volume_mount).resolve()):
        volume.commit()
    return run_dir


@APP.function(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60,
    volumes={VOLUME_MOUNT: MODEL_VOLUME},
    secrets=[HF_SECRET],
)
def benchmark(
    *,
    model_dir: str = "/mnt/klein4B-assets/FLUX.2-klein-4B",
    prequantized_repo_id: str = PREQUANTIZED_REPO_ID,
    prequantized_revision: str = PREQUANTIZED_REVISION,
    image_path: str = "/mnt/klein4B-assets/calib/blue_car.jpeg",
    height: int = 576,
    width: int = 384,
    num_inference_steps: int = 4,
    guidance_scale: float = 1.0,
    generator_seed: int = 0,
    warmup_runs: int = 5,
    prompts: list[str] | None = None,
    taef2_cache_dir: str = "/root/klein4B/.cache/taef2",
    save_outputs_dir: str | None = "/mnt/klein4B-assets/bench_outputs_nvfp4",
    local_files_only: bool = True,
) -> dict:
    import torch
    from PIL import Image

    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    repo_root = Path("/root/klein4B")
    os.chdir(repo_root)
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    diffusers_root = Path("/root/diffusers/src")
    if str(diffusers_root) not in sys.path:
        sys.path.insert(0, str(diffusers_root))

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    from klein_pipeline import Flux2KleinPipeline
    from taef2_vae import replace_pipeline_vae_with_taef2

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model_dir not found: {model_dir}")
    if not Path(image_path).exists():
        raise FileNotFoundError(f"image_path not found: {image_path}")

    prompt_list = list(prompts) if prompts is not None else list(DEFAULT_BENCHMARK_PROMPTS)
    if len(prompt_list) != 24 or len(set(prompt_list)) != 20:
        raise ValueError("benchmark requires 20 unique prompts followed by four repeats")
    input_image = Image.open(image_path).convert("RGB").resize((width, height))

    dtype = torch.bfloat16
    transformer, text_encoder, prequantized_report = _load_prequantized_components(
        repo_id=prequantized_repo_id,
        revision=prequantized_revision,
        dtype=dtype,
        token=os.environ.get("HF_TOKEN"),
    )

    pipe = Flux2KleinPipeline.from_pretrained(
        model_dir,
        transformer=transformer,
        text_encoder=text_encoder,
        torch_dtype=dtype,
        local_files_only=local_files_only,
    )
    pipe.set_progress_bar_config(disable=True)
    pipe = pipe.to("cuda")

    pipe.text_encoder.model = torch.compile(
        pipe.text_encoder.model,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
    )
    replace_pipeline_vae_with_taef2(pipe, cache_dir=taef2_cache_dir)
    if hasattr(pipe.vae, "taesd") and hasattr(pipe.vae.taesd, "decoder"):
        pipe.vae.taesd.decoder.to(memory_format=torch.channels_last)

    pipe.vae.to(memory_format=torch.channels_last)
    vae_constant_ref = torch.empty(
        (),
        device=pipe.vae.bn.running_mean.device,
        dtype=pipe.vae.dtype,
    )
    pipe._get_vae_bn_constants(vae_constant_ref)

    pipe.transformer.set_attention_backend("native")

    torch._inductor.config.conv_1x1_as_mm = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.coordinate_descent_check_all_directions = True
    torch._inductor.config.epilogue_fusion = False

    pipe.transformer = torch.compile(
        pipe.transformer,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
    )

    def _vae_encode_fn(image: torch.Tensor, generator: torch.Generator):
        return pipe._encode_vae_image(image=image, generator=generator)

    pipe._vae_encode_fn = torch.compile(
        _vae_encode_fn,
        mode="max-autotune-no-cudagraphs",
        fullgraph=False,
        dynamic=False,
    )

    def _vae_decode_fn(latents: torch.Tensor):
        return pipe.vae.decode(latents, return_dict=False)[0]

    pipe._vae_decode_fn = torch.compile(
        _vae_decode_fn,
        mode="max-autotune",
        fullgraph=False,
        dynamic=False,
    )

    # CPU preprocessing is excluded; TAEF2 image encoding remains timed.
    prepared_image = pipe.image_processor.preprocess(
        input_image,
        height=height,
        width=width,
        resize_mode="crop",
    )
    if prepared_image.device.type == "cpu" and not prepared_image.is_pinned():
        prepared_image = prepared_image.pin_memory()

    def call_pipeline(prompt: str, *, callback_on_step_end=None):
        return pipe(
            prompt=prompt,
            image=prepared_image,
            height=height,
            width=width,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=torch.Generator(device="cuda").manual_seed(generator_seed),
            output_type="pil",
            callback_on_step_end=callback_on_step_end,
        ).images

    validated_steps: list[int] = []

    def record_step(_pipe, step, _timestep, callback_kwargs):
        validated_steps.append(int(step))
        return callback_kwargs

    with torch.inference_mode():
        call_pipeline("four-step validation prompt", callback_on_step_end=record_step)
    torch.cuda.synchronize()
    if validated_steps != list(range(num_inference_steps)):
        raise RuntimeError(
            f"Expected {num_inference_steps} denoising forwards, "
            f"observed steps {validated_steps}"
        )

    warmup_times: list[float] = []
    for i in range(warmup_runs):
        warmup_prompt = f"unique compilation warmup prompt {i + 1}"
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            call_pipeline(warmup_prompt)
        torch.cuda.synchronize()
        dt_ms = (time.perf_counter() - t0) * 1000.0
        warmup_times.append(dt_ms)
        print(f"Warmup {i + 1}: {dt_ms:.1f} ms")

    print(f"Warmup avg: {sum(warmup_times) / len(warmup_times):.1f} ms")

    wall_times: list[float] = []
    cuda_times: list[float] = []
    outputs = []
    for i, prompt in enumerate(prompt_list):
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        start_event.record()
        with torch.inference_mode():
            output = call_pipeline(prompt)
        end_event.record()
        end_event.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0
        cuda_ms = start_event.elapsed_time(end_event)
        wall_times.append(wall_ms)
        cuda_times.append(cuda_ms)
        outputs.append(output[0])
        print(
            f"[{i + 1:02d}/{len(prompt_list):02d}] "
            f"wall={wall_ms:.3f} ms cuda={cuda_ms:.3f} ms | {prompt}"
        )

    def percentile(values: list[float], p: float) -> float:
        ordered = sorted(values)
        index = (len(ordered) - 1) * p
        lo = int(index)
        hi = min(lo + 1, len(ordered) - 1)
        fraction = index - lo
        return ordered[lo] * (1 - fraction) + ordered[hi] * fraction

    result = {
        "backend": "torchao-nvfp4",
        "resolution": [height, width],
        "num_inference_steps": num_inference_steps,
        "validated_denoising_steps": validated_steps,
        "guidance_scale": guidance_scale,
        "output_type": "pil",
        "vae": "taef2",
        "text_encoder_backend": "torchao-nvfp4",
        "text_encoder_compile": "max-autotune",
        "prequantized_artifact": prequantized_report,
        "text_encoder": {
            "executed_decoder_layers": 27,
            "nvfp4_linear_count": prequantized_report["text_encoder_nvfp4_linear_count"],
        },
        "warmup_ms": warmup_times,
        "wall_ms": wall_times,
        "cuda_event_ms": cuda_times,
        "wall_summary_ms": {
            "mean": statistics.fmean(wall_times),
            "median": statistics.median(wall_times),
            "p95": percentile(wall_times, 0.95),
            "min": min(wall_times),
        },
        "cuda_summary_ms": {
            "mean": statistics.fmean(cuda_times),
            "median": statistics.median(cuda_times),
            "p95": percentile(cuda_times, 0.95),
            "min": min(cuda_times),
        },
        "attention": "native",
        "transformer_compile": "max-autotune",
        "input_preprocessed_outside_timing": True,
        "prompt_embeds_precomputed_outside_timing": False,
        "noise_precomputed_outside_timing": False,
        "prompt_cache_enabled": True,
        "measured_prompt_count": len(prompt_list),
        "unique_prompt_count": len(set(prompt_list)),
        "expected_full_compute_nvfp4_gemms": 436,
    }
    print(json.dumps(result, indent=2, sort_keys=True))

    if save_outputs_dir:
        run_dir = _save_benchmark_images_after_timing(
            images=outputs,
            prompts=prompt_list,
            base_dir=Path(save_outputs_dir),
            volume=MODEL_VOLUME,
            volume_mount=VOLUME_MOUNT,
            benchmark_result=result,
        )
        print(f"Saved {len(outputs)} images and benchmark.json to {run_dir} (post-timed)")
    return result


@APP.local_entrypoint()
def main(
    height: int = 576,
    width: int = 384,
    image_path: str = "/mnt/klein4B-assets/calib/blue_car.jpeg",
) -> None:
    benchmark.remote(
        height=height,
        width=width,
        image_path=image_path,
    )

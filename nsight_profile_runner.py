from __future__ import annotations

import json

from modal_bench_inference_nvfp4 import benchmark


def main() -> None:
    result = benchmark.get_raw_f()(
        height=576,
        width=384,
        num_inference_steps=4,
        guidance_scale=1.0,
        warmup_runs=5,
        measured_runs=1,
        save_outputs_dir=None,
        optimization_profile="one-shot-exact",
        nsight_capture=True,
        enable_internal_profiler_validation=False,
    )

    contract = result["execution_contract"]
    expected = {
        "scheduler_steps": 4,
        "full_transformer_steps": 4,
        "reused_transformer_steps": 0,
        "prediction_reuse_mode": None,
    }
    observed = {key: contract[key] for key in expected}
    if observed != expected:
        raise RuntimeError(f"Nsight workload contract mismatch: expected={expected} observed={observed}")
    if result["resolution"] != [576, 384] or result["output_type"] != "pil":
        raise RuntimeError(
            "Nsight workload output contract mismatch: "
            f"resolution={result['resolution']} output_type={result['output_type']}"
        )
    if result["text_encoder_backend"] != "torchao-nvfp4":
        raise RuntimeError(f"Expected NVFP4 text encoder, got {result['text_encoder_backend']!r}")
    if result["backend"] != "torchao-nvfp4" or result["vae"] != "taef2":
        raise RuntimeError(
            "Expected TorchAO NVFP4 transformer and TAEF2: "
            f"backend={result['backend']!r} vae={result['vae']!r}"
        )
    if not result["text_encoder_compile"]:
        raise RuntimeError("Expected max-autotune compiled NVFP4 text encoder")
    if not result["transformer_compile"] or result["transformer_compile_mode"] != "max-autotune":
        raise RuntimeError(
            "Expected max-autotune compiled transformer: "
            f"compiled={result['transformer_compile']} mode={result['transformer_compile_mode']!r}"
        )
    if result["expected_executed_nvfp4_gemms"] != 356:
        raise RuntimeError(
            f"Expected 356 full-compute NVFP4 GEMMs, got {result['expected_executed_nvfp4_gemms']}"
        )
    if result["expected_attention_invocations"] != 100:
        raise RuntimeError(
            f"Expected 100 exact-attention invocations, got {result['expected_attention_invocations']}"
        )
    config = result["effective_optimization_config"]
    if config["enable_cache"] or config["enable_denoiser_step_reuse"]:
        raise RuntimeError(f"Approximate execution was enabled unexpectedly: {config}")
    print("NSIGHT_WORKLOAD_VALIDATED " + json.dumps(observed, sort_keys=True))


if __name__ == "__main__":
    main()

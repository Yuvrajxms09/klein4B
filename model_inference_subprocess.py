from __future__ import annotations

from copy import deepcopy
from multiprocessing import Manager, Process, Value
from queue import Empty
import logging
import time
import traceback

import numpy as np
import torch
from PIL import Image

from cache_dit_klein import enable_cache_dit, enable_temporal_consistency, prepare_transformer_for_speed
from interpolation import expand_batch_with_rife, load_rife_ifnet
from klein_pipeline import Flux2KleinPipeline
from temporal_consistency import TemporalConsistencyConfig, TemporalConsistencyController
from utils.shared_tensor import SharedTensor

logger = logging.getLogger(__name__)


class ModelInferenceSubprocess:
    def __init__(
        self,
        config: dict,
        input_shared_tensor_name: str,
        output_batch_shared_tensor_name: str,
        pack_is_ready,
        last_processing_time,
    ):
        self.running = Value("b", False)
        self.process = None
        self.config = config
        self.height = self.config["resolution"]["height"]
        self.width = self.config["resolution"]["width"]
        self.resolution = self.config["resolution"]
        self.prompt = self.config["default_prompt"]
        self.logging = self.config.get("logging", True)
        self.input_shared_tensor_name = input_shared_tensor_name
        self.output_batch_shared_tensor_name = output_batch_shared_tensor_name
        self.pack_is_ready = pack_is_ready
        self.last_processing_time = last_processing_time

        manager = Manager()
        self.command_queue = manager.Queue()
        self.shared_state = manager.dict()
        self.worker_error = manager.dict(message=None, traceback=None)
        self.interpolation_exp = self.config.get("interpolation_exp", 1)

    def set_status(self, phase: str, message: str | None = None) -> None:
        self.shared_state["phase"] = phase
        if message is not None:
            self.shared_state["message"] = message
            if self.logging:
                logger.info("[%s] %s", phase, message)
        elif self.logging:
            logger.info("[%s]", phase)

    def init_process_state(self):
        self.device = "cuda"
        self.process_state = {
            "prompt": self.config["default_prompt"],
            "steps": self.config["default_steps"],
            "seed": self.config["default_seed"],
        }

    def load_models(self):
        local_files_only = self.config.get("local_files_only", False)
        self.set_status(
            "loading_models",
            f"loading pipe from {self.config['model_dir']} (local_files_only={local_files_only}, compile={self.config.get('compile_models', False)}, dynamic={self.config.get('compile_dynamic', False)}, interpolate={self.config.get('interpolate', False)})",
        )
        self.pipe = Flux2KleinPipeline.from_pretrained(
            self.config["model_dir"],
            torch_dtype=torch.bfloat16,
            local_files_only=local_files_only,
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

        enable_cache_dit(self.pipe)
        prepare_transformer_for_speed(self.pipe, backend="auto", fuse_qkv=True)
        self.spatial_cache = enable_temporal_consistency(
            self.pipe,
            height=self.height,
            width=self.width,
        )

        if self.config.get("compile_models", False):
            self.pipe.enable_compile(dynamic=self.config.get("compile_dynamic", False))

        self.rife_model = None
        if self.config.get("interpolate", False) and self.interpolation_exp > 0:
            self.set_status("loading_rife", f"loading RIFE weights from {self.config['rife_weights_path']}")
            self.rife_model = load_rife_ifnet(
                self.config["rife_weights_path"],
                device="cuda",
                dtype=torch.float16,
            )
            if self.config.get("compile_models", False):
                self.rife_model = torch.compile(self.rife_model)

        self.update_controller = TemporalConsistencyController(
            TemporalConsistencyConfig(
                height=self.height,
                width=self.width,
                device="cuda",
                dtype=torch.bfloat16,
            )
        )
        self._expand_batch_with_rife = expand_batch_with_rife
        self._pil_to_tensor = lambda img: torch.from_numpy(
            np.asarray(img, dtype=np.uint8)
        ).permute(2, 0, 1).unsqueeze(0).to(torch.float16).div_(255.0)
        self._tensor_to_numpy = lambda tensor: (
            tensor.detach()
            .clamp(0, 1)
            .mul(255)
            .to(torch.uint8)
            .cpu()
            .numpy()
        )

    def update_prompt_embeds(self, prompt):
        self.prompt_embeds, self.text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        self.update_controller.reset_cache()

    def init_shared_tensors(self):
        h, w = self.resolution["height"], self.resolution["width"]

        self.input_shared_tensor = SharedTensor(
            (h, w, 3),
            name=self.input_shared_tensor_name,
        )

        output_batch_size = 2**self.interpolation_exp
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, h, w, 3),
            name=self.output_batch_shared_tensor_name,
        )

    def process_init(self):
        self.set_status("starting", "initializing shared tensors and models")
        self.init_process_state()
        self.init_shared_tensors()
        self.load_models()
        self.set_status("encoding_prompt", f"encoding prompt {self.process_state['prompt']!r}")
        self.update_prompt_embeds(self.process_state["prompt"])
        self.previous_frame = None

        if self.config.get("use_reference_image", False):
            image = self.config.get("reference_image_array")
            resolution = self.config.get("reference_image_resolution")
            if image is None:
                image = np.zeros(
                    (resolution["height"], resolution["width"], 3), dtype=np.uint8
                )
            else:
                image = np.asarray(image, dtype=np.uint8)
                if image.shape[:2] != (resolution["height"], resolution["width"]):
                    from PIL import Image as PILImage

                    image = np.asarray(
                        PILImage.fromarray(image).resize(
                            (resolution["width"], resolution["height"]),
                            resample=PILImage.Resampling.BILINEAR,
                        )
                    )
            self.reference_image = Image.fromarray(image)

        target_fps = self.config.get("target_fps", None)
        self.target_base_processing_time = None
        if target_fps is not None:
            target_base_fps = target_fps / (2**self.interpolation_exp)
            self.target_base_processing_time = 1 / target_base_fps
        self.set_status(
            "ready",
            f"ready at {self.width}x{self.height}, steps={self.process_state['steps']}, target_fps={target_fps}",
        )

    def start(self):
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()
        self.shared_state["pid"] = self.process.pid
        self.shared_state["alive"] = True
        if self.logging:
            logger.info("Model worker process started pid=%s", self.process.pid)

    def stop(self):
        self.running.value = False
        if self.process:
            self.process.join()

    def set_param(self, name: str, value) -> None:
        self.command_queue.put(("set_param", (name, value)))

    def set_reference_image(self, image: np.ndarray | None) -> None:
        if not self.config.get("use_reference_image", False):
            raise ValueError(
                "set_reference_image called but use_reference_image is not enabled in the stream processor config"
            )
        self.command_queue.put(("set_reference_image", image))

    def set_mask(self, mask) -> None:
        if self.config.get("mask_calculation_method", "auto") != "manual":
            raise ValueError(
                "set_mask called but mask_calculation_method is not set to manual in the config"
            )
        self.command_queue.put(("set_mask", mask))

    def update_process_state(self) -> None:
        try:
            while True:
                cmd, payload = self.command_queue.get_nowait()
                if cmd == "set_param":
                    name, value = payload
                    self.process_state[name] = value
                    if name == "prompt":
                        self.update_prompt_embeds(value)
                elif cmd == "set_reference_image":
                    image = payload
                    resolution = self.config["reference_image_resolution"]
                    if image is not None:
                        image = np.asarray(image, dtype=np.uint8)
                        if image.shape[:2] != (resolution["height"], resolution["width"]):
                            from PIL import Image as PILImage

                            image = np.asarray(
                                PILImage.fromarray(image).resize(
                                    (resolution["width"], resolution["height"]),
                                    resample=PILImage.Resampling.BILINEAR,
                                )
                            )
                        self.reference_image = Image.fromarray(image)
                    else:
                        self.reference_image = Image.fromarray(
                            np.zeros(
                                (resolution["height"], resolution["width"], 3),
                                dtype=np.uint8,
                            )
                        )
                    self.update_controller.reset_cache()
                elif cmd == "set_mask":
                    mask = payload
                    mask_tensor = (
                        torch.from_numpy(mask)
                        .unsqueeze(0)
                        .to(self.update_controller.device)
                    )
                    self.update_controller.set_mask(mask_tensor)
        except Empty:
            pass

    def get_input_frame(self):
        frame = self.input_shared_tensor.to_numpy()
        frame = frame[..., ::-1].copy()
        frame_gpu = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame_gpu

    def interpolate_frames(self, frame):
        if self.previous_frame is None:
            self.previous_frame = frame

        if self.interpolation_exp == 0 or self.rife_model is None:
            frames_out = frame
        else:
            frames = torch.cat([self.previous_frame, frame], dim=0)
            with torch.no_grad():
                for _ in range(self.interpolation_exp):
                    b = frames.size(0)
                    prevs = frames[:-1]
                    nexts = frames[1:]
                    mids = self.rife_model(torch.cat([prevs, nexts], dim=1))
                    h, w = frames.shape[2:]
                    new_frames = torch.empty(
                        2 * b - 1, 3, h, w, device=frames.device, dtype=frames.dtype
                    )
                    new_frames[0::2] = frames
                    new_frames[1::2] = mids
                    frames = new_frames
            frames_out = frames[1:]

        frames_cpu = self._tensor_to_numpy(frames_out)
        if frames_cpu.ndim == 4 and frames_cpu.shape[1] == 3:
            frames_cpu = np.transpose(frames_cpu, (0, 2, 3, 1))
        elif frames_cpu.ndim == 3 and frames_cpu.shape[0] == 3:
            frames_cpu = np.transpose(frames_cpu, (1, 2, 0))
            frames_cpu = frames_cpu[None, ...]
        else:
            raise RuntimeError(
                f"Unexpected interpolated frame shape {frames_cpu.shape}; expected (B, 3, H, W) or (3, H, W)"
            )
        self.previous_frame = frame
        return frames_cpu[..., ::-1].copy()

    def send_frames(self, frames):
        self.output_batch_shared_tensor.copy_from(frames)

    def sync_fps_and_send(self, prev_time, frames):
        now = time.time()
        processing_time = now - prev_time

        if self.target_base_processing_time is not None:
            sleep_time = max(self.target_base_processing_time - processing_time, 0.0)
            time.sleep(sleep_time)
            now = time.time()

        processing_time = now - prev_time
        self.last_processing_time.value = processing_time
        self.send_frames(frames)
        self.pack_is_ready.value = True

        if self.logging:
            print(
                f"base fps: {(1 / processing_time):.2f}, interpolated fps: {(1 / processing_time * 2**self.interpolation_exp):.2f}"
            )
        return now

    def process_frame_with_pipeline(self, frame):
        input_frame = Image.fromarray(frame)

        reference_list = [input_frame]
        if self.config.get("use_reference_image", False):
            reference_list.append(self.reference_image)

        out = self.pipe(
            prompt_embeds=self.prompt_embeds,
            image=reference_list,
            height=self.resolution["height"],
            width=self.resolution["width"],
            guidance_scale=1.0,
            num_inference_steps=self.process_state["steps"],
            num_images_per_prompt=1,
            generator=torch.Generator(device=self.device).manual_seed(
                self.process_state["seed"]
            ),
            output_type="np",
        )
        out_image = out.images[0] * 255
        out_image = out_image.astype(np.uint8)
        return out_image

    def convert_np_to_torch(self, frame):
        frame = (
            torch.from_numpy(frame)
            .to(self.device)
            .to(torch.float16)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div(255)
        )
        return frame

    def process_main(self):
        try:
            self.process_init()
            prev_time = time.time()
            self.set_status("running", "entering inference loop")
            while self.running.value:
                self.update_process_state()
                frame = self.input_shared_tensor.to_numpy()
                frame = frame[..., ::-1].copy()
                self.set_status("infer", "running pipeline on next webcam frame")
                frame = self.process_frame_with_pipeline(frame)
                frame = self.convert_np_to_torch(frame)
                frames = self.interpolate_frames(frame)
                self.set_status("publish", f"publishing batch of {frames.shape[0]} frame(s)")
                prev_time = self.sync_fps_and_send(prev_time, frames)
        except Exception as exc:
            logger.exception("Model worker failed")
            self.worker_error["message"] = repr(exc)
            self.worker_error["traceback"] = traceback.format_exc()
            self.shared_state["alive"] = False
            self.shared_state["phase"] = "error"
            self.shared_state["message"] = repr(exc)
            self.running.value = False
            self.pack_is_ready.value = False
            raise

    def has_error(self) -> bool:
        return self.worker_error.get("message") is not None

    def get_error(self) -> dict:
        return {
            "message": self.worker_error.get("message"),
            "traceback": self.worker_error.get("traceback"),
        }

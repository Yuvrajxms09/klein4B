import torch
import time
import cv2
import numpy as np
from safetensors.torch import load_file
from multiprocessing import Process, Value, Manager
from queue import Empty
from PIL import Image

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.models import AutoencoderKLFlux2
from transformers import Qwen2TokenizerFast, Qwen3ForCausalLM

from fluxrt.stream_processor.interpolation_model import IFNet
from fluxrt.stream_processor.transformer_flux2 import Flux2Transformer2DModel
from fluxrt.utils.shared_tensor import SharedTensor
from fluxrt.stream_processor.pipeline import Flux2KleinPipeline
from fluxrt.stream_processor.update_controller import UpdateController
from fluxrt.stream_processor.postprocessors import (
    BasePostProcessor,
    LivePortraitPostProcessor,
)
from fluxrt.flow_upscaler.upscaler_unet import UpscalerUNet
from fluxrt.flow_upscaler.flow_upscaler_pipeline import FlowUpscalerPipeline
from fluxrt.stream_processor.flux_tiny_vae import DiffusersTAEF2Wrapper


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
        self.memory_reserved = Value("i", 0)
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
        self.interpolation_exp = self.config.get("interpolation_exp", 1)
        self.debug_tracing = bool(self.config.get("debug_tracing", False))
        self.debug_trace_every_n = max(1, int(self.config.get("debug_trace_every_n", 1)))
        self._debug_frame_counter = 0

    def _trace(self, message: str) -> None:
        if self.debug_tracing:
            print(f"[FluxRTDebug][subprocess] {message}")

    def _warn(self, message: str) -> None:
        print(f"[FluxRTWarning][subprocess] {message}")

    def _configure_transformer_optimizations(self) -> None:
        if self.config.get("compile_models", False):
            try:
                self.pipe.transformer = torch.compile(
                    self.pipe.transformer,
                )
                self.transformer = self.pipe.transformer
                self.pipe.vae = torch.compile(self.pipe.vae)
                self.vae = self.pipe.vae
                self.interpolation_model = torch.compile(
                    self.interpolation_model,
                )
                self._trace("compile applied transformer=on vae=on interpolation=on")
            except Exception as exc:
                self._warn(
                    f"compile setup failed with {type(exc).__name__}: {exc}; "
                    "continuing with uncompiled modules"
                )
                self.transformer = self.pipe.transformer
                self.vae = self.pipe.vae

    def init_process_state(self):
        self.device = "cuda"
        self.dtype = torch.bfloat16
        self.process_state = {
            "prompt": self.config["default_prompt"],
            "steps": self.config["default_steps"],
            "seed": self.config["default_seed"],
        }

    def enable_quantization(self):
        """Should be called before the subprocess is started."""
        self.config["enable_int8_quantization"] = True

    def load_models_without_quantization(self):
        device = self.device
        dtype = self.dtype

        models_path = self.config["models_path"]
        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", local_files_only=True, device=device
        )
        self.transformer = Flux2Transformer2DModel.from_pretrained(
            f"{models_path}/transformer", local_files_only=True, device=device
        ).to(dtype)
        self.text_encoder = Qwen3ForCausalLM.from_pretrained(
            f"{models_path}/text_encoder", local_files_only=True
        ).to(device, dtype)
        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{models_path}/tokenizer", local_files_only=True, device=device
        )

    def load_quantized_models(self):
        from optimum.quanto import requantize
        from fluxrt.stream_processor.quantized_flux2 import (
            QuantizedFlux2Transformer2DModel,
        )

        models_path = self.config["models_path"]
        int8_models_path = self.config["int8_models_path"]

        qtransformer = QuantizedFlux2Transformer2DModel.from_pretrained(
            int8_models_path, local_files_only=True
        )
        qtransformer.to(device=self.device, dtype=self.dtype)
        self.transformer = qtransformer._wrapped

        self.scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            f"{models_path}/scheduler", local_files_only=True, device=self.device
        )

        config = AutoConfig.from_pretrained(
            f"{int8_models_path}/text_encoder", local_files_only=True
        )
        with init_empty_weights():
            text_encoder = Qwen3ForCausalLM(config)

        with open(f"{int8_models_path}/text_encoder/quanto_qmap.json", "r") as f:
            qmap = json.load(f)
        state_dict = load_file(f"{int8_models_path}/text_encoder/model.safetensors")
        requantize(text_encoder, state_dict=state_dict, quantization_map=qmap)
        text_encoder.eval()
        text_encoder.to(self.device, dtype=self.dtype)
        self.text_encoder = text_encoder

        self.tokenizer = Qwen2TokenizerFast.from_pretrained(
            f"{int8_models_path}/tokenizer", local_files_only=True
        )

    def load_models(self):
        self.interpolation_model = IFNet()
        self.interpolation_model.load_state_dict(
            load_file("RIFE-safetensors/flownet.safetensors")
        )
        # interpolation model requires float16 to avoid grid_sample artifacts
        self.interpolation_model.to(self.device, torch.float16)
        self.interpolation_model.eval()

        if self.config.get("enable_int8_quantization", False):
            self.load_quantized_models()
        else:
            self.load_models_without_quantization()

        if self.config.get("enable_flow_upscaler", False):
            self.upscaler_unet = UpscalerUNet()
            state_dict = load_file("FlowUpscaler/flow_upscaler.safetensors")
            self.upscaler_unet.load_state_dict(state_dict)
            self.upscaler_unet.to(self.device, self.dtype)
            self.upscaler_pipe = FlowUpscalerPipeline(
                self.upscaler_unet, self.scheduler
            )
        else:
            self.upscaler_pipe = None

        if self.config.get("enable_tiny_vae", False):
            self.vae = DiffusersTAEF2Wrapper(path="taef2/taef2.safetensors").to(
                self.device, self.dtype
            )
        else:
            models_path = self.config["models_path"]
            self.vae = AutoencoderKLFlux2.from_pretrained(
                f"{models_path}/vae", local_files_only=True, device=self.device
            ).to(self.dtype)

        reference_image_seq_len = None
        if self.config.get("use_reference_image", False):
            reference_image_res = self.config["reference_image_resolution"]
            reference_image_seq_len = (reference_image_res["width"] // 16) * (
                reference_image_res["height"] // 16
            )

        self.update_controller = UpdateController(
            self.config,
            self.height,
            self.width,
            compression_ratio=16,
            reference_image_seq_len=reference_image_seq_len,
        )

        self.pipe = Flux2KleinPipeline(
            scheduler=self.scheduler,
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            transformer=self.transformer,
            update_controller=self.update_controller,
            subprocess_config=self.config,
            upscaler_pipeline=self.upscaler_pipe,
        )
        self.pipe.to(self.device)
        if self.config.get("use_lora", False):
            self.pipe.load_lora_weights(self.config.get("lora_weights_path", ""))
        self._configure_transformer_optimizations()
        self.lip_processor: BasePostProcessor | None = None
        self.lip_active = False
        lp_cfg = self.config.get("lip_transfer", {})
        if lp_cfg.get("enable", False):
            self.lip_processor = LivePortraitPostProcessor(
                models_dir=lp_cfg["models_dir"]
            )
        self._trace(
            "models loaded "
            f"resolution={self.width}x{self.height} "
            f"compile={self.config.get('compile_models', False)} "
            f"spatial_cache={self.config.get('enable_spatial_cache', False)} "
            f"mask_mode={self.config.get('mask_calculation_method', 'auto')}"
        )

    def update_prompt_embeds(self, prompt):
        self.prompt_embeds, text_ids = self.pipe.encode_prompt(
            prompt=prompt,
            device=self.device,
            num_images_per_prompt=1,
            max_sequence_length=512,
            text_encoder_out_layers=(9, 18, 27),
        )
        self.update_controller.reset_cache()
        self._trace(
            f"prompt updated len={len(prompt)} "
            f"prompt_embeds={tuple(self.prompt_embeds.shape)} "
            f"text_ids={tuple(text_ids.shape)} cache_reset=True"
        )

    def init_shared_tensors(self):
        h, w = self.resolution["height"], self.resolution["width"]
        out_h, out_w = h, w
        if self.config.get("enable_flow_upscaler", False):
            out_h, out_w = h * 2, w * 2

        self.input_shared_tensor = SharedTensor(
            (h, w, 3),
            name=self.input_shared_tensor_name,
        )

        # All interpolated then one original
        output_batch_size = 2**self.interpolation_exp
        self.output_batch_shared_tensor = SharedTensor(
            (output_batch_size, out_h, out_w, 3),
            name=self.output_batch_shared_tensor_name,
        )

    def process_init(self):
        """
        Initializes all resources required by the inference subprocess.
        """
        self.init_process_state()
        self.init_shared_tensors()
        self.load_models()
        self.update_prompt_embeds(self.process_state["prompt"])
        self.previous_frame = None
        self._trace(
            "process initialized "
            f"target_fps={self.config.get('target_fps')} "
            f"interpolation_exp={self.interpolation_exp}"
        )

        if self.config.get("use_reference_image", False):
            image = cv2.imread(self.config.get("reference_image_path", ""))
            resolution = self.config.get("reference_image_resolution")
            if image is None:
                image = np.zeros(
                    (resolution["height"], resolution["width"], 3), dtype=np.uint8
                )
                print(
                    "Warning: use_reference_image is set to true but no valid reference_image_path is provided."
                )
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (resolution["width"], resolution["height"]))
            self.reference_image = Image.fromarray(image)

        target_fps = self.config.get("target_fps", None)
        self.target_base_processing_time = None
        if target_fps is not None:
            target_base_fps = target_fps / (2**self.interpolation_exp)
            self.target_base_processing_time = 1 / target_base_fps

    def start(self):
        self.running.value = True
        self.process = Process(target=self.process_main)
        self.process.start()

    def stop(self):
        self.running.value = False
        if self.process:
            self.process.join()

    def set_param(self, name: str, value) -> None:
        self.command_queue.put(("set_param", (name, value)))

    def set_reference_image(self, image: np.ndarray | None) -> None:
        """
        Update the reference image on the fly.
        image: numpy uint8 RGB array
        Only valid when use_reference_image is true in config.
        """
        if not self.config.get("use_reference_image", False):
            raise ValueError(
                "set_reference_image called but use_reference_image is not enabled in the stream processor config"
            )
        self.command_queue.put(("set_reference_image", image))

    def set_mask(self, mask) -> None:
        """
        Update the mask on the fly.
        mask: numpy uint8 array of shape (h // compression_ratio, w // compression_ratio).
        Only valid when mask_calculation_method is set to manual in config.
        """
        if self.config.get("mask_calculation_method", "auto") != "manual":
            raise ValueError(
                "set_mask called but mask_calculation_method is not set to manual in the config"
            )
        self.command_queue.put(("set_mask", mask))

    def update_process_state(self) -> None:
        """
        Called by the internal process
        """
        try:
            while True:
                cmd, payload = self.command_queue.get_nowait()
                if cmd == "set_param":
                    name, value = payload
                    self.process_state[name] = value
                    if name == "prompt":
                        self.update_prompt_embeds(value)
                elif cmd == "set_reference_image":
                    image = payload  # numpy uint8 RGB array or None
                    resolution = self.config["reference_image_resolution"]
                    if image is not None:
                        image = cv2.resize(
                            image, (resolution["width"], resolution["height"])
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
                    self._trace(
                        "reference image updated "
                        f"shape={(resolution['height'], resolution['width'], 3)} cache_reset=True"
                    )

                elif cmd == "set_mask":
                    mask = payload  # numpy uint8 array of shape (h // compression_ratio, w // compression_ratio)
                    mask_tensor = (
                        torch.from_numpy(mask)
                        .unsqueeze(0)
                        .to(self.update_controller.device)
                    )
                    self.update_controller.set_mask(mask_tensor)
                    active = int(np.count_nonzero(mask))
                    total = int(mask.size)
                    self._trace(
                        f"manual mask updated shape={mask.shape} active={active}/{total} "
                        f"active_pct={(active / total * 100.0) if total else 0.0:.2f}"
                    )

        except Empty:
            pass

    def receive_frame(self):
        """
        Reads frame from input shared memory, converts to RGB float16 GPU tensors.
        """
        frame = self.input_shared_tensor.to_numpy()
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
        """
        Takes one new generated frame (torch tensor, RGB, on GPU, float16)
        Interpolates according to interpolation_exp times.
        Batches to [interpolated frames, new frame].
        """
        if self.previous_frame is None:
            self.previous_frame = frame

        if self.interpolation_exp == 0:
            frames_out = frame
        else:
            frames = torch.cat([self.previous_frame, frame], dim=0)
            with torch.no_grad():
                for _ in range(self.interpolation_exp):
                    B = frames.size(0)
                    prevs = frames[:-1]
                    nexts = frames[1:]
                    mids = self.interpolation_model(torch.cat([prevs, nexts], dim=1))
                    H, W = frames.shape[2:]
                    new_frames = torch.empty(
                        2 * B - 1, 3, H, W, device=frames.device, dtype=frames.dtype
                    )
                    new_frames[0::2] = frames
                    new_frames[1::2] = mids
                    frames = new_frames
            frames_out = frames[1:]

        frames_cpu = (
            frames_out.mul(255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )

        self.previous_frame = frame

        return frames_cpu[..., ::-1]

    def send_frames(self, frames):
        self.output_batch_shared_tensor.copy_from(frames)

    def sync_fps_and_send(self, prev_time, frames):
        now = time.time()
        processing_time = now - prev_time

        if self.target_base_processing_time is not None:
            sleep_time = max(0, self.target_base_processing_time - processing_time)
            time.sleep(sleep_time)
            now = time.time()

        processing_time = now - prev_time

        self.last_processing_time.value = processing_time
        self.send_frames(frames)
        self.pack_is_ready.value = True
        self.memory_reserved.value = torch.cuda.memory_reserved() // (1024 * 1024)

        if self.logging:
            base_fps = 1 / processing_time
            interpolated_fps = base_fps * (2**self.interpolation_exp)
            print(
                f"processing_time_s: {processing_time:.4f}, base fps: {base_fps:.2f}, "
                f"interpolated fps: {interpolated_fps:.2f}, reserved_mb: {self.memory_reserved.value}"
            )
        if self.debug_tracing and self._debug_frame_counter % self.debug_trace_every_n == 0:
            self._trace(
                f"frame_complete idx={self._debug_frame_counter} "
                f"processing_ms={processing_time * 1000.0:.1f} "
                f"reserved_mb={self.memory_reserved.value}"
            )
        return now

    def process_frame_with_pipeline(self, frame):
        """
        Takes frame as np uint8 RGB array
        Returns frame as np uint8 RGB array
        """
        input_frame = Image.fromarray(frame)

        reference_list = [input_frame]
        if self.config["use_reference_image"]:
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
        out_image = out.images[0]
        out_image = out_image * 255
        out_image = out_image.astype(np.uint8)
        if self.debug_tracing and self._debug_frame_counter % self.debug_trace_every_n == 0:
            self._trace(
                f"pipe_call idx={self._debug_frame_counter} "
                f"input_shape={frame.shape} output_shape={out_image.shape} "
                f"steps={self.process_state['steps']} refs={len(reference_list)}"
            )
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
        self.process_init()
        prev_time = time.time()
        while self.running.value:
            self._debug_frame_counter += 1
            self.update_process_state()
            frame = self.input_shared_tensor.to_numpy()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = self.process_frame_with_pipeline(frame)
            frame = self.convert_np_to_torch(frame)
            frames = self.interpolate_frames(frame)
            prev_time = self.sync_fps_and_send(prev_time, frames)

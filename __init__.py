"""Glue for richservo/comfyui-moss-tts on Graydient runners.

Why this exists
---------------
richservo's MossTTSModelLoader (a) snapshot_download()s the model from the Hub at run
time and (b) only passes a local codec_path for MOSS-TTSD, so SoundEffect/VoiceGenerator
would pull the 7 GB OpenMOSS-Team/MOSS-Audio-Tokenizer from Hugging Face *inside* the job
clock. MossLocalLoader reads the folders that the workflow's concept_mapping pre-staged
under models/moss-tts/ and returns the same MOSS_TTS_PIPE tuple, so richservo's
MossTTSSoundEffect / MossTTSVoiceDesign nodes work unchanged.

It also keeps the fp32 codec (~7 GB) on the CPU during generation on GPUs under ~30 GB
(8B bf16 LLM ~16.7 GB + codec would not fit a 24 GB card) and swaps them just for decode.

MossVoicePromptSplit: one Graydient prompt -> (speech text, voice instruction)
using the confirmed-working `::` marker:  "<voice description> :: <what to say>".
"""
import os

import torch

import comfy.model_management as mm
import folder_paths

SAMPLE_RATE = 24000
MOSS_DIR = os.path.join(folder_paths.models_dir, "moss-tts")
CODEC_ID = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
VARIANTS = {
    "MOSS-SoundEffect": "OpenMOSS-Team/MOSS-SoundEffect",
    "MOSS-VoiceGenerator": "OpenMOSS-Team/MOSS-VoiceGenerator",
}
DEFAULT_INSTRUCTION = "A clear, natural, neutral voice."

_cache = {}


def _local_dir(repo_id):
    d = os.path.join(MOSS_DIR, repo_id.replace("/", "--"))
    if os.path.isfile(os.path.join(d, "config.json")):
        print(f"[MossGlue] using staged local copy: {d}")
        return d
    from huggingface_hub import snapshot_download

    print(f"[MossGlue] {d} not staged -> downloading {repo_id} (counts against the run clock!)")
    return snapshot_download(repo_id, local_dir=d)


class MossLocalLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model_variant": (list(VARIANTS.keys()),)}}

    RETURN_TYPES = ("MOSS_TTS_PIPE",)
    FUNCTION = "load"
    CATEGORY = "audio/MOSS-TTS"

    def load(self, model_variant):
        from transformers import AutoModel

        if model_variant in _cache:
            return (_cache[model_variant],)

        torch.backends.cuda.enable_cudnn_sdp(False)  # model card: cuDNN SDPA is broken for MOSS
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)

        model_id = VARIANTS[model_variant]
        device = mm.get_torch_device()
        cuda = str(device).startswith("cuda")
        dtype = torch.bfloat16 if cuda else torch.float32
        mm.unload_all_models()

        model_dir = _local_dir(model_id)
        codec_dir = _local_dir(CODEC_ID)

        kwargs = {"trust_remote_code": True, "codec_path": codec_dir}
        if model_variant == "MOSS-VoiceGenerator":
            kwargs["normalize_inputs"] = True
        # AutoProcessor (transformers 5.0.0) injects revision="main", which MOSS's processor forwards
        # into ProcessorMixin.__init__ ("Unexpected keyword argument revision"). Load the class directly.
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        proc_cls = get_class_from_dynamic_module("processing_moss_tts.MossTTSDelayProcessor", model_dir)
        processor = proc_cls.from_pretrained(model_dir, **kwargs)

        model = AutoModel.from_pretrained(
            model_dir, trust_remote_code=True,
            attn_implementation="sdpa" if cuda else "eager", torch_dtype=dtype,
        ).to(device).eval()

        codec = processor.audio_tokenizer
        total = torch.cuda.get_device_properties(device).total_memory if cuda else 0
        offload = cuda and total < 30 * 1024**3 and model_variant == "MOSS-SoundEffect"
        print(f"[MossGlue] {model_variant}: gpu={total/1024**3:.0f} GB, codec offload during generate={offload}")

        if offload:
            codec.to("cpu")
            orig_decode = processor.decode

            def decode(*a, **k):
                model.to("cpu")
                torch.cuda.empty_cache()
                codec.to(device)
                try:
                    return orig_decode(*a, **k)
                finally:
                    codec.to("cpu")
                    torch.cuda.empty_cache()

            processor.decode = decode
        else:
            processor.audio_tokenizer = codec.to(device)

        pipe = (model, processor, SAMPLE_RATE, device, model_id)
        _cache.clear()
        _cache[model_variant] = pipe
        return (pipe,)

    @classmethod
    def IS_CHANGED(cls, model_variant):
        return model_variant


class MossVoicePromptSplit:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"prompt": ("STRING", {"default": "", "multiline": True})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "instruction")
    FUNCTION = "split"
    CATEGORY = "audio/MOSS-TTS"

    def split(self, prompt):
        if "::" in prompt:
            instr, text = prompt.split("::", 1)
        else:
            instr, text = "", prompt
        instr, text = instr.strip() or DEFAULT_INSTRUCTION, text.strip()
        if not text:
            raise ValueError("MossVoicePromptSplit: no speech text. Use '<voice description> :: <what to say>'.")
        print(f"[MossGlue] instruction={instr!r}\n[MossGlue] text={text!r}")
        return (text, instr)


NODE_CLASS_MAPPINGS = {"MossLocalLoader": MossLocalLoader, "MossVoicePromptSplit": MossVoicePromptSplit}
NODE_DISPLAY_NAME_MAPPINGS = {"MossLocalLoader": "MOSS Local Loader (staged weights)",
                              "MossVoicePromptSplit": "MOSS Voice Prompt Split (voice :: text)"}

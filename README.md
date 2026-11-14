# ComfyUI-MossGlue
Glue for `richservo/comfyui-moss-tts` on Graydient: loads MOSS-SoundEffect / MOSS-VoiceGenerator and the MOSS-Audio-Tokenizer from `concept_mapping`-staged folders in `models/moss-tts/` (no run-time HF download), offloads the codec on <30 GB GPUs, and splits `voice description :: text` prompts.

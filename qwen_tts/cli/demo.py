# coding=utf-8
# Copyright 2026 The Alibaba Qwen team. SPDX-License-Identifier: Apache-2.0
"""A gradio demo for Qwen3 TTS models - English only UI."""

import argparse
import os
import tempfile
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr
import numpy as np
import torch

from qwen_tts import Qwen3TTSModel, VoiceClonePromptItem


def _title_case_display(s: str) -> str:
    s = (s or "").strip().replace("_", " ")
    return " ".join([w[:1].upper() + w[1:] if w else "" for w in s.split()])


def _build_choices_and_map(items: Optional[List[str]]) -> Tuple[List[str], Dict[str, str]]:
    if not items:
        return [], {}
    display = [_title_case_display(x) for x in items]
    mapping = {d: r for d, r in zip(display, items)}
    return display, mapping


def _dtype_from_str(s: str) -> torch.dtype:
    s = (s or "").strip().lower()
    if s in ("bf16", "bfloat16"): return torch.bfloat16
    if s in ("fp16", "float16", "half"): return torch.float16
    if s in ("fp32", "float32"): return torch.float32
    raise ValueError(f"Unsupported torch dtype: {s}.")


def _maybe(v):
    return v if v is not None else gr.update()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="qwen-tts-demo", formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("checkpoint_pos", nargs="?", default=None)
    parser.add_argument("-c", "--checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16","bf16","float16","fp16","float32","fp32"])
    parser.add_argument("--flash-attn/--no-flash-attn", dest="flash_attn", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--ip", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--share/--no-share", dest="share", default=False, action=argparse.BooleanOptionalAction)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--ssl-certfile", default=None)
    parser.add_argument("--ssl-keyfile", default=None)
    parser.add_argument("--ssl-verify/--no-ssl-verify", dest="ssl_verify", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--subtalker-top-k", type=int, default=None)
    parser.add_argument("--subtalker-top-p", type=float, default=None)
    parser.add_argument("--subtalker-temperature", type=float, default=None)
    return parser


def _resolve_checkpoint(args):
    ckpt = args.checkpoint or args.checkpoint_pos
    if not ckpt: raise SystemExit(0)
    return ckpt


def _collect_gen_kwargs(args):
    mapping = {
        "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
        "top_k": args.top_k, "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
        "subtalker_top_k": args.subtalker_top_k, "subtalker_top_p": args.subtalker_top_p,
        "subtalker_temperature": args.subtalker_temperature,
    }
    return {k: v for k, v in mapping.items() if v is not None}


def _normalize_audio(wav, eps=1e-12, clip=True):
    x = np.asarray(wav)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        y = x.astype(np.float32) / max(abs(info.min), info.max) if info.min < 0 else (x.astype(np.float32) - (info.max+1)/2.0) / ((info.max+1)/2.0)
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6: y = y / (m + eps)
    else:
        raise TypeError(f"Unsupported dtype: {x.dtype}")
    if clip: y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1: y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio):
    if audio is None: return None
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        return _normalize_audio(audio["data"]), int(audio["sampling_rate"])
    return None


def _wav_to_gradio_audio(wav, sr):
    return sr, np.asarray(wav, dtype=np.float32)


def _detect_model_kind(ckpt, tts):
    mt = getattr(tts.model, "tts_model_type", None)
    if mt in ("custom_voice", "voice_design", "base"): return mt
    raise ValueError(f"Unknown model type: {mt}")


def build_demo(tts, ckpt, gen_kwargs_default):
    model_kind = _detect_model_kind(ckpt, tts)
    supported_langs_raw = tts.model.get_supported_languages() if callable(getattr(tts.model, "get_supported_languages", None)) else None
    supported_spks_raw = tts.model.get_supported_speakers() if callable(getattr(tts.model, "get_supported_speakers", None)) else None
    lang_choices_disp, lang_map = _build_choices_and_map(list(supported_langs_raw or []))
    spk_choices_disp, spk_map = _build_choices_and_map(list(supported_spks_raw or []))

    def _gen_common_kwargs(): return dict(gen_kwargs_default)

    theme = gr.themes.Soft(font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"])
    css = ".gradio-container {max-width: none !important;}"

    with gr.Blocks(theme=theme, css=css) as demo:
        gr.Markdown(f"# Qwen3 TTS Demo\n**Checkpoint:** `{ckpt}`  **Model Type:** `{model_kind}`")

        if model_kind == "custom_voice":
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(label="Text", lines=4, placeholder="Enter text to synthesize.")
                    with gr.Row():
                        lang_in = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Auto", interactive=True)
                        spk_in = gr.Dropdown(label="Speaker", choices=spk_choices_disp, value="Vivian", interactive=True)
                    instruct_in = gr.Textbox(label="Instruction (Optional)", lines=2, placeholder="e.g. Say it in a very angry tone.")
                    btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="Output Audio", type="numpy")
                    err = gr.Textbox(label="Status", lines=2)

            def run_instruct(text, lang_disp, spk_disp, instruct):
                try:
                    if not (text or "").strip(): return None, "Text is required."
                    if not spk_disp: return None, "Speaker is required."
                    wavs, sr = tts.generate_custom_voice(text=text.strip(), language=lang_map.get(lang_disp,"Auto"), speaker=spk_map.get(spk_disp,spk_disp), instruct=(instruct or "").strip() or None, **_gen_common_kwargs())
                    return _wav_to_gradio_audio(wavs[0], sr), "Finished."
                except Exception as e: return None, f"{type(e).__name__}: {e}"
            btn.click(run_instruct, inputs=[text_in,lang_in,spk_in,instruct_in], outputs=[audio_out,err])

        elif model_kind == "voice_design":
            with gr.Row():
                with gr.Column(scale=2):
                    text_in = gr.Textbox(label="Text", lines=4, value="It's in the top drawer... wait, it's empty? No way, that's impossible!")
                    lang_in = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Auto", interactive=True)
                    design_in = gr.Textbox(label="Voice Design Instruction", lines=3, value="Speak in an incredulous tone, but with a hint of panic beginning to creep into your voice.")
                    btn = gr.Button("Generate", variant="primary")
                with gr.Column(scale=3):
                    audio_out = gr.Audio(label="Output Audio", type="numpy")
                    err = gr.Textbox(label="Status", lines=2)

            def run_voice_design(text, lang_disp, design):
                try:
                    if not (text or "").strip(): return None, "Text is required."
                    if not (design or "").strip(): return None, "Voice design instruction is required."
                    wavs, sr = tts.generate_voice_design(text=text.strip(), language=lang_map.get(lang_disp,"Auto"), instruct=design.strip(), **_gen_common_kwargs())
                    return _wav_to_gradio_audio(wavs[0], sr), "Finished."
                except Exception as e: return None, f"{type(e).__name__}: {e}"
            btn.click(run_voice_design, inputs=[text_in,lang_in,design_in], outputs=[audio_out,err])

        else:  # base
            with gr.Tabs():
                with gr.Tab("Clone & Generate"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            ref_audio = gr.Audio(label="Reference Audio", type="numpy")
                            ref_text = gr.Textbox(
                                label="Reference Audio Text",
                                lines=2,
                                placeholder="Required unless 'Use x-vector only' is checked.",
                            )
                            xvec_only = gr.Checkbox(label="Use x-vector only (lower quality, no reference text needed)", value=False)
                        with gr.Column(scale=2):
                            text_in = gr.Textbox(label="Target Text", lines=4, placeholder="Enter text to synthesize.")
                            lang_in = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Auto", interactive=True)
                            btn = gr.Button("Generate", variant="primary")
                        with gr.Column(scale=3):
                            audio_out = gr.Audio(label="Output Audio", type="numpy")
                            err = gr.Textbox(label="Status", lines=2)

                            gr.Markdown("---\n### Automatic Transcription (FasterWhisper)")
                            gr.Markdown(
                                "Transcribes the **Reference Audio** and automatically fills in the **Reference Audio Text** field. "
                                "FasterWhisper may make mistakes — always listen and correct the transcription for best results."
                            )
                            with gr.Row():
                                whisper_model_select = gr.Dropdown(
                                    label="FasterWhisper Model",
                                    choices=[
                                        "tiny", "tiny.en",
                                        "base", "base.en",
                                        "small", "small.en",
                                        "medium", "medium.en",
                                        "large-v1", "large-v2", "large-v3",
                                        "distil-large-v2", "distil-large-v3",
                                    ],
                                    value="large-v3",
                                    interactive=True,
                                )
                            with gr.Row():
                                whisper_device = gr.Dropdown(
                                    label="Device",
                                    choices=["cuda", "cpu"],
                                    value="cuda",
                                    interactive=True,
                                )
                                whisper_compute = gr.Dropdown(
                                    label="Compute Type",
                                    choices=["float16", "int8_float16", "int8", "float32"],
                                    value="float16",
                                    interactive=True,
                                )
                            whisper_btn = gr.Button("Transcribe and fill reference text", variant="secondary")
                            whisper_lang_out = gr.Textbox(label="Detected Language", lines=1, interactive=False)
                            whisper_status = gr.Textbox(label="Transcription Status", lines=1, interactive=False)

                    def run_voice_clone(ref_aud, ref_txt, use_xvec, text, lang_disp):
                        try:
                            if not (text or "").strip(): return None, "Target text is required."
                            at = _audio_to_tuple(ref_aud)
                            if at is None: return None, "Reference audio is required."
                            if not use_xvec and not (ref_txt or "").strip():
                                return None, "Reference text is required when 'Use x-vector only' is NOT enabled.\nOtherwise check 'Use x-vector only' (lower quality)."
                            wavs, sr = tts.generate_voice_clone(text=text.strip(), language=lang_map.get(lang_disp,"Auto"), ref_audio=at, ref_text=(ref_txt.strip() if ref_txt else None), x_vector_only_mode=bool(use_xvec), **_gen_common_kwargs())
                            return _wav_to_gradio_audio(wavs[0], sr), "Finished."
                        except Exception as e: return None, f"{type(e).__name__}: {e}"
                    btn.click(run_voice_clone, inputs=[ref_audio,ref_text,xvec_only,text_in,lang_in], outputs=[audio_out,err])

                    def run_whisper_from_ref(ref_aud, model_size, device, compute_type):
                        """Transcribes the reference audio already loaded in the UI."""
                        import soundfile as sf
                        if ref_aud is None:
                            return "", "", "Please load a reference audio first."
                        try:
                            from faster_whisper import WhisperModel
                            if isinstance(ref_aud, tuple):
                                sr, wav = ref_aud
                            elif isinstance(ref_aud, dict):
                                sr = ref_aud["sampling_rate"]
                                wav = ref_aud["data"]
                            else:
                                return "", "", "Unknown audio format."

                            wav = np.asarray(wav, dtype=np.float32)
                            if wav.ndim > 1:
                                wav = np.mean(wav, axis=-1)
                            if np.max(np.abs(wav)) > 1.0:
                                wav = wav / np.max(np.abs(wav))

                            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
                            os.close(fd)
                            sf.write(tmp_path, wav, sr)

                            if device == "cpu" and compute_type in ("float16", "int8_float16"):
                                compute_type = "int8"
                            wmodel = WhisperModel(model_size, device=device, compute_type=compute_type)
                            segments, info = wmodel.transcribe(tmp_path, beam_size=5)
                            full_text = " ".join(seg.text.strip() for seg in segments)
                            lang_info = f"{info.language} (probability: {info.language_probability:.2f})"
                            del wmodel
                            os.remove(tmp_path)
                            return full_text, lang_info, "Transcription complete."
                        except Exception as e:
                            return "", "", f"{type(e).__name__}: {e}"

                    whisper_btn.click(
                        run_whisper_from_ref,
                        inputs=[ref_audio, whisper_model_select, whisper_device, whisper_compute],
                        outputs=[ref_text, whisper_lang_out, whisper_status],
                    )

                with gr.Tab("Save / Load Voice"):
                    with gr.Row():
                        with gr.Column(scale=2):
                            gr.Markdown("### Save Voice\nUpload reference audio and text, then save a reusable voice prompt file.")
                            ref_audio_s = gr.Audio(label="Reference Audio", type="numpy")
                            ref_text_s = gr.Textbox(label="Reference Text", lines=2, placeholder="Required unless 'Use x-vector only' is checked.")
                            xvec_only_s = gr.Checkbox(label="Use x-vector only (lower quality, no reference text needed)", value=False)
                            save_btn = gr.Button("Save Voice File", variant="primary")
                            prompt_file_out = gr.File(label="Voice File")
                        with gr.Column(scale=2):
                            gr.Markdown("### Load Voice & Generate\nUpload a previously saved voice file, then synthesize new text.")
                            prompt_file_in = gr.File(label="Upload Prompt File")
                            text_in2 = gr.Textbox(label="Target Text", lines=4, placeholder="Enter text to synthesize.")
                            lang_in2 = gr.Dropdown(label="Language", choices=lang_choices_disp, value="Auto", interactive=True)
                            gen_btn2 = gr.Button("Generate", variant="primary")
                        with gr.Column(scale=3):
                            audio_out2 = gr.Audio(label="Output Audio", type="numpy")
                            err2 = gr.Textbox(label="Status", lines=2)

                    def save_prompt(ref_aud, ref_txt, use_xvec):
                        try:
                            at = _audio_to_tuple(ref_aud)
                            if at is None: return None, "Reference audio is required."
                            if not use_xvec and not (ref_txt or "").strip():
                                return None, "Reference text is required when 'Use x-vector only' is NOT enabled."
                            items = tts.create_voice_clone_prompt(ref_audio=at, ref_text=(ref_txt.strip() if ref_txt else None), x_vector_only_mode=bool(use_xvec))
                            fd, out_path = tempfile.mkstemp(prefix="voice_clone_prompt_", suffix=".pt")
                            os.close(fd)
                            torch.save({"items": [asdict(it) for it in items]}, out_path)
                            return out_path, "Finished."
                        except Exception as e: return None, f"{type(e).__name__}: {e}"

                    def load_prompt_and_gen(file_obj, text, lang_disp):
                        try:
                            if file_obj is None: return None, "Voice file is required."
                            if not (text or "").strip(): return None, "Target text is required."
                            path = getattr(file_obj,"name",None) or getattr(file_obj,"path",None) or str(file_obj)
                            payload = torch.load(path, map_location="cpu", weights_only=True)
                            if not isinstance(payload,dict) or "items" not in payload: return None, "Invalid file format."
                            items_raw = payload["items"]
                            if not isinstance(items_raw,list) or len(items_raw)==0: return None, "Empty voice items."
                            items = []
                            for d in items_raw:
                                if not isinstance(d,dict): return None, "Invalid item format in file."
                                ref_code = d.get("ref_code",None)
                                if ref_code is not None and not torch.is_tensor(ref_code): ref_code = torch.tensor(ref_code)
                                ref_spk = d.get("ref_spk_embedding",None)
                                if ref_spk is None: return None, "Missing ref_spk_embedding."
                                if not torch.is_tensor(ref_spk): ref_spk = torch.tensor(ref_spk)
                                items.append(VoiceClonePromptItem(ref_code=ref_code, ref_spk_embedding=ref_spk, x_vector_only_mode=bool(d.get("x_vector_only_mode",False)), icl_mode=bool(d.get("icl_mode", not bool(d.get("x_vector_only_mode",False)))), ref_text=d.get("ref_text",None)))
                            wavs, sr = tts.generate_voice_clone(text=text.strip(), language=lang_map.get(lang_disp,"Auto"), voice_clone_prompt=items, **_gen_common_kwargs())
                            return _wav_to_gradio_audio(wavs[0], sr), "Finished."
                        except Exception as e: return None, f"Failed to use voice file.\n{type(e).__name__}: {e}"

                    save_btn.click(save_prompt, inputs=[ref_audio_s,ref_text_s,xvec_only_s], outputs=[prompt_file_out,err2])
                    gen_btn2.click(load_prompt_and_gen, inputs=[prompt_file_in,text_in2,lang_in2], outputs=[audio_out2,err2])

        gr.Markdown("**Disclaimer:** Audio is AI-generated for demo purposes only. Do not use to generate harmful, unlawful, or fraudulent content.")

    return demo


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.checkpoint and not args.checkpoint_pos:
        parser.print_help()
        return 0
    ckpt = _resolve_checkpoint(args)
    tts = Qwen3TTSModel.from_pretrained(ckpt, device_map=args.device, dtype=_dtype_from_str(args.dtype), attn_implementation="flash_attention_2" if args.flash_attn else None)
    demo = build_demo(tts, ckpt, _collect_gen_kwargs(args))
    launch_kwargs: Dict[str, Any] = dict(server_name=args.ip, server_port=args.port, share=args.share, ssl_verify=bool(args.ssl_verify))
    if args.ssl_certfile: launch_kwargs["ssl_certfile"] = args.ssl_certfile
    if args.ssl_keyfile: launch_kwargs["ssl_keyfile"] = args.ssl_keyfile
    demo.queue(default_concurrency_limit=int(args.concurrency)).launch(**launch_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
import sys
import tempfile
import types
from pathlib import Path


LANGUAGE_TO_MELO = {
    "en": "EN_NEWEST",
    "es": "ES",
    "fr": "FR",
    "zh": "ZH",
    "jp": "JP",
    "kr": "KR",
}


def main() -> int:
    _configure_traceback_watchdog()
    parser = argparse.ArgumentParser(description="Run OpenVoice V2 synthesis in an isolated Python environment.")
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()

    payload = json.loads(args.request.read_text(encoding="utf-8"))
    output = Path(payload["output_wav"])
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(payload["checkpoint_dir"])
    language = payload["language"]
    melo_language = LANGUAGE_TO_MELO[language]
    melo_model_dir = Path(payload.get("melo_model_dir") or checkpoint_dir.parent / "melo" / melo_language)

    _log("importing torch")
    import torch  # type: ignore

    _log(f"torch={torch.__version__}")
    _configure_torch_threads(torch)
    _install_lightweight_runtime_shims(melo_language, melo_model_dir)
    _log("importing MeloTTS")
    from melo.api import TTS  # type: ignore

    _log("importing OpenVoice converter")
    from openvoice.api import ToneColorConverter  # type: ignore

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _log(f"device={device}")
    _log("loading tone color converter")
    converter = ToneColorConverter(str(checkpoint_dir / "converter" / "config.json"), device=device)
    converter.watermark_model = None
    converter.load_ckpt(str(checkpoint_dir / "converter" / "checkpoint.pth"))

    _log(f"loading MeloTTS language={melo_language}")
    tts = TTS(language=melo_language, device=device)
    tts.hps.data.disable_bert = True
    _log("disabled MeloTTS BERT feature extraction for local inference")
    speaker_key = _choose_speaker(tts.hps.data.spk2id)
    source_se = _source_se_path(checkpoint_dir, speaker_key)
    speaker_id = tts.hps.data.spk2id[speaker_key]
    _log(f"extracting target speaker embedding from {payload['reference_wav']}")
    target_se = converter.extract_se([str(payload["reference_wav"])])

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as handle:
        source_wav = Path(handle.name)
    try:
        _log("generating base TTS audio")
        tts.tts_to_file(payload["text"], speaker_id, str(source_wav), speed=1.0)
        _log("converting tone color")
        converter.convert(
            audio_src_path=str(source_wav),
            src_se=torch.load(source_se, map_location=device),
            tgt_se=target_se,
            output_path=str(output),
        )
        _log(f"done output={output}")
    finally:
        source_wav.unlink(missing_ok=True)
    return 0


def _choose_speaker(speaker_ids: dict[str, int]) -> str:
    preferred = ["EN-Newest", "EN-US", "EN-BR", "ES", "FR", "ZH", "JP", "KR"]
    for key in preferred:
        if key in speaker_ids:
            return key
    return next(iter(speaker_ids))


class _NoWatermarkModel:
    def to(self, device: str):
        return None


def _log(message: str) -> None:
    print(f"[openvoice] {message}", file=sys.stderr, flush=True)


def _configure_traceback_watchdog() -> None:
    faulthandler.enable(file=sys.stderr)
    trace_after_sec = _env_int("OPENVOICE_TRACE_AFTER_SEC", 120)
    if trace_after_sec > 0:
        faulthandler.dump_traceback_later(trace_after_sec, repeat=True, file=sys.stderr)
        _log(f"traceback watchdog active after {trace_after_sec}s")


def _configure_torch_threads(torch_module) -> None:
    thread_count = max(1, min(4, os.cpu_count() or 1))
    try:
        torch_module.set_num_threads(thread_count)
        _log(f"torch threads={thread_count}")
    except RuntimeError as exc:
        _log(f"could not set torch threads: {exc}")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _install_lightweight_runtime_shims(melo_language: str, melo_model_dir: Path) -> None:
    """Avoid slow numba/librosa imports in OpenVoice inference-only runs."""
    _install_melo_monotonic_align_shim()
    _install_librosa_shim()
    _install_torchaudio_shim()
    _install_transformers_shim()
    _install_melo_download_shim(melo_language, melo_model_dir)
    _install_melo_text_language_shims(melo_language)
    _install_openvoice_text_shim()
    _install_wavmark_shim()
    _log(
        "installed inference shims for librosa, torchaudio, transformers, downloads, "
        "openvoice.text, wavmark, and melo.monotonic_align"
    )


def _install_melo_monotonic_align_shim() -> None:
    module = types.ModuleType("melo.monotonic_align")

    def maximum_path_jit(paths, values, t_ys, t_xs) -> None:
        batch = paths.shape[0]
        max_neg_val = -1e9
        for index_batch in range(int(batch)):
            path = paths[index_batch]
            value = values[index_batch]
            t_y = int(t_ys[index_batch])
            t_x = int(t_xs[index_batch])
            index = t_x - 1

            for y_index in range(t_y):
                x_start = max(0, t_x + y_index - t_y)
                x_end = min(t_x, y_index + 1)
                for x_index in range(x_start, x_end):
                    if x_index == y_index:
                        v_cur = max_neg_val
                    else:
                        v_cur = value[y_index - 1, x_index]
                    if x_index == 0:
                        v_prev = 0.0 if y_index == 0 else max_neg_val
                    else:
                        v_prev = value[y_index - 1, x_index - 1]
                    value[y_index, x_index] += max(v_prev, v_cur)

            for y_index in range(t_y - 1, -1, -1):
                path[y_index, index] = 1
                if index != 0 and (
                    index == y_index or value[y_index - 1, index] < value[y_index - 1, index - 1]
                ):
                    index -= 1

    def maximum_path(neg_cent, mask):
        import numpy as np  # type: ignore
        import torch  # type: ignore

        device = neg_cent.device
        dtype = neg_cent.dtype
        neg_cent_np = neg_cent.detach().cpu().numpy().astype(np.float32)
        path = np.zeros(neg_cent_np.shape, dtype=np.int32)
        t_t_max = mask.sum(1)[:, 0].detach().cpu().numpy().astype(np.int32)
        t_s_max = mask.sum(2)[:, 0].detach().cpu().numpy().astype(np.int32)
        maximum_path_jit(path, neg_cent_np, t_t_max, t_s_max)
        return torch.from_numpy(path).to(device=device, dtype=dtype)

    module.maximum_path = maximum_path
    module.maximum_path_jit = maximum_path_jit
    sys.modules["melo.monotonic_align"] = module
    sys.modules["melo.monotonic_align.core"] = module


def _install_librosa_shim() -> None:
    librosa_module = types.ModuleType("librosa")
    filters_module = types.ModuleType("librosa.filters")
    util_module = types.ModuleType("librosa.util")

    filters_module.mel = _librosa_mel
    util_module.pad_center = _pad_center
    librosa_module.filters = filters_module
    librosa_module.util = util_module
    librosa_module.load = _load_audio
    sys.modules["librosa"] = librosa_module
    sys.modules["librosa.filters"] = filters_module
    sys.modules["librosa.util"] = util_module


def _install_torchaudio_shim() -> None:
    torchaudio_module = types.ModuleType("torchaudio")
    functional_module = types.ModuleType("torchaudio.functional")
    torchaudio_module.load = _torchaudio_load
    torchaudio_module.functional = functional_module
    sys.modules["torchaudio"] = torchaudio_module
    sys.modules["torchaudio.functional"] = functional_module


def _install_transformers_shim() -> None:
    transformers_module = types.ModuleType("transformers")

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(model_id: str):
            _log(f"using lightweight tokenizer shim for {model_id}")
            return _SimpleTokenizer()

    class AutoModelForMaskedLM:
        @staticmethod
        def from_pretrained(model_id: str):
            raise RuntimeError(
                f"Transformers model loading is disabled in the OpenVoice runner ({model_id})."
            )

    transformers_module.AutoTokenizer = AutoTokenizer
    transformers_module.AutoModelForMaskedLM = AutoModelForMaskedLM
    sys.modules["transformers"] = transformers_module


def _install_melo_download_shim(melo_language: str, melo_model_dir: Path) -> None:
    module = types.ModuleType("melo.download_utils")

    def load_or_download_config(locale, use_hf=True, config_path=None):
        del locale, use_hf
        path = Path(config_path) if config_path else melo_model_dir / "config.json"
        _require_melo_file(path, melo_language, "config")
        from melo import utils  # type: ignore

        return utils.get_hparams_from_file(str(path))

    def load_or_download_model(locale, device, use_hf=True, ckpt_path=None):
        del locale, use_hf
        path = Path(ckpt_path) if ckpt_path else melo_model_dir / "checkpoint.pth"
        _require_melo_file(path, melo_language, "checkpoint")
        import torch  # type: ignore

        return torch.load(str(path), map_location=device)

    module.load_or_download_config = load_or_download_config
    module.load_or_download_model = load_or_download_model
    module.load_pretrain_model = lambda: []
    sys.modules["melo.download_utils"] = module


def _install_openvoice_text_shim() -> None:
    module = types.ModuleType("openvoice.text")

    def text_to_sequence(*_: object, **__: object) -> list[int]:
        raise RuntimeError("openvoice.text is stubbed because MeloTTS supplies the base audio in this runner.")

    module.text_to_sequence = text_to_sequence
    sys.modules["openvoice.text"] = module


def _install_wavmark_shim() -> None:
    module = types.ModuleType("wavmark")
    module.load_model = lambda: _NoWatermarkModel()
    sys.modules["wavmark"] = module


def _require_melo_file(path: Path, melo_language: str, kind: str) -> None:
    if path.exists():
        return
    raise RuntimeError(
        f"Missing local MeloTTS {kind} for {melo_language}: {path}. "
        "Download config.json and checkpoint.pth from the matching myshell-ai/MeloTTS Hugging Face repo "
        "into this directory before running OpenVoice synthesis."
    )


class _SimpleTokenizer:
    def tokenize(self, text: str) -> list[str]:
        import re

        return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\w\s]", text.lower())

    def __call__(self, text: str, return_tensors: str | None = None):
        import torch  # type: ignore

        token_count = len(self.tokenize(text)) + 2
        return {"input_ids": torch.arange(token_count, dtype=torch.long).unsqueeze(0)}


def _install_melo_text_language_shims(melo_language: str) -> None:
    target = melo_language.split("_")[0]
    module_names = {
        "ZH": "chinese",
        "JP": "japanese",
        "EN": "english",
        "KR": "korean",
        "FR": "french",
        "ES": "spanish",
    }
    for code, module_name in module_names.items():
        if code == target and code != "EN":
            continue
        if code == "EN":
            stub = _english_language_stub()
        else:
            stub = _language_stub(f"melo.text.{module_name}")
            if module_name == "japanese":
                stub.distribute_phone = _distribute_phone
        sys.modules[f"melo.text.{module_name}"] = stub
    sys.modules["melo.text.chinese_mix"] = sys.modules.get(
        "melo.text.chinese_mix", _language_stub("melo.text.chinese_mix")
    )


def _language_stub(name: str):
    stub = types.ModuleType(name)
    stub.text_normalize = lambda text: text
    stub.g2p = _unsupported_g2p
    stub.get_bert_feature = _unsupported_bert
    return stub


def _english_language_stub():
    stub = types.ModuleType("melo.text.english")
    stub.text_normalize = _english_text_normalize
    stub.g2p = _english_g2p
    stub.get_bert_feature = _unsupported_bert
    return stub


def _english_text_normalize(text: str) -> str:
    return text.lower().strip()


def _english_g2p(text: str, pad_start_end: bool = True, tokenized: list[str] | None = None):
    tokens = tokenized if tokenized is not None else _SimpleTokenizer().tokenize(text)
    phones: list[str] = []
    tones: list[int] = []
    word2ph: list[int] = []
    for token in tokens:
        token_phones = _token_to_phones(token)
        if not token_phones:
            continue
        phones.extend(token_phones)
        tones.extend([0] * len(token_phones))
        word2ph.extend([1] * len(token_phones))
    if not phones:
        phones = ["UNK"]
        tones = [0]
        word2ph = [1]
    if pad_start_end:
        phones = ["_"] + phones + ["_"]
        tones = [0] + tones + [0]
        word2ph = [1] + word2ph + [1]
    return phones, tones, word2ph


def _token_to_phones(token: str) -> list[str]:
    punctuation = {"!", "?", ",", ".", "'", "-"}
    if token in punctuation:
        return [token]
    letter_map = {
        "a": ["ah"],
        "b": ["b"],
        "c": ["k"],
        "d": ["d"],
        "e": ["eh"],
        "f": ["f"],
        "g": ["g"],
        "h": ["hh"],
        "i": ["ih"],
        "j": ["jh"],
        "k": ["k"],
        "l": ["l"],
        "m": ["m"],
        "n": ["n"],
        "o": ["ow"],
        "p": ["p"],
        "q": ["k"],
        "r": ["r"],
        "s": ["s"],
        "t": ["t"],
        "u": ["uh"],
        "v": ["V"],
        "w": ["w"],
        "x": ["k", "s"],
        "y": ["y"],
        "z": ["z"],
    }
    phones: list[str] = []
    for character in token.lower():
        phones.extend(letter_map.get(character, []))
    return phones


def _unsupported_g2p(*_: object, **__: object):
    raise RuntimeError("This MeloTTS language module was stubbed because it is not used by the current language.")


def _unsupported_bert(*_: object, **__: object):
    raise RuntimeError("BERT feature extraction is disabled for the local OpenVoice runner.")


def _distribute_phone(n_phone: int, n_word: int) -> list[int]:
    phones_per_word = [0] * int(n_word)
    for _ in range(int(n_phone)):
        min_tasks = min(phones_per_word)
        min_index = phones_per_word.index(min_tasks)
        phones_per_word[min_index] += 1
    return phones_per_word


def _load_audio(path: str, sr: int | None = 22050, mono: bool = True, **_: object):
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore

    audio, source_sr = sf.read(path, dtype="float32", always_2d=False)
    if mono and getattr(audio, "ndim", 1) > 1:
        audio = np.mean(audio, axis=1, dtype=np.float32)
    if sr is not None and int(source_sr) != int(sr):
        audio = _resample_audio(audio, int(source_sr), int(sr))
        source_sr = int(sr)
    return np.asarray(audio, dtype=np.float32), int(source_sr)


def _torchaudio_load(
    path: str,
    frame_offset: int = 0,
    num_frames: int = -1,
    normalize: bool = True,
    channels_first: bool = True,
    **_: object,
):
    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    import torch  # type: ignore

    audio, sample_rate = sf.read(path, dtype="float32" if normalize else "int16", always_2d=True)
    if frame_offset:
        audio = audio[frame_offset:]
    if num_frames and num_frames > 0:
        audio = audio[:num_frames]
    if channels_first:
        audio = np.swapaxes(audio, 0, 1)
    return torch.from_numpy(np.asarray(audio, dtype=np.float32)), int(sample_rate)


def _resample_audio(audio, source_sr: int, target_sr: int):
    import numpy as np  # type: ignore

    audio = np.asarray(audio, dtype=np.float32)
    if audio.size == 0:
        return audio
    target_length = max(1, int(round(audio.shape[0] * target_sr / source_sr)))
    source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    if audio.ndim == 1:
        return np.interp(target_positions, source_positions, audio).astype(np.float32)
    channels = [
        np.interp(target_positions, source_positions, audio[:, channel]).astype(np.float32)
        for channel in range(audio.shape[1])
    ]
    return np.stack(channels, axis=1)


def _librosa_mel(sr: int, n_fft: int, n_mels: int = 128, fmin: float = 0.0, fmax: float | None = None, **_: object):
    import numpy as np  # type: ignore

    upper = float(fmax) if fmax is not None else float(sr) / 2.0
    fft_freqs = np.linspace(0.0, float(sr) / 2.0, int(n_fft) // 2 + 1)
    mel_f = _mel_frequencies(int(n_mels) + 2, float(fmin), upper)
    fdiff = np.diff(mel_f)
    ramps = mel_f[:, np.newaxis] - fft_freqs[np.newaxis, :]
    lower = -ramps[:-2] / fdiff[:-1, np.newaxis]
    upper_ramp = ramps[2:] / fdiff[1:, np.newaxis]
    weights = np.maximum(0.0, np.minimum(lower, upper_ramp))
    enorm = 2.0 / (mel_f[2 : int(n_mels) + 2] - mel_f[: int(n_mels)])
    weights *= enorm[:, np.newaxis]
    return weights.astype(np.float32)


def _mel_frequencies(n_mels: int, fmin: float, fmax: float):
    import numpy as np  # type: ignore

    min_mel = _hz_to_mel(fmin)
    max_mel = _hz_to_mel(fmax)
    mels = np.linspace(min_mel, max_mel, int(n_mels))
    return _mel_to_hz(mels)


def _hz_to_mel(frequencies):
    import numpy as np  # type: ignore

    scalar_input = np.isscalar(frequencies)
    frequencies = np.asanyarray(frequencies)
    f_sp = 200.0 / 3
    mels = frequencies / f_sp
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = frequencies >= min_log_hz
    if scalar_input:
        return min_log_mel + math.log(float(frequencies) / min_log_hz) / logstep if bool(log_t) else float(mels)
    mels[log_t] = min_log_mel + np.log(frequencies[log_t] / min_log_hz) / logstep
    return mels


def _mel_to_hz(mels):
    import numpy as np  # type: ignore

    scalar_input = np.isscalar(mels)
    mels = np.asanyarray(mels)
    f_sp = 200.0 / 3
    freqs = f_sp * mels
    min_log_hz = 1000.0
    min_log_mel = min_log_hz / f_sp
    logstep = math.log(6.4) / 27.0
    log_t = mels >= min_log_mel
    if scalar_input:
        return min_log_hz * math.exp(logstep * (float(mels) - min_log_mel)) if bool(log_t) else float(freqs)
    freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))
    return freqs


def _pad_center(data, size: int, axis: int = -1, **_: object):
    length = int(data.shape[axis])
    if size < length:
        raise ValueError(f"Target size ({size}) must be at least input size ({length})")
    left = (size - length) // 2
    right = size - length - left
    if hasattr(data, "dim"):
        if data.dim() != 1 or axis not in (-1, 0):
            raise ValueError("Torch pad_center shim currently supports one-dimensional tensors only")
        import torch.nn.functional as functional  # type: ignore

        return functional.pad(data, (left, right))

    import numpy as np  # type: ignore

    pad_width = [(0, 0)] * data.ndim
    pad_width[axis] = (left, right)
    return np.pad(data, pad_width, mode="constant")


def _source_se_path(checkpoint_dir: Path, speaker_key: str) -> Path:
    ses_dir = checkpoint_dir / "base_speakers" / "ses"
    candidates = [
        ses_dir / f"{speaker_key}.pth",
        ses_dir / f"{speaker_key.lower()}.pth",
        ses_dir / f"{speaker_key.replace('_', '-').lower()}.pth",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    available = ", ".join(path.name for path in sorted(ses_dir.glob("*.pth")))
    raise RuntimeError(f"No source speaker embedding found for {speaker_key}. Available: {available}")


if __name__ == "__main__":
    raise SystemExit(main())

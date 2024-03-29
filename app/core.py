import os
from io import StringIO
from threading import Lock
from typing import Union, BinaryIO

import torch
from faster_whisper import WhisperModel

from .utils import ResultWriter, WriteTXT, WriteSRT, WriteVTT, WriteTSV, WriteJSON


# ASR_MODELs: tiny, base, small, medium, large (only OpenAI Whisper), large-v1, large-v2 and large-v3, distil-large-v2
model_name = os.getenv("ASR_MODEL", "distil-large-v2")
model_path = os.getenv("ASR_MODEL_PATH", os.path.join(
    os.path.expanduser("~"), ".cache", "distil-large-v2"))

# More about available quantization levels is here:
#   https://opennmt.net/CTranslate2/quantization.html
if torch.cuda.is_available():
    device = "cuda"
    model_quantization = os.getenv("ASR_QUANTIZATION", "int8")
else:
    device = "cpu"
    model_quantization = os.getenv("ASR_QUANTIZATION", "int8")

model = WhisperModel(
    model_size_or_path=model_name,
    device=device,
    compute_type=model_quantization,
    download_root=model_path,
)

model_lock = Lock()


def transcribe(
        audio,
        task: Union[str, None],
        language: Union[str, None],
        initial_prompt: Union[str, None],
        vad_filter: Union[bool, None],
        word_timestamps: Union[bool, None],
        output,
):
    options_dict = {"task": task}
    if language:
        options_dict["language"] = language
    if initial_prompt:
        options_dict["initial_prompt"] = initial_prompt
    if word_timestamps:
        options_dict["vad_filter"] = True
    if word_timestamps:
        options_dict["word_timestamps"] = True

    options_dict["condition_on_previous_text"] = False
    options_dict["max_new_tokens"] = 128
    with model_lock:
        segments = []
        text = ""
        segment_generator, info = model.transcribe(
            audio, beam_size = 5, **options_dict)
        for segment in segment_generator:
            segments.append(segment)
            text = text + segment.text
        result = {
            "language": options_dict.get("language", info.language),
            "segments": segments,
            "text": text
        }

    output_file = StringIO()
    write_result(result, output_file, output)
    output_file.seek(0)

    return output_file

def write_result(
        result: dict, file: BinaryIO, output: Union[str, None]
):
    if output == "srt":
        WriteSRT(ResultWriter).write_result(result, file=file)
    elif output == "vtt":
        WriteVTT(ResultWriter).write_result(result, file=file)
    elif output == "tsv":
        WriteTSV(ResultWriter).write_result(result, file=file)
    elif output == "json":
        WriteJSON(ResultWriter).write_result(result, file=file)
    elif output == "txt":
        WriteTXT(ResultWriter).write_result(result, file=file)
    else:
        return 'Please select an output method!'

# Wren

**A research series of small (<3B parameter) open-weight speech-text multimodal models.**

Wren explores **unified representations for speech and text** — treating both
modalities as first-class citizens rather than bolting speech onto a language model.
The core research questions are:

- **Unified speech-text modelling** — shared representations that generalise across
  synthesis, recognition, and understanding without task-specific heads
- **Speech representation learning** — learning compact, general-purpose
  representations that capture content, speaker, prosody, and acoustics
- **Speech disentanglement** — explicitly separating and controlling the factors
  of variation in speech (what is said, who says it, how, and where)

Each release is small enough to train and run on a single GPU, with all weights,
training code, and datasets public. The architecture — codec, backbone, token
layout — is treated as a variable, not a commitment.

---

## Models

| Model | Task | Languages | Status |
|---|---|---|---|
| [Wren-TTS-360M-v1](https://huggingface.co/shangeth/Wren-TTS-360M-v1) | Text → Speech | English | ✅ Released |
| [Wren-TTS-0.5B-multi](https://huggingface.co/shangeth/Wren-TTS-0.5B-multi) | Text → Speech (multilingual) | en · de · fr · es · nl · it · pl · pt | ✅ Released |
| Wren-ASR | Speech → Text | — | 📋 Planned |
| Wren-LM | Speech language modelling | — | 📋 Planned |
| Wren-Omni | Unified speech understanding + synthesis | — | 📋 Planned |

## Datasets

Pre-extracted [Kyutai Mimi](https://huggingface.co/kyutai/mimi) codec tokens for the
underlying speech corpora — see [wren-datasets](https://github.com/shangeth/wren-datasets)
for extraction tooling and full details.

| Dataset | Source | Rows | License |
|---|---|---|---|
| [shangeth/ljspeech-mimi-codes](https://huggingface.co/datasets/shangeth/ljspeech-mimi-codes) | LJSpeech | ~13k | CC0 |
| [shangeth/librispeech-mimi-codes](https://huggingface.co/datasets/shangeth/librispeech-mimi-codes) | LibriSpeech | ~280k | CC-BY-4.0 |
| [shangeth/libritts-r-mimi-codes](https://huggingface.co/datasets/shangeth/libritts-r-mimi-codes) | LibriTTS-R | ~360k | CC-BY-4.0 |
| [shangeth/hifi-tts-mimi-codes](https://huggingface.co/datasets/shangeth/hifi-tts-mimi-codes) | HiFi-TTS | ~290k | CC-BY-4.0 |
| [shangeth/vctk-mimi-codes](https://huggingface.co/datasets/shangeth/vctk-mimi-codes) | VCTK | ~44k | CC-BY-4.0 |
| [shangeth/jenny-mimi-codes](https://huggingface.co/datasets/shangeth/jenny-mimi-codes) | Jenny TTS | ~21k | Apache-2.0 |
| [shangeth/expresso-mimi-codes](https://huggingface.co/datasets/shangeth/expresso-mimi-codes) | Expresso (conversational) | ~40k | **CC-BY-NC-4.0** |
| [shangeth/mls-mimi-codes](https://huggingface.co/datasets/shangeth/mls-mimi-codes) | Multilingual LibriSpeech (7 langs) | ~6M | CC-BY-4.0 |

---

## Quickstart (v1 TTS)

```bash
pip install torch torchaudio transformers
```

```python
import torch
from transformers import AutoModel, AutoProcessor

model_id  = "shangeth/Wren-TTS-360M-v1"
device    = "cuda" if torch.cuda.is_available() else "cpu"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model     = AutoModel.from_pretrained(model_id, trust_remote_code=True).to(device).eval()

inputs   = processor("Hello world, how are you today?")
inputs   = {k: v.to(device) for k, v in inputs.items()}
waveform = model.generate(**inputs, output_audio=True, max_audio_frames=200)
processor.save_audio(waveform, "out.wav")
```

---

## v1 Architecture

The first release models speech and text in a shared discrete token sequence —
speech is encoded into neural codec tokens, interleaved with text tokens, and
processed by a single autoregressive model:

```
text ──► [shared token sequence] ──► audio codec tokens ──► decoded audio
audio ──► codec encoder ──► [shared token sequence] ──► text   (ASR, coming)
```

**v1 implementation choices** (not fixed for future releases):

| Component | v1 (English) | v1-multi (multilingual) | Alternatives being explored |
|---|---|---|---|
| LLM backbone | SmolLM2-360M | Qwen2.5-0.5B | Phi, Llama variants, custom |
| Audio codec | Kyutai Mimi | Kyutai Mimi | EnCodec, DAC, SoundStream, learned |
| Token layout | MusicGen-style delay | MusicGen-style delay | Flat interleaved, parallel, hierarchical |
| Conditioning | Reference-audio prefix | Reference-audio prefix | Speaker embedding, style tokens |

---

## Research directions

### Unified speech-text representations
Building a shared representation space where speech and text are not
separate modalities with bridges between them, but different views of the
same underlying meaning. Investigating what token layouts, training objectives,
and architectures encourage this kind of alignment at the representation level.

### Speech representation learning
Learning representations that capture what matters for downstream tasks —
not just what a codec was trained to reconstruct. What do models learn about
phonetics, prosody, and speaker identity as a side effect of sequence modelling?
Can those representations transfer across tasks and domains?

### Speech disentanglement
Explicitly separating the factors of variation in speech: *what* is said
(content), *who* says it (speaker), *how* it is said (prosody, style), and
*where* it was recorded (acoustics). Target applications: voice conversion,
expressive TTS, accent adaptation, privacy-preserving speech models.

### Architecture exploration
Treating the codec, backbone, and token layout as independent variables.
Does discrete tokenisation generalise better than continuous representations?
What compression rate is optimal for modelling? How does the choice of codec
affect what the model can and cannot disentangle?

---

## Repositories

| Repo | Purpose |
|---|---|
| [wren-tts](https://github.com/shangeth/wren-tts) | TTS training, evaluation, HF publishing |
| [wren-datasets](https://github.com/shangeth/wren-datasets) | Codec-token extraction and HF dataset publishing |
| [wren-asr](https://github.com/shangeth/wren-asr) | *(coming)* ASR |
| [wren-llm](https://github.com/shangeth/wren-llm) | *(coming)* Speech language modelling |

---

## Design principles

- **Speech and text as equals.** Neither modality is an afterthought bolted onto the other.
- **Small by design.** All models target < 3B parameters — trainable and runnable on a single GPU.
- **Open everything.** Weights, training code, datasets, evaluation pipelines.
- **Architecture as variable.** Codec, backbone, and token layout are research choices, not fixed infrastructure.
- **Reproducible.** Every release ships the training config and evaluation script so published numbers can be verified.

---

## Citation

```bibtex
@misc{wren2026,
  title  = {Wren: A Family of Small Open-Weight Models for Unified Speech-Text Modelling},
  author = {Shangeth Rajaa},
  year   = {2026},
  url    = {https://github.com/shangeth/wren}
}
```

---

## License

Apache-2.0. See individual model and dataset cards for upstream component licenses.

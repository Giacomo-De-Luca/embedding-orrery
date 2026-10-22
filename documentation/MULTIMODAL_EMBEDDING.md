# Multimodal embedding (Image / Audio / Video / Mixed)

`google/embeddinggemma-2` (EmbeddingGemma 2, "EG2", Apache 2.0, not gated) maps text, images,
audio and video into one 768-d space, including mixed inputs such as a caption with its image.
The Collections page's **"Image / Audio / Video / Mixed"** data type embeds such datasets, from
local files and from Hugging Face datasets. A text query then searches them like any text
collection.

It replaces the old image mode, which used ViT (`google/vit-base-patch16-384`). That mode only
read local files, ignored the selected model, recorded no provider (so semantic search on it
failed), and gave vectors that shared no space with text.

## Data type and inputs

| Layer | Name |
|---|---|
| GraphQL | `DataTypeEnum.MULTIMODAL` on `EmbedLocalFileInput` and `EmbedDatasetInput` |
| Stored | `data_type: "multimodal"` (DuckDB `datasets.data_type`, ChromaDB collection metadata) |
| Media columns | `mediaColumns: [String!]` (image, audio or video columns) |
| Text columns | `columns` + `textTemplate`, optional: embedded together with the media |
| Per-job options | `mediaOptions: MediaOptionsInput` (omitted values keep the defaults) |

- **The model:** an empty model name means the catalog's multimodal default
  (`default_for: ["multimodal"]` in `config/embedding_models.json`: EG2). Only
  sentence-transformers models can embed media; the factory rejects any other provider.
- **Re-embedding a multimodal dataset is refused:** its stored documents are its text, or
  file-name labels for media-only rows, so a text re-embed would silently drop the media. Embed
  the source file or dataset again instead.

## Pipeline

`MultimodalEmbeddingRun` (`backend/embedding_functions/embed_media.py`) runs the multimodal jobs
of both `embed_local_file` and `embed_huggingface`, on the shared `embedding_job` loop (progress,
cancel, resume):

1. **Detect** each media column's kind (`MediaColumn.detect`): the dataset's declared Image /
   Audio / Video types first (HF jobs, `get_media_column_kinds`), then magic bytes (PNG, JPEG, GIF,
   BMP, WEBP, RIFF WAVE/AVI, ID3/MP3, FLAC, Ogg, MP4/MOV `ftyp` brands, WebM/MKV), then file
   extensions, by majority over the first rows (`utils/media_kinds.py`). A generic ISO brand
   (`isom`, `mp42`, `dash`…) is shared by `.m4a` audio and `.mp4` video, so the cell's file
   extension decides; without one it reads as video. Folder-based HF datasets declare no types,
   but their cells carry file paths.
2. **Check the token budget** (below) before any model loads.
3. **Load the model** with only the encoder towers those kinds need, and check that it supports
   them (`require_modalities`).
4. **Per batch,** decode each row's media (`utils/media_decoding.py`), build one input dict with
   its text (`utils/media_rows.py`) and embed (`embed_inputs`).
5. **Store** the vectors in ChromaDB and the documents in DuckDB. In the item metadata, each
   media cell is replaced by its path or a placeholder such as `[image 34 KB]`.

Rows whose media cannot be decoded are skipped. The count, with the first failure, is reported as
one entry in the result's `warnings`. Rows keep their source order (no length sort), so ids stay
stable when a job is resumed.

### Cell shapes

`MediaDecoder` accepts raw bytes (parquet), file paths (relative paths resolve against the data
file's folder), `http(s)` URLs when `allow_remote_urls` is on, Hugging Face `{"bytes", "path"}`
dicts, decoded `{"array", "sampling_rate"}` audio, and PIL images. Streamed HF datasets
(the "first N rows" portion) point cells at the repo's own files (`hf://…`, `zip://…::https://…`,
`https://huggingface.co/…`): HF jobs read them through datasets' own `xopen`
(`HFMediaFeatures.read_file`, the `open_path` reader), without the `allow_remote_urls` opt-in,
which is for URLs in your own data. It returns:

| Kind | Model input |
|---|---|
| Image | RGB `PIL.Image` |
| Audio | `{"array": float32 mono, "sampling_rate": 16000}`, capped at `max_audio_seconds`. Files are decoded with PyAV (its wheels bundle FFmpeg, so mp3/m4a/aac need no system install) and every waveform is resampled to 16 kHz: sentence-transformers passes arrays through unresampled and the EG2 feature extractor does not check the rate |
| Video | `{"array": uint8 (frames, H, W, 3), "video_metadata": …}`: frames sampled at `video_fps`, at most `video_max_frames` spread evenly over longer clips, downscaled to `video_frame_max_side`. Only the pictures are embedded, not the soundtrack |

Hugging Face Image, Audio and Video features are always read with `decode=False`
(`HFMediaFeatures`), in every mode. `datasets` 4.x decodes Audio and Video with torchcodec,
which needs a system FFmpeg and is not installed, so such datasets used to fail to load even in
text mode. In a text-mode HF job, a media column (or any bytes cell) that lands in the metadata is
stored as its path or a placeholder such as `[image 34 KB]` or `[binary 2 KB]`
(`extract_metadata` → `MediaReference.describe`); its bytes used to crash `json.dumps`. Local text
jobs still turn such cells into a `str(bytes)` dump before that point (pre-existing,
`local_data_client.load_local_file_portion`).

### Input order and prompts

One row becomes one input dict with the keys in a fixed order: text, image, video, audio
(sentence-transformers follows dict key order). The task prefix (see
`EMBEDDING_TASK_PREFIXES.md`) or the sentence-transformers prompt applies to the **text only**:
`embed_inputs` encodes rows with text and media-only rows in separate calls, and media never get
a prefix. Media-only rows are stored with a label document such as `image: cat.jpg`.

## Encoder towers

EG2's `config.json` has a `vision_config` and an `audio_config`. **Video runs on the vision tower**
(its frames go through it), so there is no separate video tower. A tower is skipped by setting its
config to `None` through `config_kwargs` (`SentenceTransformerLoadOptions`,
`utils/model_load_options.py`), driven by the catalog's `encoders` map
(`image → vision_config`, `video → vision_config`, `audio → audio_config`):

| Data | Towers loaded | `config_kwargs` | Params |
|---|---|---|---|
| Text dataset, and every text query | text | `{"vision_config": None, "audio_config": None}` | ~270M |
| Image and/or video columns | text + vision | `{"audio_config": None}` | ~440M |
| Audio columns | text + audio | `{"vision_config": None}` | ~570M |
| Image/video + audio columns | all | `{}` | ~740M |

The sentence-transformers model cache keeps one resident model. A cached load that already
covers the needed towers is reused (a text query after a full media job), otherwise the model
reloads with only the towers needed. The EG2 processor always builds its video sub-processor,
which requires **torchvision**, so torchvision is a dependency even for text-only loads.

## Media settings and token budget

`config/media_embedding.json` (`MediaEmbeddingSettings`, `utils/media_settings.py`) holds the
defaults and limits. The `mediaEmbeddingSettings` query serves them (plus the default model) to
the embed form, which shows only the fields of the selected kinds and sends only the values that
differ from the defaults. Overrides are validated server-side, kept in the job's stored input (so
a resume replays them) and recorded in the collection metadata (`media_options`).

| Option | Default | Limit |
|---|---|---|
| `image_max_soft_tokens` | 280 | one of 70, 140, 280, 560, 1120 |
| `video_fps` | 1 | `video_fps_limit` 4 |
| `video_max_frames` | 32 | `video_frames_limit` 128 |
| `video_max_soft_tokens` (per frame) | 140 | one of 70, 140, 280, 560, 1120 |
| `max_audio_seconds` | 120 | `audio_seconds_limit` 327 |

Fixed settings: `audio_sample_rate` 16000, `video_frame_max_side` 768, `rows_per_batch` 16
(the job's batch size is capped at it), `encode_batch_size` 8, `detection_sample_rows` 20,
`allow_remote_urls` false, `remote_timeout_seconds` 10, and optional per-kind extension lists.

EG2 has one 8192-token context per input, shared by all its parts. The worst case per row is

```
reserved_tokens (512)
+ image columns × image_max_soft_tokens
+ video columns × video_max_frames × video_max_soft_tokens
+ audio columns × ceil(max_audio_seconds × 25)
```

and a job whose estimate exceeds `context_tokens` (8192) is rejected before the model loads, with
the estimate in the message. The text's share is what the worst-case media leave
(`text_token_budget`: `context_tokens` minus the media terms, never below `reserved_tokens`):
the sentence-transformers wrapper cuts longer text to it with the model's tokenizer, prompt
included (`cap_text_tokens`), and the cut rows are reported as a truncation warning. Without the
cut, a long caption plus a video and an audio clip could overflow the context and fail the batch. The form shows the same estimate ("≈ N of 8192 tokens per row").
At the defaults one video column needs about 4,992 tokens and one audio column 3,512, but an
image, a video and an audio column together need 8,272: lower the frames, the tokens per frame
or the audio seconds to mix all three.

## Limitations

- **Memory:** the loaders hold whole rows in memory, bytes included; use the portion and sample
  caps on large media sets. Lazy row loading is a follow-up.
- **Video is slow:** at the defaults a clip is about 32 × 140 tokens, far more than an image.
- **Topics on media-only collections are weak:** topic keywords and LLM labels only see the
  file-name labels.
- **Uploaded CSVs lose the files next to them:** use a file path, or parquet with the bytes.
- **Alternating full and text-only EG2 loads reloads the model;** reusing a covering load avoids
  most of this.

## Tests

`unit_tests/test_media_kinds.py`, `test_media_decoding.py` (a tiny MPEG-4 clip built with PyAV),
`test_media_rows.py`, `test_media_settings.py`, `test_media_converters.py`,
`test_huggingface_media.py`, `test_embed_media.py` (fake client and embedder: batching, cancel,
resume, skipped rows, towers requested), `test_text_processing.py` (media cells in metadata),
`test_job_cancellation.py::TestReEmbedMultimodalSource`, and `test_dependency_surface.py`
(transformers ≥ 5.19 with `embedding_gemma2`, sentence-transformers ≥ 6.1, torchvision, PyAV).
None of them downloads a model.

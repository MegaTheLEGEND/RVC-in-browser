# Notice for humans from a human:

I directed an AI agent to create this software. I like to disclose when i make something using AI, this is one of those things. I was curious if RVC could run in the browser. turns out it can.

# RVC in the Browser

A voice changer that runs entirely in your web browser. You give it a recording (or your microphone) and an RVC voice model, and it converts the speech so it sounds like that voice. Nothing is uploaded anywhere: all the processing happens on your own computer.

It is a browser port of the **inference** half of [Retrieval-based Voice Conversion (RVC)](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI). Training new voices is not included. (yet)

## What it does

1. **Loads your models.** You pick the files in the page:
   - **HuBERT**: the speech-feature model (`pytorch_model.bin` or `model.safetensors`, or a ready-made `.onnx`), plus an optional `config.json`.
   - **Pitch model (optional)**: `rmvpe.pt` or `rmvpe.onnx`. Without it you can use the built-in YIN pitch tracker.
   - **Voice**: your RVC `.pth` file (v1 or v2, with or without pitch).
   - **Index (optional)**: the voice's faiss `.index` file, which makes the result sound closer to the target voice.
2. **Converts audio.** Choose an audio or video file and press Convert. The speech is turned into features, the pitch is found, features are optionally blended with the closest matches from the index, and the voice model generates the new audio.
3. **Gives you the result.** Play it in the page or download it as a WAV file.
4. **Realtime mode.** Converts your microphone live and plays it back. Use headphones to avoid feedback.

Everything is built in the browser: the voice `.pth` and the HuBERT/RMVPE weights are turned into ONNX graphs by the JavaScript code, so no Python or model export step is needed. Models you load are cached in the browser (IndexedDB) so you don't have to pick them again.

## Settings

| Setting | What it does |
|---|---|
| Transpose | Shifts pitch in semitones (-24 to +24). |
| Pitch extractor | RMVPE (best quality) or YIN (no extra model needed). |
| Speaker ID | Chooses the speaker in multi-speaker models. |
| Index rate | How strongly to pull the result toward the voice's index (default 0.75). |
| Protect voiceless | Keeps consonants and breaths from being distorted (default 0.33). |
| Volume envelope mix | How much of the original loudness shape to keep (default 0.25). |
| Output sample rate | Model native, 44.1 kHz or 48 kHz. |
| Runtime | WASM (CPU, most compatible) or WebGPU for HuBERT (experimental). |
| Realtime context | How much past audio the live mode looks at; more is better quality but slower. |

## Running it

You need Python 3 only to serve the files (browsers won't run ES modules or workers from `file://`):

    python serve.py          # then open http://localhost:8000

`serve.py` adds the headers that let onnxruntime-web use multi-threaded WASM, which is noticeably faster. Any static server works, but without those headers it runs single-threaded.

Then in the page: add HuBERT, optionally RMVPE, pick your voice `.pth` (and `.index`), choose an audio file, and press Convert.

## Files

    index.html        the page
    styles.css        styling
    serve.py          local server with the required headers
    src/
      main.js         UI wiring, file loading, model cache hookup
      worker.js       inference worker (keeps the page responsive)
      pipeline.js     the conversion pipeline (chunking, pitch, index blend, synth, output)
      realtime.js     microphone to speakers live mode
      stitch.js       crossfades live blocks together
      hubert.js       runs the HuBERT feature model
      rmvpe.js, yin.js    the two pitch extractors
      synth.js        runs the voice (generator) model
      faiss.js        reads faiss .index files and does the retrieval blend
      torchpkl.js     reads PyTorch checkpoints (.pth/.pt/.bin) in the browser
      build_hubert.js, build_rmvpe.js, build_synth.js   build the ONNX models from weights
      onnxwriter.js, nn.js    helpers for writing ONNX graphs
      dsp.js          resampling, filters, mel spectrogram, volume matching
      audio.js        decoding input audio and writing WAV
      cache.js        IndexedDB model cache
      ort.js, config.js   onnxruntime-web loading and settings
    vendor/ort/       bundled onnxruntime-web 1.22.0 (WASM), so it works offline

## Differences from the Python RVC app

- Resampling uses a windowed-sinc resampler instead of soxr.
- Index search is exact rather than approximate, so results can differ very slightly.
- Random noise comes from a seeded generator, so output is reproducible.
- Pitch methods are RMVPE and YIN only (no `pm` or FCPE).
- Output is WAV only.
- WebGPU is experimental and loads onnxruntime-web from the jsDelivr CDN (pinned version) since only the WASM build is bundled here.

## Not included

Training, vocal/instrument separation (UVR), the realtime VST plugin and the Python/Gradio UI from the original project.

## License

MIT, same as the original project (see `LICENSE`).

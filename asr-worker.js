// Runs Whisper transcription off the main thread so the page stays responsive
// (video preview, buttons, the elapsed-time ticker) while the model downloads and runs.
let pipe = null;
let pipeModel = null;

self.onmessage = async (e) => {
  const { type, modelSize, pcm } = e.data;
  if (type !== 'transcribe') return;

  try {
    const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2');
    // Multi-threaded WASM needs cross-origin-isolation headers most static hosts can't set —
    // force single-threaded so the CPU backend actually works on plain static hosting.
    try { env.backends.onnx.wasm.numThreads = 1; } catch (e) {}

    if (!pipe || pipeModel !== modelSize) {
      const progressCb = (p) => {
        if (p.status === 'progress') {
          self.postMessage({ type: 'progress', progress: p.progress, file: p.file });
        }
      };
      let lastErr = null;
      // Skip WebGPU (many devices, especially integrated graphics, don't support it) and
      // try the CPU-backed device identifiers this library version might expect.
      for (const dev of ['cpu', 'wasm', undefined]) {
        try {
          pipe = await pipeline(
            'automatic-speech-recognition',
            'Xenova/whisper-' + modelSize,
            dev ? { device: dev, progress_callback: progressCb } : { progress_callback: progressCb }
          );
          lastErr = null;
          break;
        } catch (err) {
          lastErr = err;
        }
      }
      if (lastErr) throw lastErr;
      pipeModel = modelSize;
    }

    self.postMessage({ type: 'status', text: 'Transcribing…' });
    const result = await pipe(pcm, { return_timestamps: 'word', chunk_length_s: 30, stride_length_s: 5 });
    self.postMessage({ type: 'done', result });
  } catch (err) {
    self.postMessage({ type: 'error', message: (err && err.message) ? err.message : String(err) });
  }
};

// Runs Whisper transcription off the main thread so the page stays responsive
// (video preview, buttons, the elapsed-time ticker) while the model downloads and runs.
//
// Two engine versions are available:
//   stable = transformers.js 3.0.2 (the one that has been working)
//   next   = transformers.js 4.3.1 (newer, opt-in, experimental)
const ENGINES = {
  stable: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.0.2',
  next: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1/dist/transformers.web.min.js'
};

let pipe = null;
let pipeKey = null;

// Word-level timing depends on a fragile alignment step. There is a known bug in this library where
// word timing with 30-second chunks gives every word in a chunk the same timestamp. If the result looks
// like that (many words sharing one timestamp, zero-length words, or times going backwards) we detect it.
function timingLooksValid(chunks) {
  if (!Array.isArray(chunks) || chunks.length === 0) return false;
  const n = chunks.length;
  let zero = 0, backwards = 0, prev = -Infinity;
  const starts = new Set();
  for (const c of chunks) {
    const ts = c.timestamp || [];
    const s = ts[0], e = ts[1];
    if (s == null) { zero++; continue; }
    if (e == null || e - s < 0.02) zero++;
    if (s < prev - 0.05) backwards++;
    prev = s;
    starts.add(s.toFixed(2));
  }
  if (n < 4) return zero <= 1;
  return zero / n < 0.3 && backwards / n < 0.1 && starts.size / n > 0.6;
}

self.onmessage = async (e) => {
  const { type, modelSize, pcm, timing } = e.data;
  if (type !== 'transcribe') return;
  const engine = e.data.engine === 'next' ? 'next' : 'stable';

  try {
    const { pipeline, env } = await import(ENGINES[engine]);
    // Multi-threaded WASM needs cross-origin-isolation headers most static hosts can't set,
    // so force single-threaded to make sure the CPU backend works on plain static hosting.
    try { env.backends.onnx.wasm.numThreads = 1; } catch (err) {}

    const key = engine + '|' + modelSize;
    if (!pipe || pipeKey !== key) {
      const progressCb = (p) => {
        if (p.status === 'progress') {
          self.postMessage({ type: 'progress', progress: p.progress, file: p.file });
        }
      };
      let lastErr = null;
      // Skip WebGPU (many devices, especially integrated graphics, don't support it, and some
      // GPU/driver combinations give corrupted output) and use the CPU-backed device names.
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
      pipeKey = key;
    }

    // Word timing uses 29-second windows (one reported workaround for the same-timestamp bug);
    // phrase timing uses the normal 30-second windows.
    const runPass = (m) => pipe(pcm, {
      return_timestamps: m === 'word' ? 'word' : true,
      chunk_length_s: m === 'word' ? 29 : 30,
      stride_length_s: 5
    });

    let mode = (timing === 'word') ? 'word' : 'segment';
    self.postMessage({ type: 'status', text: mode === 'word' ? 'Transcribing with word timing…' : 'Transcribing with phrase timing…' });
    let result = await runPass(mode);
    let ok = timingLooksValid(result.chunks);

    if (mode === 'word' && !ok) {
      // Word timing is unusable for this clip: redo it with phrase timing, which is much more dependable.
      self.postMessage({ type: 'status', text: 'Word timing came back unreliable for this clip, redoing with phrase timing (more dependable)…' });
      mode = 'segment';
      result = await runPass('segment');
      ok = timingLooksValid(result.chunks);
    }
    self.postMessage({ type: 'done', result, mode, timingOk: ok, engine });
  } catch (err) {
    let message = (err && err.message) ? err.message : String(err);
    if (engine === 'next') message += ' (The newer engine is experimental. Switch Engine back to Stable and try again.)';
    self.postMessage({ type: 'error', message });
  }
};

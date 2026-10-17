// Runs Whisper transcription off the main thread so the page stays responsive
// (video preview, buttons, the elapsed-time ticker) while the model downloads and runs.
let pipe = null;
let pipeModel = null;

// Word-level timing depends on a fragile alignment step. If it comes back unusable (many words
// sharing one timestamp, zero-length words, or times going backwards) we detect that here.
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
  // Many words sharing the same start time means the aligner collapsed them.
  return zero / n < 0.3 && backwards / n < 0.1 && starts.size / n > 0.6;
}

self.onmessage = async (e) => {
  const { type, modelSize, pcm, timing } = e.data;
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

    const runPass = (m) => pipe(pcm, {
      return_timestamps: m === 'word' ? 'word' : true,
      chunk_length_s: 30,
      stride_length_s: 5
    });

    let mode = (timing === 'segment') ? 'segment' : 'word';
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
    self.postMessage({ type: 'done', result, mode, timingOk: ok });
  } catch (err) {
    self.postMessage({ type: 'error', message: (err && err.message) ? err.message : String(err) });
  }
};

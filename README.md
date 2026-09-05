# Caption Deck

A free, browser-only tool for auto-styled video captions — import a transcript or subtitle file (or transcribe audio right in the browser with Whisper), translate it, style it as animated karaoke-style captions, and burn it into a downloadable video. No server, no account, no upload of your video anywhere.

## Features

- **Load video** — drag & drop a file, or paste a direct video URL
- **Get captions** — import `.srt`/`.vtt`, paste a plain transcript (auto-timed against your video), or transcribe audio in-browser using Whisper (via [transformers.js](https://github.com/huggingface/transformers.js))
- **Translate** — free, no API key, translates every caption into your chosen language
- **Style** — Opus-Clip-style word-by-word karaoke captions: position, highlight color, words-shown-at-once, uppercase toggle
- **Sync tools** — shift all caption timing by a fixed offset if it's out of sync
- **Export** — download `.srt`/`.vtt` (original, translated, or bilingual), or burn captions directly into the video and download as MP4/WebM

## Usage

Just open `index.html` in a modern browser (Chrome or Edge recommended for full feature support, including the burned-in video export and in-browser transcription). No build step, no install.

To host it as a live site, deploy this repo as-is to Netlify, Vercel, GitHub Pages, or any static host — `index.html` is the entry point.

## Notes & limitations

- YouTube/TikTok/Instagram links can't be loaded directly — download the video file first, then upload it here. No browser-only tool can pull video from those platforms without breaking their Terms of Service.
- Burned-in video export works most reliably on locally uploaded files. A remote URL may fail to export if that server doesn't send proper CORS headers (it will still preview fine either way).
- In-browser transcription downloads a Whisper model on first use and runs on your device — it can be slow (sometimes much slower than the video's own runtime) without GPU acceleration. For long videos, a dedicated desktop app such as [Subtitle Edit](https://www.nikse.dk/subtitleedit) (free, Windows) will be faster.
- Video export prefers MP4 where the browser supports it, falling back to WebM otherwise.

## License

Personal project — do whatever you'd like with it.

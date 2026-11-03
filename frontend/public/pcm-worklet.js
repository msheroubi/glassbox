// AudioWorklet: microphone float32 at the context rate -> 16 kHz mono PCM16, posted in ~100 ms chunks
// (the format Muse Voice Transcribe expects over /ws/voice).
const TARGET_RATE = 16000;
const CHUNK = 1600;

class PCM16Writer extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / TARGET_RATE;
    // one-pole low-pass at ~7 kHz to limit aliasing before decimation
    this.alpha = 1 - Math.exp((-2 * Math.PI * 7000) / sampleRate);
    this.lp = 0;
    this.prev = 0;
    this.pos = 0;
    this.buf = new Int16Array(CHUNK);
    this.n = 0;
    this.port.onmessage = (e) => {
      if (e.data === "flush") this.flush();
    };
  }

  flush() {
    if (this.n > 0) {
      const out = this.buf.slice(0, this.n);
      this.port.postMessage(out.buffer, [out.buffer]);
      this.n = 0;
    }
  }

  emit(v) {
    const s = Math.max(-1, Math.min(1, v));
    this.buf[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.n === CHUNK) this.flush();
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || input.length === 0) return true;
    const ch = input[0];
    const len = ch.length;
    const filtered = new Float32Array(len);
    let lp = this.lp;
    for (let i = 0; i < len; i++) {
      lp += this.alpha * (ch[i] - lp);
      filtered[i] = lp;
    }
    this.lp = lp;
    // sample positions are relative to this block; index -1 is the last sample of the previous block
    let pos = this.pos;
    while (pos <= len - 1) {
      const i = Math.floor(pos);
      const f = pos - i;
      const a = i < 0 ? this.prev : filtered[i];
      const b = i + 1 < len ? filtered[i + 1] : a;
      this.emit(a + (b - a) * f);
      pos += this.step;
    }
    this.pos = pos - len;
    this.prev = filtered[len - 1];
    return true;
  }
}

registerProcessor("pcm16-writer", PCM16Writer);

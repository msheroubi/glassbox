import { useCallback, useEffect, useRef, useState } from "react";
import { wsUrl } from "./api";
import type { AskResponse } from "./types";

export type VoicePhase = "idle" | "connecting" | "listening" | "processing";

export type VoiceReply =
  | { type: "transcript"; text: string; final: boolean }
  | { type: "command"; text: string; command: string; message: string }
  | ({ type: "answer" } & AskResponse)
  | { type: "error"; message: string };

/**
 * Push-to-talk over /ws/voice: press -> {"type":"start"}; once the server says "listening", microphone audio
 * is streamed as 16 kHz PCM16 (public/pcm-worklet.js); release -> {"type":"stop"}. The server relays the audio
 * to Muse Voice Transcribe and routes the final transcript to a command or a Muse Spark answer.
 */
export function useVoice(onReply: (r: VoiceReply) => void) {
  const [phase, setPhase] = useState<VoicePhase>("idle");
  const [partial, setPartial] = useState("");
  const ws = useRef<WebSocket | null>(null);
  const audio = useRef<{ ctx: AudioContext; stream: MediaStream; node: AudioWorkletNode } | null>(null);
  const released = useRef(false);
  const replyRef = useRef(onReply);
  replyRef.current = onReply;

  const stopMic = useCallback(async () => {
    const a = audio.current;
    audio.current = null;
    if (!a) return;
    a.node.port.postMessage("flush");
    await new Promise((r) => setTimeout(r, 30)); // let the final chunk arrive
    a.stream.getTracks().forEach((t) => t.stop());
    a.node.disconnect();
    await a.ctx.close().catch(() => undefined);
  }, []);

  const teardown = useCallback(() => {
    void stopMic();
    ws.current?.close();
    ws.current = null;
    setPhase("idle");
  }, [stopMic]);

  const startMic = useCallback(async (sock: WebSocket) => {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule("/pcm-worklet.js");
    const src = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, "pcm16-writer");
    const mute = ctx.createGain();
    mute.gain.value = 0;
    src.connect(node).connect(mute).connect(ctx.destination); // keep the node pulled by the graph, silently
    node.port.onmessage = (e: MessageEvent<ArrayBuffer>) => {
      if (sock.readyState === WebSocket.OPEN) sock.send(e.data);
    };
    audio.current = { ctx, stream, node };
  }, []);

  const open = useCallback(
    (onOpen: (sock: WebSocket) => void) => {
      const sock = new WebSocket(wsUrl("/ws/voice"));
      sock.binaryType = "arraybuffer";
      ws.current = sock;
      sock.onopen = () => onOpen(sock);
      sock.onerror = () => {
        replyRef.current({ type: "error", message: "Voice websocket error" });
        teardown();
      };
      sock.onclose = () => {
        if (ws.current === sock) {
          ws.current = null;
          void stopMic();
          setPhase("idle");
        }
      };
      sock.onmessage = async (ev) => {
        if (typeof ev.data !== "string") return;
        const msg = JSON.parse(ev.data);
        switch (msg.type) {
          case "listening":
            if (released.current) {
              // button already released before the ASR session opened: nothing to transcribe
              sock.send(JSON.stringify({ type: "stop" }));
              setPhase("processing");
              return;
            }
            try {
              await startMic(sock);
              setPhase("listening");
            } catch (e) {
              replyRef.current({ type: "error", message: `Microphone unavailable: ${(e as Error).message}` });
              sock.send(JSON.stringify({ type: "stop" }));
              setPhase("processing");
            }
            break;
          case "transcript":
            setPartial(msg.text);
            replyRef.current(msg);
            break;
          case "command":
          case "answer":
            replyRef.current(msg);
            break;
          case "error":
            replyRef.current(msg);
            if (!audio.current) teardown();
            break;
          case "done":
            teardown();
            break;
          default:
            break;
        }
      };
    },
    [startMic, stopMic, teardown],
  );

  const press = useCallback(() => {
    if (ws.current) return;
    released.current = false;
    setPartial("");
    setPhase("connecting");
    open((sock) => sock.send(JSON.stringify({ type: "start" })));
  }, [open]);

  const release = useCallback(() => {
    released.current = true;
    const sock = ws.current;
    if (!sock || !audio.current) return; // still connecting: handled when "listening" arrives
    setPhase("processing");
    void stopMic().then(() => {
      if (sock.readyState === WebSocket.OPEN) sock.send(JSON.stringify({ type: "stop" }));
    });
  }, [stopMic]);

  /** Typed command through the same intent router as speech (e.g. "apply the fix"). */
  const sendText = useCallback(
    (text: string) => {
      if (ws.current) return;
      setPhase("processing");
      open((sock) => sock.send(JSON.stringify({ type: "text", text })));
    },
    [open],
  );

  useEffect(() => () => teardown(), [teardown]);

  return { phase, partial, press, release, sendText, cancel: teardown };
}

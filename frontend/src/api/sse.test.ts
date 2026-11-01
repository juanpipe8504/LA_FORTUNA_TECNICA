import { describe, expect, it } from "vitest";
import { SSEParser, type SSEEvent } from "./sse";

/** Feeds `text` split at every given position and returns all events (including flush). */
function parseChunks(chunks: string[]): SSEEvent[] {
  const p = new SSEParser();
  const out: SSEEvent[] = [];
  for (const c of chunks) out.push(...p.feed(c));
  out.push(...p.flush());
  return out;
}

/** Decodes byte chunks the way streamChat does (TextDecoder with stream: true). */
function parseBytes(bytes: Uint8Array, cuts: number[]): SSEEvent[] {
  const decoder = new TextDecoder("utf-8");
  const p = new SSEParser();
  const out: SSEEvent[] = [];
  let prev = 0;
  for (const cut of [...cuts, bytes.length]) {
    out.push(...p.feed(decoder.decode(bytes.subarray(prev, cut), { stream: true })));
    prev = cut;
  }
  out.push(...p.feed(decoder.decode()));
  out.push(...p.flush());
  return out;
}

const STREAM =
  ': ping\n\n' +
  'event: token\ndata: {"text": "Hola "}\n\n' +
  'event: token\ndata: {"text": "señor — ✓ 😀"}\n\n' +
  ': keep-alive\n' +
  'event: final\ndata: {"answer": "Hola [1]", "has_sufficient_context": true, "sources": [], "images": []}\n\n';

const EXPECTED: SSEEvent[] = [
  { event: "token", data: '{"text": "Hola "}' },
  { event: "token", data: '{"text": "señor — ✓ 😀"}' },
  { event: "final", data: '{"answer": "Hola [1]", "has_sufficient_context": true, "sources": [], "images": []}' },
];

describe("SSEParser", () => {
  it("parses a complete stream in one chunk, ignoring comments", () => {
    expect(parseChunks([STREAM])).toEqual(EXPECTED);
  });

  it("handles every possible single split point", () => {
    for (let i = 0; i <= STREAM.length; i++) {
      expect(parseChunks([STREAM.slice(0, i), STREAM.slice(i)])).toEqual(EXPECTED);
    }
  });

  it("handles one character per chunk", () => {
    expect(parseChunks([...STREAM])).toEqual(EXPECTED);
  });

  it("supports \\r\\n and lone \\r line endings, including \\r\\n split across chunks", () => {
    const crlf = STREAM.replace(/\n/g, "\r\n");
    expect(parseChunks([crlf])).toEqual(EXPECTED);
    for (let i = 0; i <= crlf.length; i++) {
      expect(parseChunks([crlf.slice(0, i), crlf.slice(i)])).toEqual(EXPECTED);
    }
    expect(parseChunks([STREAM.replace(/\n/g, "\r")])).toEqual(EXPECTED);
  });

  it("joins multiple data lines with newlines and strips only one leading space", () => {
    const events = parseChunks(["event: final\ndata: line1\ndata:  line2\ndata:line3\ndata\n\n"]);
    expect(events).toEqual([{ event: "final", data: "line1\n line2\nline3\n" }]);
  });

  it("defaults the event name to 'message' and resets it between events", () => {
    expect(parseChunks(["event: token\ndata: a\n\ndata: b\n\n"])).toEqual([
      { event: "token", data: "a" },
      { event: "message", data: "b" },
    ]);
  });

  it("does not dispatch events without data and ignores id/retry/unknown fields", () => {
    expect(parseChunks(["event: token\n\nid: 7\nretry: 1000\nfoo: bar\ndata: x\n\n"])).toEqual([
      { event: "message", data: "x" },
    ]);
  });

  it("keeps colons inside the value", () => {
    expect(parseChunks(['data: {"a": "b:c"}\n\n'])).toEqual([{ event: "message", data: '{"a": "b:c"}' }]);
  });

  it("returns nothing until an event is terminated, and flushes a trailing unterminated event", () => {
    const p = new SSEParser();
    expect(p.feed("event: final\ndata: {")).toEqual([]);
    expect(p.feed('"x": 1}')).toEqual([]);
    expect(p.flush()).toEqual([{ event: "final", data: '{"x": 1}' }]);
  });

  it("reassembles multibyte UTF-8 characters split at every byte boundary", () => {
    const bytes = new TextEncoder().encode(STREAM);
    for (let i = 0; i <= bytes.length; i++) {
      expect(parseBytes(bytes, [i])).toEqual(EXPECTED);
    }
    // Every byte in its own chunk.
    expect(parseBytes(bytes, Array.from({ length: bytes.length }, (_, i) => i))).toEqual(EXPECTED);
  });
});

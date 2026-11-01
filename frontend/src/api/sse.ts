/**
 * Minimal, robust Server-Sent Events parser (WHATWG "event stream" rules).
 *
 * - Lines may end in `\n`, `\r\n` or `\r` (including a `\r\n` pair split across chunks).
 * - Events may be split across arbitrary chunk boundaries.
 * - Multiple `data:` lines are joined with `\n`.
 * - Lines starting with `:` are comments (keep-alive pings) and are ignored.
 * - A single space after the colon is stripped; `id` / `retry` are ignored.
 *
 * Feed it *decoded text* (use `TextDecoder#decode(chunk, { stream: true })` so
 * multibyte UTF-8 characters split across network chunks are reassembled).
 */

export interface SSEEvent {
  /** Event name; defaults to "message" when no `event:` field was sent. */
  event: string;
  data: string;
}

export class SSEParser {
  private buffer = "";
  /** Last chunk ended in `\r`: a leading `\n` in the next chunk belongs to the same line break. */
  private skipLeadingLF = false;
  private eventName = "";
  private dataLines: string[] = [];

  /** Parses a chunk of text and returns the events completed by it. */
  feed(text: string): SSEEvent[] {
    if (this.skipLeadingLF && text.startsWith("\n")) text = text.slice(1);
    this.skipLeadingLF = false;
    this.buffer += text;

    const events: SSEEvent[] = [];
    let start = 0;
    for (let i = 0; i < this.buffer.length; i++) {
      const ch = this.buffer.charCodeAt(i);
      if (ch !== 10 /* \n */ && ch !== 13 /* \r */) continue;
      const line = this.buffer.slice(start, i);
      if (ch === 13) {
        if (i + 1 < this.buffer.length) {
          if (this.buffer.charCodeAt(i + 1) === 10) i++;
        } else {
          this.skipLeadingLF = true;
        }
      }
      start = i + 1;
      const ev = this.processLine(line);
      if (ev) events.push(ev);
    }
    this.buffer = this.buffer.slice(start);
    return events;
  }

  /**
   * Call at end of stream. Leniently dispatches a trailing event that was not
   * terminated by a blank line (the spec would drop it).
   */
  flush(): SSEEvent[] {
    const events: SSEEvent[] = [];
    if (this.buffer) {
      const ev = this.processLine(this.buffer);
      if (ev) events.push(ev);
      this.buffer = "";
    }
    const ev = this.dispatch();
    if (ev) events.push(ev);
    return events;
  }

  private processLine(line: string): SSEEvent | null {
    if (line === "") return this.dispatch();
    if (line.startsWith(":")) return null; // comment / keep-alive
    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1);
    if (field === "event") this.eventName = value;
    else if (field === "data") this.dataLines.push(value);
    // `id`, `retry` and unknown fields are ignored.
    return null;
  }

  private dispatch(): SSEEvent | null {
    const hasData = this.dataLines.length > 0;
    const ev: SSEEvent | null = hasData ? { event: this.eventName || "message", data: this.dataLines.join("\n") } : null;
    this.eventName = "";
    this.dataLines = [];
    return ev;
  }
}

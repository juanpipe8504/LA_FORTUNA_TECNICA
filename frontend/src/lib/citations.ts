/**
 * Citation helpers.
 *
 * The LLM cites context blocks as `[1]`, `[2]`, `[1, 3]` or `[2-4]`. The API
 * returns `sources` in citation order but — per the contract — only the chunks
 * that were actually cited when citations exist. That means the numbers in the
 * text may be *context block* numbers (e.g. `[1]` and `[4]`) while `sources`
 * only has two entries. `buildCitationMap` resolves both situations:
 *
 *  1. Every cited number fits inside `sources` → direct mapping `[n] → sources[n-1]`.
 *  2. Otherwise, if the number of distinct cited numbers equals `sources.length`,
 *     the backend kept only the cited chunks in order → map the sorted distinct
 *     numbers onto sources positionally (`[1]→0`, `[4]→1`).
 *  3. Fallback: direct mapping where possible; the rest stay unresolved and are
 *     rendered as non-clickable badges.
 */

/** Matches `[1]`, `[1, 2]`, `[1;2]`, `[2-4]` (not markdown links, not `[[MARKER]]`). */
export const CITATION_PATTERN = /\[(\d{1,3}(?:\s*[,;\-–]\s*\d{1,3})*)\](?!\()/g;

const MAX_RANGE = 50;

/** Expands the inner text of a citation group ("1, 3-4") into numbers. */
export function expandCitationGroup(group: string): number[] {
  const out: number[] = [];
  const parts = group.split(/\s*[,;]\s*/);
  for (const part of parts) {
    const range = part.split(/\s*[-–]\s*/).map((p) => Number.parseInt(p, 10));
    if (range.length === 2 && range.every(Number.isFinite) && range[1] >= range[0] && range[1] - range[0] <= MAX_RANGE) {
      for (let n = range[0]; n <= range[1]; n++) out.push(n);
    } else {
      for (const n of range) if (Number.isFinite(n)) out.push(n);
    }
  }
  return out.filter((n) => n > 0);
}

/** Removes fenced and inline code so citations inside code are ignored. */
function stripCode(markdown: string): string {
  return markdown.replace(/```[\s\S]*?```/g, " ").replace(/`[^`\n]*`/g, " ");
}

/** Distinct citation numbers in order of first appearance. */
export function parseCitationNumbers(markdown: string): number[] {
  const seen = new Set<number>();
  const order: number[] = [];
  for (const match of stripCode(markdown).matchAll(CITATION_PATTERN)) {
    for (const n of expandCitationGroup(match[1])) {
      if (!seen.has(n)) {
        seen.add(n);
        order.push(n);
      }
    }
  }
  return order;
}

export interface CitationMap {
  /** citation number → index in `sources` */
  toSource: Map<number, number>;
  /** index in `sources` → citation number shown on the card */
  toCitation: Map<number, number>;
}

export function buildCitationMap(markdown: string, sourcesLength: number): CitationMap {
  const cited = parseCitationNumbers(markdown);
  const toSource = new Map<number, number>();

  if (cited.length > 0 && cited.every((n) => n <= sourcesLength)) {
    for (const n of cited) toSource.set(n, n - 1);
  } else if (cited.length > 0 && cited.length === sourcesLength) {
    [...cited].sort((a, b) => a - b).forEach((n, i) => toSource.set(n, i));
  } else {
    for (const n of cited) if (n <= sourcesLength) toSource.set(n, n - 1);
  }

  const toCitation = new Map<number, number>();
  for (const [n, idx] of toSource) if (!toCitation.has(idx)) toCitation.set(idx, n);
  // Uncited sources keep a positional number that doesn't collide with cited ones.
  const used = new Set(toSource.keys());
  let next = 1;
  for (let i = 0; i < sourcesLength; i++) {
    if (toCitation.has(i)) continue;
    while (used.has(next)) next++;
    toCitation.set(i, next);
    used.add(next);
  }
  return { toSource, toCitation };
}

/** Defensive clean-up of the answer text before rendering. */
export function cleanAnswer(markdown: string): string {
  return markdown.replace(/\[\[NO_CONTEXT\]\]/g, "").trim();
}

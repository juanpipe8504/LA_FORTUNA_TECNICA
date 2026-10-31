import { CITATION_PATTERN, expandCitationGroup } from "./citations";

/**
 * Minimal remark plugin: turns `[n]` citation text into link nodes with
 * `#cite-n` hrefs so the Markdown renderer can swap them for badges.
 * Only plain `text` nodes are touched, so code / inline code stay untouched.
 */

interface MdNode {
  type: string;
  value?: string;
  url?: string;
  children?: MdNode[];
  data?: Record<string, unknown>;
}

export const CITE_HREF_PREFIX = "#cite-";

function splitText(value: string): MdNode[] | null {
  const pattern = new RegExp(CITATION_PATTERN.source, "g");
  const nodes: MdNode[] = [];
  let last = 0;
  let found = false;
  for (const match of value.matchAll(pattern)) {
    found = true;
    const start = match.index ?? 0;
    if (start > last) nodes.push({ type: "text", value: value.slice(last, start) });
    for (const n of expandCitationGroup(match[1])) {
      nodes.push({ type: "link", url: `${CITE_HREF_PREFIX}${n}`, children: [{ type: "text", value: String(n) }] });
    }
    last = start + match[0].length;
  }
  if (!found) return null;
  if (last < value.length) nodes.push({ type: "text", value: value.slice(last) });
  return nodes;
}

function walk(node: MdNode): void {
  if (!node.children || node.type === "link" || node.type === "linkReference") return;
  const next: MdNode[] = [];
  for (const child of node.children) {
    if (child.type === "text" && typeof child.value === "string") {
      const replaced = splitText(child.value);
      if (replaced) next.push(...replaced);
      else next.push(child);
    } else {
      walk(child);
      next.push(child);
    }
  }
  node.children = next;
}

export default function remarkCitations() {
  return (tree: MdNode) => walk(tree);
}

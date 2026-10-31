import { memo, type ComponentProps } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import type { SourceCitation } from "../api/types";
import remarkCitations, { CITE_HREF_PREFIX } from "../lib/remarkCitations";

interface Props {
  content: string;
  sources: SourceCitation[];
  citationToSource: Map<number, number>;
  onCite: (sourceIndex: number) => void;
}

function CitationBadge({
  n,
  source,
  onClick,
}: {
  n: number;
  source: SourceCitation | undefined;
  onClick: () => void;
}) {
  if (!source) {
    return (
      <sup className="cite cite--unresolved" title="Fuente no incluida en la respuesta">
        [{n}]
      </sup>
    );
  }
  const label = `Fuente ${n}: ${source.document_name}, página ${source.page}`;
  return (
    <sup>
      <button type="button" className="cite" onClick={onClick} title={label} aria-label={`Ver ${label}`}>
        {n}
      </button>
    </sup>
  );
}

function MarkdownAnswerImpl({ content, sources, citationToSource, onCite }: Props) {
  const components: Components = {
    a({ href, children, node: _node, ...rest }: ComponentProps<"a"> & { node?: unknown }) {
      if (href?.startsWith(CITE_HREF_PREFIX)) {
        const n = Number.parseInt(href.slice(CITE_HREF_PREFIX.length), 10);
        const idx = citationToSource.get(n);
        return (
          <CitationBadge
            n={n}
            source={idx === undefined ? undefined : sources[idx]}
            onClick={() => idx !== undefined && onCite(idx)}
          />
        );
      }
      return (
        <a href={href} target="_blank" rel="noopener noreferrer" {...rest}>
          {children}
        </a>
      );
    },
    table({ node: _node, ...rest }) {
      return (
        <div className="md-table-wrap" tabIndex={0} role="region" aria-label="Tabla">
          <table {...rest} />
        </div>
      );
    },
    img({ node: _node, alt }) {
      // Images inside the answer text are not trusted/served; visual context is shown separately.
      return <span className="md-img-alt">[{alt || "imagen"}]</span>;
    },
  };

  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkCitations]} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}

export const MarkdownAnswer = memo(MarkdownAnswerImpl);

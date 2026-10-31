interface Props {
  documentName: string;
  page: number;
  className?: string;
}

/** "📄 manual.pdf · Página 4" — the anchor that ties every piece of evidence to its origin. */
export function PageBadge({ documentName, page, className = "" }: Props) {
  return (
    <span className={`page-badge ${className}`} title={`${documentName}, página ${page}`}>
      <span aria-hidden="true">📄</span>
      <span className="page-badge__doc">{documentName}</span>
      <span aria-hidden="true">·</span>
      <span className="page-badge__page">Página {page}</span>
    </span>
  );
}

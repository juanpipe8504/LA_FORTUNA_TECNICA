import { act, render, screen, waitFor } from "@testing-library/react";
import { useEffect } from "react";
import { describe, expect, it, vi } from "vitest";
import { UploadList } from "../components/UploadList";
import { useUploads, validatePdf } from "./useUploads";

function job(status: string, progress: number, extra: Record<string, unknown> = {}) {
  return {
    job_id: "job-1",
    document_id: "doc-1",
    filename: "informe.pdf",
    status,
    progress,
    message: status === "processing" ? "Extrayendo páginas" : "",
    error: null,
    page_count: status === "completed" ? 12 : null,
    chunk_count: status === "completed" ? 40 : null,
    image_count: status === "completed" ? 3 : null,
    created_at: "2026-01-01T00:00:00Z",
    updated_at: "2026-01-01T00:00:00Z",
    ...extra,
  };
}

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });

const accepted = () =>
  json({ job_id: "job-1", document_id: "doc-1", filename: "informe.pdf", status: "pending" }, 202);

function Harness({ file, onCompleted }: { file: File; onCompleted: (id: string) => void }) {
  const { items, upload, dismiss, clearFinished } = useUploads({ pollIntervalMs: 5, onCompleted });
  useEffect(() => upload([file]), [file, upload]);
  return <UploadList items={items} onDismiss={dismiss} onClearFinished={clearFinished} />;
}

const pdf = () => new File(["%PDF-1.7 fake"], "informe.pdf", { type: "application/pdf" });

describe("upload + job polling", () => {
  it("uploads, polls GET /jobs/{id} and shows progress until completed", async () => {
    const responses = [accepted(), json(job("processing", 40)), json(job("processing", 80)), json(job("completed", 100))];
    const fetchMock = vi.fn().mockImplementation(() => Promise.resolve(responses.shift()));
    vi.stubGlobal("fetch", fetchMock);
    const onCompleted = vi.fn();

    render(<Harness file={pdf()} onCompleted={onCompleted} />);

    await waitFor(() => expect(screen.getByText("Completado")).toBeInTheDocument());
    expect(onCompleted).toHaveBeenCalledWith("doc-1");
    expect(screen.getByRole("progressbar")).toHaveAttribute("aria-valuenow", "100");
    expect(screen.getByText("12 páginas · 40 fragmentos · 3 imágenes")).toBeInTheDocument();

    const urls = fetchMock.mock.calls.map((c) => c[0]);
    expect(urls[0]).toBe("/api/v1/documents");
    expect(fetchMock.mock.calls[0][1].method).toBe("POST");
    expect(urls.slice(1)).toEqual(["/api/v1/jobs/job-1", "/api/v1/jobs/job-1", "/api/v1/jobs/job-1"]);
  });

  it("keeps polling through a transient network error", async () => {
    const responses: Array<() => Promise<Response>> = [
      () => Promise.resolve(accepted()),
      () => Promise.reject(new TypeError("Failed to fetch")),
      () => Promise.resolve(json(job("completed", 100))),
    ];
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => responses.shift()!()));
    render(<Harness file={pdf()} onCompleted={vi.fn()} />);
    await waitFor(() => expect(screen.getByText("Completado")).toBeInTheDocument());
  });

  it("shows the failed chip with the backend error message", async () => {
    const responses = [accepted(), json(job("failed", 30, { error: "PDF cifrado: no se puede leer" }))];
    vi.stubGlobal("fetch", vi.fn().mockImplementation(() => Promise.resolve(responses.shift())));
    const onCompleted = vi.fn();
    render(<Harness file={pdf()} onCompleted={onCompleted} />);
    await waitFor(() => expect(screen.getByText("Fallido")).toBeInTheDocument());
    expect(screen.getByRole("alert")).toHaveTextContent("PDF cifrado: no se puede leer");
    expect(onCompleted).not.toHaveBeenCalled();
  });

  it("rejects non-PDF files client-side without calling the API", async () => {
    const fetchMock = vi.fn();
    vi.stubGlobal("fetch", fetchMock);
    const txt = new File(["hola"], "notas.txt", { type: "text/plain" });
    expect(validatePdf(txt)).toMatch(/Solo se admiten/);
    await act(async () => {
      render(<Harness file={txt} onCompleted={vi.fn()} />);
    });
    expect(screen.getByText("Fallido")).toBeInTheDocument();
    expect(fetchMock).not.toHaveBeenCalled();
  });
});

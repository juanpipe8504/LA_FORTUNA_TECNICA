from __future__ import annotations

from app.domain.exceptions import NotFoundError, ValidationError
from app.domain.models import JobStatus
from tests.unit.api.conftest import API, make_job

PDF = b"%PDF-1.7\n%fake pdf body\n"


def _upload(client, data=PDF, filename="manual.pdf", content_type="application/pdf"):
    return client.post(f"{API}/documents", files={"file": (filename, data, content_type)})


class TestUpload:
    def test_returns_202_with_location_header(self, client, container):
        container.submit_document.execute.return_value = make_job()

        response = _upload(client)

        assert response.status_code == 202
        assert response.headers["location"] == f"{API}/jobs/job-1"
        assert response.json() == {
            "job_id": "job-1",
            "document_id": "doc-1",
            "filename": "manual.pdf",
            "status": "pending",
        }
        container.submit_document.execute.assert_called_once_with("manual.pdf", PDF)

    def test_accepts_octet_stream_with_pdf_extension(self, client, container):
        container.submit_document.execute.return_value = make_job()

        response = _upload(client, content_type="application/octet-stream")

        assert response.status_code == 202

    def test_rejects_non_pdf_with_415(self, client, container):
        response = _upload(client, data=b"hello", filename="notes.txt", content_type="text/plain")

        assert response.status_code == 415
        assert response.json()["error"]["code"] == "unsupported_media_type"
        container.submit_document.execute.assert_not_called()

    def test_rejects_file_over_limit_with_413(self, client, container):
        too_big = PDF + b"0" * (1024 * 1024)  # settings.max_upload_mb == 1

        response = _upload(client, data=too_big)

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"
        container.submit_document.execute.assert_not_called()

    def test_use_case_validation_error_maps_to_422(self, client, container):
        container.submit_document.execute.side_effect = ValidationError("Not a valid PDF file")

        response = _upload(client, data=b"not really a pdf")

        assert response.status_code == 422
        body = response.json()["error"]
        assert body["code"] == "validation_error"
        assert body["message"] == "Not a valid PDF file"
        assert body["request_id"]

    def test_missing_file_field_is_422(self, client):
        response = client.post(f"{API}/documents")

        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


class TestListDocuments:
    def test_maps_jobs_to_documents(self, client, container):
        container.list_documents.execute.return_value = [
            make_job(status=JobStatus.COMPLETED, page_count=3, chunk_count=12, image_count=2)
        ]

        response = client.get(f"{API}/documents")

        assert response.status_code == 200
        assert response.json() == [
            {
                "document_id": "doc-1",
                "filename": "manual.pdf",
                "page_count": 3,
                "chunk_count": 12,
                "image_count": 2,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ]

    def test_empty_list(self, client, container):
        container.list_documents.execute.return_value = []

        assert client.get(f"{API}/documents").json() == []


class TestGetImage:
    URL = f"{API}/documents/doc-1/images/doc-1_p1_i0"

    def test_returns_bytes_with_immutable_cache_headers(self, client, container):
        container.blob_storage.get_image.return_value = (b"\x89PNG-bytes", "image/png")

        response = client.get(self.URL)

        assert response.status_code == 200
        assert response.content == b"\x89PNG-bytes"
        assert response.headers["content-type"] == "image/png"
        assert response.headers["cache-control"] == "public, max-age=31536000, immutable"
        container.blob_storage.get_image.assert_called_once_with("doc-1", "doc-1_p1_i0")

    def test_missing_image_is_404(self, client, container):
        container.blob_storage.get_image.side_effect = NotFoundError("Image not found")

        response = client.get(self.URL)

        assert response.status_code == 404
        assert response.json()["error"]["code"] == "not_found"

    def test_rejects_unsafe_ids(self, client, container):
        response = client.get(f"{API}/documents/doc..1/images/img")

        assert response.status_code == 422
        container.blob_storage.get_image.assert_not_called()

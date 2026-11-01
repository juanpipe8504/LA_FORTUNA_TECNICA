"""FastAPI dependency providers.

Routes depend on use cases (or, for pure lookups, on ports) obtained from the
container stored in ``app.state.container`` - never on concrete adapters.
Tests swap the whole graph with ``create_app(container=fake)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from fastapi import Depends, Request

from app.bootstrap.container import Container, HealthCheck
from app.core.config import Settings
from app.domain.ports import BlobStoragePort, JobRepositoryPort

if TYPE_CHECKING:  # pragma: no cover - typing only; keeps the API import graph light
    from app.application.documents import ListDocumentsUseCase
    from app.application.ingestion import SubmitDocumentUseCase
    from app.application.jobs import GetJobUseCase
    from app.application.query import AnswerQuestionUseCase


def get_container(request: Request) -> Container:
    container = getattr(request.app.state, "container", None)
    if container is None:  # pragma: no cover - lifespan always sets it
        raise RuntimeError("Application container is not initialised")
    return container


ContainerDep = Annotated[Container, Depends(get_container)]


def get_settings(container: ContainerDep) -> Settings:
    return container.settings


def get_submit_document(container: ContainerDep) -> SubmitDocumentUseCase:
    return container.submit_document


def get_list_documents(container: ContainerDep) -> ListDocumentsUseCase:
    return container.list_documents


def get_get_job(container: ContainerDep) -> GetJobUseCase:
    return container.get_job


def get_answer_question(container: ContainerDep) -> AnswerQuestionUseCase:
    return container.answer_question


def get_job_repository(container: ContainerDep) -> JobRepositoryPort:
    return container.job_repository


def get_blob_storage(container: ContainerDep) -> BlobStoragePort:
    return container.blob_storage


def get_health_checks(container: ContainerDep) -> dict[str, HealthCheck]:
    return container.health_checks


SettingsDep = Annotated[Settings, Depends(get_settings)]
SubmitDocumentDep = Annotated["SubmitDocumentUseCase", Depends(get_submit_document)]
ListDocumentsDep = Annotated["ListDocumentsUseCase", Depends(get_list_documents)]
GetJobDep = Annotated["GetJobUseCase", Depends(get_get_job)]
AnswerQuestionDep = Annotated["AnswerQuestionUseCase", Depends(get_answer_question)]
JobRepositoryDep = Annotated[JobRepositoryPort, Depends(get_job_repository)]
BlobStorageDep = Annotated[BlobStoragePort, Depends(get_blob_storage)]
HealthChecksDep = Annotated[dict[str, HealthCheck], Depends(get_health_checks)]

"""Domain-level exceptions. The API layer maps them to HTTP status codes."""


class DomainError(Exception):
    """Base class for all expected application errors."""


class NotFoundError(DomainError):
    pass


class ValidationError(DomainError):
    pass


class ExternalServiceError(DomainError):
    """An external dependency (LLM, embeddings, vector DB) failed."""


class TransientError(ExternalServiceError):
    """Failure worth retrying: timeouts, connection resets, 429, 5xx."""


class CircuitOpenError(ExternalServiceError):
    """The circuit breaker is open; calls are short-circuited."""


class DocumentParsingError(DomainError):
    pass

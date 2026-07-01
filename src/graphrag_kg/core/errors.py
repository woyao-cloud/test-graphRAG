"""Custom exception hierarchy for GraphRAG-KG."""


class GraphRAGKGError(Exception):
    """Base exception for all GraphRAG-KG errors."""


class ConfigError(GraphRAGKGError):
    """Configuration-related errors (missing keys, invalid values, etc.)."""


class ConfigValidationError(ConfigError):
    """Configuration validation failed."""


class ProjectError(GraphRAGKGError):
    """Project initialization or structure errors."""


class IngestionError(GraphRAGKGError):
    """Document ingestion errors."""


class UnsupportedFormatError(IngestionError):
    """Document format not supported."""


class ParserError(IngestionError):
    """Document parsing failed."""


class IndexingError(GraphRAGKGError):
    """Knowledge graph indexing errors."""


class Neo4jConnectionError(GraphRAGKGError):
    """Neo4j connection or authentication errors."""


class Neo4jSyncError(GraphRAGKGError):
    """Neo4j data synchronization errors."""


class QueryError(GraphRAGKGError):
    """Query execution errors."""


class StorageError(GraphRAGKGError):
    """Storage backend errors."""


class DataGenerationError(GraphRAGKGError):
    """Test data generation errors."""

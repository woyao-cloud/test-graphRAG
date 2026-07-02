"""Configuration loader for GraphRAG-KG.

Loads and validates YAML configuration files with environment variable
substitution, profile merging, and CLI override support.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import ValidationError

from graphrag_kg.core.config import KGConfig, PROFILES
from graphrag_kg.core.errors import ConfigError, ConfigValidationError

# Pattern for ${VAR_NAME} and ${VAR_NAME:default_value} substitution
_ENV_VAR_PATTERN = re.compile(r"\$\{(\w+)(?::([^}]*))?\}")


class ConfigLoader:
    """Loads and validates KGConfig from YAML files.

    Supports:
    - YAML loading with environment variable substitution
    - Profile-based configuration (default, fast, production)
    - CLI parameter overrides
    - Nested key access via dot notation
    """

    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = Path(config_path) if config_path else None
        self._raw_data: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load(self, config_path: Optional[Path] = None) -> KGConfig:
        """Load and validate configuration from a YAML file.

        If no path is provided, returns defaults.
        """
        path = Path(config_path) if config_path else self.config_path

        if path and path.exists():
            self.config_path = path
            self._raw_data = self._read_yaml(path)
        else:
            self._raw_data = {}

        return self._build_config()

    def load_profile(self, profile_name: str) -> KGConfig:
        """Load a named profile (default, fast, production).

        Merges the profile with any YAML file if one is set.
        """
        if profile_name not in PROFILES:
            available = list(PROFILES.keys())
            raise ConfigError(
                f"Unknown profile '{profile_name}'. Available: {available}"
            )

        # Start with profile defaults
        profile_config = PROFILES[profile_name]()

        # Merge with YAML file if present
        if self._raw_data:
            profile_config = self._merge_config(profile_config, self._raw_data)

        return profile_config

    def load_with_overrides(
        self,
        config_path: Optional[Path] = None,
        profile: Optional[str] = None,
        overrides: Optional[dict[str, Any]] = None,
    ) -> KGConfig:
        """Load config with profile and CLI override support.

        Precedence (highest to lowest):
        1. CLI overrides
        2. YAML file values
        3. Profile defaults
        4. System defaults
        """
        # Base: profile or defaults
        if profile:
            config = self.load_profile(profile)
        else:
            config = self.load(config_path)

        # Apply YAML overrides
        if config_path and config_path.exists():
            raw = self._read_yaml(config_path)
            config = self._merge_config(config, raw)

        # Apply CLI overrides
        if overrides:
            config = self._apply_overrides(config, overrides)

        return config

    # ------------------------------------------------------------------
    # YAML reading
    # ------------------------------------------------------------------

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        """Read a YAML file with environment variable substitution."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except FileNotFoundError:
            raise ConfigError(f"Configuration file not found: {path}")
        except OSError as e:
            raise ConfigError(f"Cannot read configuration file {path}: {e}")

        # Substitute environment variables
        resolved_text = self._resolve_env_vars(raw_text)

        try:
            data = yaml.safe_load(resolved_text)
            if not isinstance(data, dict):
                raise ConfigError(f"Configuration must be a YAML mapping: {path}")
            return data
        except yaml.YAMLError as e:
            raise ConfigError(f"Invalid YAML in {path}: {e}")

    def _resolve_env_vars(self, text: str) -> str:
        """Replace ${VAR} and ${VAR:default} with environment values."""

        def _replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)
            env_value = os.environ.get(var_name)
            if env_value is not None:
                return env_value
            if default is not None:
                return default
            # Return the original placeholder if no default
            return match.group(0)

        return _ENV_VAR_PATTERN.sub(_replacer, text)

    # ------------------------------------------------------------------
    # Config building
    # ------------------------------------------------------------------

    def _build_config(self) -> KGConfig:
        """Build a KGConfig from raw YAML data and environment variables."""
        import os as _os

        # Read from env vars (used when no YAML config is present)
        env_chat = _os.environ.get("GRAPHRAG_CHAT_MODEL", "")
        env_embed = _os.environ.get("GRAPHRAG_EMBEDDING_MODEL", "")
        env_key = _os.environ.get("GRAPHRAG_API_KEY", "")
        env_base = _os.environ.get("GRAPHRAG_API_BASE", "")

        if not self._raw_data:
            return KGConfig(
                chat_model=env_chat or "gpt-4.1",
                embedding_model=env_embed or "text-embedding-3-large",
                api_key=env_key,
                api_base=env_base,
            )

        try:
            # Extract our kg section if present
            kg_data = self._raw_data.get("kg", {})

            # Build config with Pydantic validation
            # YAML values take priority, fall back to env vars, then defaults
            # Priority: env vars > YAML values > hardcoded defaults.
            # This allows .env to override settings.yaml for model configuration.
            config = KGConfig(
                project_name=kg_data.get("project_name", "my-knowledge-graph"),
                description=kg_data.get("description", ""),
                chat_model=env_chat or kg_data.get("chat_model") or "gpt-4.1",
                chat_model_provider=os.environ.get("GRAPHRAG_CHAT_MODEL_PROVIDER")
                    or kg_data.get("chat_model_provider", "openai"),
                embedding_model=env_embed or kg_data.get("embedding_model") or "text-embedding-3-large",
                embedding_model_provider=os.environ.get("GRAPHRAG_EMBEDDING_MODEL_PROVIDER")
                    or kg_data.get("embedding_model_provider", "openai"),
                api_key=env_key or kg_data.get("api_key") or "",
                api_base=env_base or kg_data.get("api_base") or "",
            )

            # Apply sub-configs
            if "neo4j" in kg_data:
                config.neo4j = self._build_neo4j(kg_data["neo4j"])
            if "ingestion" in kg_data:
                config.ingestion = self._build_ingestion(kg_data["ingestion"])
            if "pipeline" in kg_data:
                config.pipeline = self._build_pipeline(kg_data["pipeline"])
            if "query" in kg_data:
                config.query = self._build_query(kg_data["query"])

            # Store raw graphrag settings
            config.graphrag_settings = {
                k: v for k, v in self._raw_data.items()
                if k != "kg"
            }

            return config

        except ValidationError as e:
            raise ConfigValidationError(
                f"Configuration validation failed:\n{e}"
            ) from e

    def _build_neo4j(self, data: dict) -> Any:
        from graphrag_kg.core.config import Neo4jConfig
        return Neo4jConfig(**data)

    def _build_ingestion(self, data: dict) -> Any:
        from graphrag_kg.core.config import IngestionConfig
        return IngestionConfig(**data)

    def _build_pipeline(self, data: dict) -> Any:
        from graphrag_kg.core.config import PipelineConfig
        return PipelineConfig(**data)

    def _build_query(self, data: dict) -> Any:
        from graphrag_kg.core.config import QueryConfig
        return QueryConfig(**data)

    # ------------------------------------------------------------------
    # Merging and overrides
    # ------------------------------------------------------------------

    def _merge_config(self, base: KGConfig, overrides: dict[str, Any]) -> KGConfig:
        """Merge override dict into an existing KGConfig."""
        kg_overrides = overrides.get("kg", {})
        if not kg_overrides:
            return base

        merge_fields = {
            "project_name": kg_overrides.get("project_name"),
            "description": kg_overrides.get("description"),
            "chat_model": kg_overrides.get("chat_model"),
            "chat_model_provider": kg_overrides.get("chat_model_provider"),
            "embedding_model": kg_overrides.get("embedding_model"),
            "embedding_model_provider": kg_overrides.get("embedding_model_provider"),
            "api_key": kg_overrides.get("api_key"),
            "api_base": kg_overrides.get("api_base"),
        }

        for field, value in merge_fields.items():
            if value is not None:
                setattr(base, field, value)

        # Merge sub-configs
        if "neo4j" in kg_overrides:
            from graphrag_kg.core.config import Neo4jConfig
            neo4j_data = base.neo4j.model_dump()
            neo4j_data.update(kg_overrides["neo4j"])
            base.neo4j = Neo4jConfig(**neo4j_data)

        if "ingestion" in kg_overrides:
            from graphrag_kg.core.config import IngestionConfig
            ingest_data = base.ingestion.model_dump()
            ingest_data.update(kg_overrides["ingestion"])
            base.ingestion = IngestionConfig(**ingest_data)

        if "pipeline" in kg_overrides:
            from graphrag_kg.core.config import PipelineConfig
            pipe_data = base.pipeline.model_dump()
            pipe_data.update(kg_overrides["pipeline"])
            base.pipeline = PipelineConfig(**pipe_data)

        if "query" in kg_overrides:
            from graphrag_kg.core.config import QueryConfig
            query_data = base.query.model_dump()
            query_data.update(kg_overrides["query"])
            base.query = QueryConfig(**query_data)

        # Merge graphrag settings
        graphrag_overrides = {
            k: v for k, v in overrides.items() if k != "kg"
        }
        if graphrag_overrides:
            base.graphrag_settings.update(graphrag_overrides)

        return base

    def _apply_overrides(self, config: KGConfig, overrides: dict[str, Any]) -> KGConfig:
        """Apply CLI-style overrides to a config.

        Supports dot-notation keys like "neo4j.uri" and "query.default_method".
        """
        for key, value in overrides.items():
            parts = key.split(".")
            if len(parts) == 1:
                # Top-level field
                if hasattr(config, parts[0]):
                    setattr(config, parts[0], value)
            elif len(parts) == 2:
                # Nested field
                parent_name, child_name = parts
                if hasattr(config, parent_name):
                    parent = getattr(config, parent_name)
                    if hasattr(parent, child_name):
                        setattr(parent, child_name, value)

        return config

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, config: Optional[KGConfig] = None) -> list[str]:
        """Validate a config and return list of warnings/issues.

        Returns empty list if config is valid.
        """
        if config is None:
            config = self._build_config()

        warnings: list[str] = []

        # Check for missing API key
        if not config.api_key and not os.environ.get("GRAPHRAG_API_KEY"):
            warnings.append(
                "No API key configured. Set 'api_key' in config or "
                "GRAPHRAG_API_KEY environment variable."
            )

        # Check Neo4j connectivity hints
        if config.neo4j.password == "password":
            warnings.append(
                "Neo4j password is still the default 'password'. "
                "Consider changing it for production use."
            )

        # Check directory existence
        for dir_path in [config.input_dir, config.prompts_dir]:
            if not dir_path.exists():
                warnings.append(
                    f"Directory does not exist: {dir_path}. "
                    f"Run 'graphrag-kg init' to create it."
                )

        return warnings

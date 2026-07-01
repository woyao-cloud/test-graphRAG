"""Tests for the ConfigLoader."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from graphrag_kg.core.config import KGConfig
from graphrag_kg.core.config_loader import ConfigLoader
from graphrag_kg.core.errors import ConfigError, ConfigValidationError


class TestConfigLoaderBasics:
    """Basic loading tests."""

    def test_load_defaults(self, config_loader):
        """Loading with no file should return defaults."""
        config = config_loader.load()
        assert isinstance(config, KGConfig)
        assert config.project_name == "my-knowledge-graph"

    def test_load_from_yaml(self, config_loader, temp_yaml_config):
        """Loading from a valid YAML file should populate config."""
        config = config_loader.load(temp_yaml_config)
        assert config.project_name == "test-project"
        assert config.neo4j.username == "neo4j"
        assert config.chat_model == "gpt-4.1-mini"

    def test_load_nonexistent_file(self, config_loader):
        """Loading a nonexistent file should return defaults."""
        config = config_loader.load(Path("/nonexistent/settings.yaml"))
        assert isinstance(config, KGConfig)
        assert config.project_name == "my-knowledge-graph"

    def test_load_invalid_yaml(self, config_loader, temp_dir):
        """Loading invalid YAML should raise ConfigError."""
        bad_yaml = temp_dir / "bad.yaml"
        bad_yaml.write_text("{ invalid: yaml: : }")
        with pytest.raises(ConfigError):
            config_loader.load(bad_yaml)


class TestConfigLoaderProfiles:
    """Profile-based loading tests."""

    def test_load_fast_profile(self, config_loader):
        config = config_loader.load_profile("fast")
        assert config.chat_model == "gpt-4.1-mini"
        assert config.pipeline.max_workers == 8

    def test_load_production_profile(self, config_loader):
        config = config_loader.load_profile("production")
        assert config.chat_model == "gpt-4.1"
        assert config.query.default_method == "drift"

    def test_load_unknown_profile(self, config_loader):
        with pytest.raises(ConfigError, match="Unknown profile"):
            config_loader.load_profile("nonexistent")

    def test_profile_merge_with_yaml(self, config_loader, temp_yaml_config):
        """Profile should be mergeable with YAML overrides."""
        config = config_loader.load_with_overrides(
            config_path=temp_yaml_config,
            profile="fast",
        )
        # YAML should override profile values
        assert config.project_name == "test-project"  # From YAML
        assert config.chat_model == "gpt-4.1-mini"  # From profile (not in YAML kg section)


class TestEnvVarResolution:
    """Environment variable substitution tests."""

    def test_env_var_substitution(self, config_loader, temp_dir):
        """${VAR} should be replaced with environment variable value."""
        os.environ["TEST_DB_PASSWORD"] = "secret123"
        try:
            yaml_content = """\
kg:
  project_name: "env-test"
  neo4j:
    password: "${TEST_DB_PASSWORD}"
"""
            config_path = temp_dir / "env_settings.yaml"
            config_path.write_text(yaml_content)

            config = config_loader.load(config_path)
            assert config.neo4j.password == "secret123"
        finally:
            del os.environ["TEST_DB_PASSWORD"]

    def test_env_var_with_default(self, config_loader, temp_dir):
        """${VAR:default} should use default when env var is missing."""
        yaml_content = """\
kg:
  project_name: "default-test"
  neo4j:
    uri: "${MISSING_VAR:bolt://default:7687}"
"""
        config_path = temp_dir / "default_settings.yaml"
        config_path.write_text(yaml_content)

        config = config_loader.load(config_path)
        assert config.neo4j.uri == "bolt://default:7687"

    def test_missing_env_var_no_default(self, config_loader, temp_dir):
        """${VAR} without default should be kept as-is when env var missing."""
        yaml_content = """\
kg:
  project_name: "missing-test"
  neo4j:
    uri: "${MISSING_VAR}"
"""
        config_path = temp_dir / "missing_settings.yaml"
        config_path.write_text(yaml_content)

        config = config_loader.load(config_path)
        assert "${MISSING_VAR}" in config.neo4j.uri


class TestCLIOverrides:
    """CLI override tests."""

    def test_top_level_override(self, config_loader, temp_yaml_config):
        """CLI overrides should take highest precedence."""
        config = config_loader.load_with_overrides(
            config_path=temp_yaml_config,
            overrides={"project_name": "cli-override"},
        )
        assert config.project_name == "cli-override"

    def test_nested_override(self, config_loader, temp_yaml_config):
        """Dot-notation overrides should work for nested fields."""
        config = config_loader.load_with_overrides(
            config_path=temp_yaml_config,
            overrides={"neo4j.uri": "bolt://custom:9999"},
        )
        assert config.neo4j.uri == "bolt://custom:9999"


class TestValidation:
    """Config validation tests."""

    def test_validate_missing_api_key(self, config_loader):
        """Should warn about missing API key."""
        config = config_loader.load()
        warnings = config_loader.validate(config)
        assert any("API key" in w for w in warnings)

    def test_validate_default_password(self, config_loader):
        """Should warn about default Neo4j password."""
        config = config_loader.load()
        warnings = config_loader.validate(config)
        assert any("default 'password'" in w for w in warnings)

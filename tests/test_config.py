import pytest
from pydantic import ValidationError

from literature_buddy.config import available_profiles, load_config


def test_all_profiles_validate():
    for name in available_profiles():
        assert load_config(profile=name, environ={}).llm.model


def test_env_override_and_types():
    c = load_config(profile="lite", environ={"LB_RETRIEVAL__TOP_K": "12", "LB_OFFLINE": "true"})
    assert c.retrieval.top_k == 12 and c.offline is True


def test_unknown_profile():
    with pytest.raises(ValueError):
        load_config(profile="nope", environ={})


def test_unknown_key_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("llm:\n  modle: typo\n")
    with pytest.raises(ValidationError):
        load_config(profile="lite", config_path=p, environ={})

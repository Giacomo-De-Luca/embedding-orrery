"""Tests for deploy/hf-space/deploy.py — Space card rendering + site-URL variable.

The deploy script is not a package (it lives under the hyphenated
``deploy/hf-space/``), so it is loaded by path. Nothing here touches the Hub:
the HfApi surface the script uses is replaced by an in-memory fake.
"""

import fnmatch
import importlib.util
import re
from pathlib import Path
from types import ModuleType, SimpleNamespace

import httpx
import pytest
from huggingface_hub.errors import HfHubHTTPError

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPLOY_SCRIPT = REPO_ROOT / "deploy" / "hf-space" / "deploy.py"
REPO_ID = "SomeUser/orrery-demo"
SPACE_URL = f"https://huggingface.co/spaces/{REPO_ID}"


def _hub_http_error(status: int) -> HfHubHTTPError:
    """An HfHubHTTPError as the Hub client would raise it (needs a real response)."""
    request = httpx.Request("GET", f"https://huggingface.co/api/spaces/{REPO_ID}/variables")
    return HfHubHTTPError(f"{status} error", response=httpx.Response(status, request=request))


@pytest.fixture(scope="module")
def deploy() -> ModuleType:
    spec = importlib.util.spec_from_file_location("hf_space_deploy", DEPLOY_SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSpaceApi:
    """Minimal stand-in for the HfApi methods ``ensure_site_url_variable`` uses."""

    def __init__(self, variables: dict[str, str] | None = None, fail_with: Exception | None = None):
        self.variables = dict(variables or {})
        self.fail_with = fail_with
        self.added: list[tuple[str, str, str]] = []

    def get_space_variables(self, repo_id: str, **_: object) -> dict[str, SimpleNamespace]:
        if self.fail_with is not None:
            raise self.fail_with
        return {key: SimpleNamespace(value=value) for key, value in self.variables.items()}

    def add_space_variable(self, repo_id: str, key: str, value: str, **_: object) -> None:
        self.added.append((repo_id, key, value))
        self.variables[key] = value


def _frontmatter_value(text: str, key: str) -> str:
    match = re.search(rf"^{key}:\s*(.+?)\s*$", text, re.MULTILINE)
    assert match is not None, f"{key!r} missing from the rendered Space README"
    return match.group(1)


def test_space_direct_url_is_the_hf_space_host(deploy):
    assert (
        deploy.space_direct_url("SomeUser/orrery_demo") == "https://someuser-orrery-demo.hf.space"
    )


def test_rendered_readme_resolves_every_placeholder(deploy):
    text = deploy.render_space_readme(REPO_ID).decode("utf-8")
    assert text.startswith("---\n")
    assert "{{" not in text and "}}" not in text
    assert f"{SPACE_URL}?tour=1" in text
    assert "https://someuser-orrery-demo.hf.space" in text


def test_rendered_readme_thumbnail_is_a_hub_url_to_an_uploaded_file(deploy):
    """The social card must point at a file that (a) exists in the repo and
    (b) is not excluded from the Space upload by IGNORE_PATTERNS."""
    text = deploy.render_space_readme(REPO_ID).decode("utf-8")
    thumbnail = _frontmatter_value(text, "thumbnail")

    prefix = f"{SPACE_URL}/resolve/main/"
    assert thumbnail.startswith(prefix), thumbnail
    rel_path = thumbnail[len(prefix) :]
    assert rel_path == deploy.SOCIAL_PREVIEW_PATH
    assert (REPO_ROOT / rel_path).is_file(), f"{rel_path} is missing from the repo"

    excluded = [p for p in deploy.IGNORE_PATTERNS if fnmatch.fnmatch(rel_path, p)]
    assert not excluded, f"{rel_path} would be filtered out of the Space upload by {excluded}"


def test_validate_rejects_relative_thumbnail(deploy):
    text = "---\ntitle: x\nthumbnail: /preview.jpg\n---\n"
    with pytest.raises(SystemExit, match="thumbnail"):
        deploy.validate_space_readme(text)


def test_validate_rejects_unresolved_thumbnail_placeholder(deploy):
    text = "---\ntitle: x\nthumbnail: {{SOCIAL_PREVIEW_URL}}\n---\n"
    with pytest.raises(SystemExit, match="thumbnail"):
        deploy.validate_space_readme(text)


def test_validate_rejects_overlong_short_description(deploy):
    text = "---\ntitle: x\nshort_description: " + "a" * 61 + "\n---\n"
    with pytest.raises(SystemExit, match="short_description"):
        deploy.validate_space_readme(text)


def test_site_url_variable_is_set_when_missing(deploy):
    api = FakeSpaceApi()
    deploy.ensure_site_url_variable(api, REPO_ID)
    assert api.added == [
        (REPO_ID, deploy.SITE_URL_VARIABLE, "https://someuser-orrery-demo.hf.space")
    ]


def test_site_url_variable_is_corrected_when_stale(deploy):
    api = FakeSpaceApi({deploy.SITE_URL_VARIABLE: "https://old-name.hf.space"})
    deploy.ensure_site_url_variable(api, REPO_ID)
    assert [value for _, _, value in api.added] == ["https://someuser-orrery-demo.hf.space"]


def test_site_url_variable_is_left_alone_when_current(deploy):
    api = FakeSpaceApi({deploy.SITE_URL_VARIABLE: "https://someuser-orrery-demo.hf.space"})
    deploy.ensure_site_url_variable(api, REPO_ID)
    assert api.added == []


def test_site_url_variable_failure_warns_instead_of_aborting(deploy, capsys):
    api = FakeSpaceApi(fail_with=_hub_http_error(403))
    deploy.ensure_site_url_variable(api, REPO_ID)  # must not raise
    assert api.added == []
    assert deploy.SITE_URL_VARIABLE in capsys.readouterr().err

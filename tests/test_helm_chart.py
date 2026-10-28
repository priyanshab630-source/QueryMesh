"""Helm chart structure checks that do not need a cluster (or helm itself).

Every template is stripped of its Go template directives and then parsed as YAML,
which catches the mistakes that actually break a release: wrong indentation,
misplaced lists, unquoted values. Rendering the real manifests is a `helm
template` call, which needs helm installed - see deploy/README.md.
"""

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML is needed for the chart check")

CHART = Path("deploy/helm/querymesh")
DIRECTIVE = re.compile(r"\{\{.*?\}\}")


def _strip_templates(text: str) -> str:
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            continue                     # pure control-flow line
        lines.append(DIRECTIVE.sub("placeholder", line))
    return "\n".join(lines)


def test_chart_metadata():
    chart = yaml.safe_load((CHART / "Chart.yaml").read_text())
    assert chart["apiVersion"] == "v2"
    assert chart["name"] == "querymesh"
    assert chart["version"]


def test_values_cover_what_templates_use():
    values = yaml.safe_load((CHART / "values.yaml").read_text())
    for key in ("image", "service", "ingress", "resources", "autoscaling",
                "env", "serviceAccount", "externalSecret"):
        assert key in values, key
    assert values["service"]["targetPort"] == 8000
    assert values["env"]["USE_LANGGRAPH"] == "1"
    assert values["livenessProbe"]["path"] == "/health"
    assert values["readinessProbe"]["path"] == "/health"


def test_templates_parse_as_yaml():
    templates = sorted((CHART / "templates").glob("*.yaml"))
    assert templates, "no templates found"
    for template in templates:
        rendered = _strip_templates(template.read_text())
        documents = [doc for doc in yaml.safe_load_all(rendered) if doc]
        assert documents, f"{template.name} produced no documents"
        for document in documents:
            assert "apiVersion" in document and "kind" in document


def test_deployment_is_hardened():
    deployment = yaml.safe_load(
        _strip_templates((CHART / "templates" / "deployment.yaml").read_text()))
    container = deployment["spec"]["template"]["spec"]["containers"][0]
    assert "httpGet" in container["livenessProbe"]
    assert "httpGet" in container["readinessProbe"]
    assert container["securityContext"]["allowPrivilegeEscalation"] is False
    assert deployment["spec"]["template"]["spec"]["securityContext"]["runAsNonRoot"] is True
    # the probe path is templated from values, not hardcoded
    source = (CHART / "templates" / "deployment.yaml").read_text()
    assert ".Values.livenessProbe.path" in source

    # secrets must come from a secretRef, never from the ConfigMap
    env_from = container["envFrom"]
    assert any("secretRef" in entry for entry in env_from)


def test_no_secrets_are_hardcoded():
    for path in [CHART / "values.yaml"] + sorted((CHART / "templates").glob("*.yaml")):
        text = path.read_text()
        for marker in ("SARVAM_API_KEY=", "JWT_SECRET=", "AKIA", "sk-"):
            assert marker not in text, f"{path.name} looks like it contains a secret"

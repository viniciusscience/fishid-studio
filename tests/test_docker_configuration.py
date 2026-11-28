import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO_ROOT = Path(__file__).resolve().parents[1]
COMPOSE = REPO_ROOT / "docker-compose.yml"
BACKEND_DOCKERFILE = REPO_ROOT / "backend" / "Dockerfile"
FRONTEND_DOCKERFILE = REPO_ROOT / "frontend" / "Dockerfile"
BACKEND_MAIN = REPO_ROOT / "backend" / "main.py"
FRONTEND_APP = REPO_ROOT / "frontend" / "app.py"


def _compose_text():
    return COMPOSE.read_text(encoding="utf-8")


def _port_mappings():
    return re.findall(r'-\s*"?(\d+):(\d+)"?', _compose_text())


def test_compose_exposes_host_ports_8000_and_8080():
    mappings = _port_mappings()
    assert ("8000", "8080") in mappings, "backend must publish host port 8000 to container port 8080"
    assert ("8080", "8080") in mappings, "frontend must publish host port 8080 to container port 8080"


def test_compose_backend_url_matches_backend_container_port():
    text = _compose_text()
    match = re.search(r"BACKEND_URL=http://backend:(\d+)", text)
    assert match, "frontend must set BACKEND_URL pointing at the backend service"
    backend_port = match.group(1)
    assert ("8080", backend_port) in _port_mappings(), (
        "BACKEND_URL port must match the backend container port"
    )


def test_backend_dockerfile_exposes_port_and_valid_entrypoint():
    text = BACKEND_DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^EXPOSE\s+8080\s*$", text, re.MULTILINE), "backend must EXPOSE 8080"
    cmd_match = re.search(r'^CMD\s+\[(.*)\]\s*$', text, re.MULTILINE)
    assert cmd_match, "backend must declare a CMD entrypoint"
    cmd = cmd_match.group(1)
    assert "uvicorn" in cmd and "main:app" in cmd, "backend entrypoint must serve main:app via uvicorn"
    assert "8080" in cmd, "backend entrypoint must bind port 8080"
    assert "COPY main.py" in text, "backend image must copy the entrypoint module"
    assert BACKEND_MAIN.is_file(), "backend entrypoint module must exist"


def test_frontend_dockerfile_exposes_port_and_valid_entrypoint():
    text = FRONTEND_DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^EXPOSE\s+8080\s*$", text, re.MULTILINE), "frontend must EXPOSE 8080"
    cmd_match = re.search(r'^CMD\s+\[(.*)\]\s*$', text, re.MULTILINE)
    assert cmd_match, "frontend must declare a CMD entrypoint"
    cmd = cmd_match.group(1)
    assert "streamlit" in cmd and "app.py" in cmd, "frontend entrypoint must run streamlit on app.py"
    assert "8080" in cmd, "frontend entrypoint must bind port 8080"
    assert "COPY app.py" in text, "frontend image must copy the entrypoint module"
    assert FRONTEND_APP.is_file(), "frontend entrypoint module must exist"


def test_backend_exposes_health_route_for_container_healthcheck():
    text = BACKEND_MAIN.read_text(encoding="utf-8")
    assert re.search(r'"/health"', text), "backend must register the /health route used by healthchecks"
    assert re.search(r"def health\(", text), "backend must implement the health handler"

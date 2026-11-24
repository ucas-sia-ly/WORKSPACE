"""Health checks shared by generation clients and the demo preflight."""

from urllib.parse import urlparse

import requests


def health_url(api_url: str) -> str:
    parsed = urlparse(api_url)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def probe_service(session, api_url: str) -> tuple[bool, str]:
    url = health_url(api_url)
    try:
        response = session.get(url, timeout=2)
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        return False, f"{url}: {exc}"
    if not isinstance(data, dict):
        return False, f"{url}: expected a JSON object"
    if data.get("error"):
        raise RuntimeError(f"{url}: model initialization failed: {data['error']}")
    ready = (
        data.get("status") == "ok"
        and data.get("model_loaded") is True
        and data.get("generator_ready", data.get("model_loaded")) is True
    )
    return ready, (
        f"{url}: status={data.get('status')!r}, "
        f"model_loaded={data.get('model_loaded')!r}, "
        f"generator_ready={data.get('generator_ready', data.get('model_loaded'))!r}"
    )

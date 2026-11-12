"""Where the server listens, from the environment.

Without ``HOST`` the server binds to 127.0.0.1 (one local user; loopback host names only; the
page cannot be framed). With ``HOST`` (e.g. ``HOST=0.0.0.0`` in the container of a hosted demo)
it binds there and, unless ``HOST`` is a loopback address, runs in public mode: research routes
off, host names from ``ALLOWED_HOSTS`` (default: any), the page may be framed by
``FRAME_ANCESTORS`` (default: https://huggingface.co, which shows a Space in a frame) and, with
``TRUST_PROXY=1``, client addresses for rate limits come from X-Forwarded-For behind
``PROXY_HOPS`` proxies. ``PORT`` sets the port.

Tunnel mode (``scripts/serve.py --tunnel``) keeps the loopback address, so the app is reached
only through a tunnel client running on this computer (e.g. cloudflared or ngrok forwarding to
127.0.0.1), and switches on public mode: any host name (the tunnel's), client addresses from the
tunnel's CF-Connecting-IP header (else X-Forwarded-For), the research routes off. Only in tunnel
mode is CF-Connecting-IP read: elsewhere a client could send it to choose its own rate-limit key.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field

LOOPBACK_ADDRESS = "127.0.0.1"
LOOPBACK_NAMES = ("127.0.0.1", "localhost", "::1")
DEFAULT_FRAME_ANCESTORS = "https://huggingface.co"
TUNNEL_CLIENT_HEADER = "cf-connecting-ip"  # set by Cloudflare's edge for every request
_SOURCE_RE = re.compile(r"^(?:'none'|'self'|https://[A-Za-z0-9*.-]+(?::\d{1,5})?)$")


@dataclass
class Binding:
    host: str
    port: int
    public: bool
    allowed_hosts: list[str] = field(default_factory=lambda: ["127.0.0.1", "localhost"])
    frame_ancestors: str = "'none'"
    trust_forwarded: bool = False
    forwarded_hops: int = 1
    client_ip_header: str | None = None  # a header naming the client (tunnel mode only)
    tunnel: bool = False


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in ("1", "true", "yes", "on")


def frame_ancestors_value(raw: str | None, default: str) -> str:
    """A CSP frame-ancestors source list; anything but 'none', 'self' or https origins is
    refused (so the header cannot be widened or broken by a typo)."""
    if raw is None or not raw.strip():
        return default
    parts = raw.split()
    if not all(_SOURCE_RE.match(p) for p in parts):
        raise ValueError(f"FRAME_ANCESTORS must list 'none', 'self' or https origins, got {raw!r}")
    return " ".join(parts)


def _hops(env: Mapping[str, str]) -> int:
    raw = (env.get("PROXY_HOPS") or "").strip()
    return int(raw) if raw.isdigit() and int(raw) > 0 else 1


def binding_from_env(env: Mapping[str, str], default_port: int = 8000) -> Binding:
    host = (env.get("HOST") or "").strip()
    port_raw = (env.get("PORT") or "").strip()
    port = int(port_raw) if port_raw.isdigit() else default_port
    if not host:
        return Binding(LOOPBACK_ADDRESS, port, False)
    public = host not in LOOPBACK_NAMES
    if not public:
        return Binding(host, port, False)
    allowed = [h.strip() for h in (env.get("ALLOWED_HOSTS") or "").split(",") if h.strip()]
    return Binding(
        host,
        port,
        True,
        allowed or ["*"],
        frame_ancestors_value(env.get("FRAME_ANCESTORS"), DEFAULT_FRAME_ANCESTORS),
        _truthy(env.get("TRUST_PROXY")),
        _hops(env),
    )


def tunnel_binding(env: Mapping[str, str], default_port: int = 8000) -> Binding:
    """Loopback address, public mode, any host name, client addresses from the tunnel."""
    port_raw = (env.get("PORT") or "").strip()
    return Binding(
        LOOPBACK_ADDRESS,
        int(port_raw) if port_raw.isdigit() else default_port,
        True,
        ["*"],
        frame_ancestors_value(env.get("FRAME_ANCESTORS"), "'none'"),
        True,
        _hops(env),
        client_ip_header=TUNNEL_CLIENT_HEADER,
        tunnel=True,
    )


def resolve_binding(env: Mapping[str, str], default_port: int, *, tunnel: bool) -> Binding:
    """Tunnel mode when asked for (``HOST`` is then ignored), else :func:`binding_from_env`."""
    return tunnel_binding(env, default_port) if tunnel else binding_from_env(env, default_port)

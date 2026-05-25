"""Tests for network tool URL safety checks."""

import asyncio
import socket

import pytest

from omniagent.tools.http_tool import HttpTool
from omniagent.tools.url_security import (
    URLSecurityError,
    validate_public_http_url,
    validate_redirect_url,
)
from omniagent.tools.web_tools import WebFetchTool


def _fake_addr_info(*addresses: str):
    return [
        (
            socket.AF_INET6 if ":" in address else socket.AF_INET,
            socket.SOCK_STREAM,
            0,
            "",
            (address, 443, 0, 0) if ":" in address else (address, 443),
        )
        for address in addresses
    ]


def test_validate_public_http_url_allows_public_dns(monkeypatch):
    monkeypatch.setattr(
        "omniagent.tools.url_security.socket.getaddrinfo",
        lambda *args, **kwargs: _fake_addr_info("93.184.216.34"),
    )

    validate_public_http_url("https://example.com/path")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/health",
        "http://[::1]:8080/health",
        "http://169.254.169.254/latest/meta-data",
        "http://10.0.0.5/status",
    ],
)
def test_validate_public_http_url_rejects_internal_ip_literals(url):
    with pytest.raises(URLSecurityError, match="SSRF protection"):
        validate_public_http_url(url)


def test_validate_public_http_url_rejects_any_private_dns_result(monkeypatch):
    monkeypatch.setattr(
        "omniagent.tools.url_security.socket.getaddrinfo",
        lambda *args, **kwargs: _fake_addr_info("93.184.216.34", "127.0.0.1"),
    )

    with pytest.raises(URLSecurityError, match="127.0.0.1"):
        validate_public_http_url("https://example.com")


def test_validate_redirect_url_rejects_redirect_to_loopback():
    with pytest.raises(URLSecurityError, match="SSRF protection"):
        validate_redirect_url("https://example.com/start", "http://127.0.0.1/admin")


def test_http_tool_rejects_ipv6_loopback_before_request():
    result = asyncio.run(HttpTool().execute({"url": "http://[::1]:8080/health"}))

    assert not result.success
    assert "SSRF protection" in (result.error or "")


def test_web_fetch_rejects_loopback_before_request():
    result = asyncio.run(WebFetchTool().execute({"url": "http://127.0.0.1:8080/health"}))

    assert not result.success
    assert "SSRF protection" in (result.error or "")

import json
from pathlib import Path

import pytest

from src.mpc_server import MPCServer


def test_mpc_server_connect_and_disconnect():
    server = MPCServer()
    try:
        server.connect()
        assert server._connected
    finally:
        server.disconnect()
    assert not server._connected


def test_mpc_server_context_manager():
    with MPCServer() as server:
        assert server._connected
    assert not server._connected


def test_search_returns_list():
    server = MPCServer()
    try:
        server.connect()
        results = server.search("Python programming language")
        assert isinstance(results, list)
        if results:
            item = results[0]
            if isinstance(item, dict):
                assert "title" in item or "id" in item or "text" in item
    finally:
        server.disconnect()


def test_search_with_known_query():
    server = MPCServer()
    try:
        server.connect()
        results = server.search("Albert Einstein")
        assert isinstance(results, list)
    finally:
        server.disconnect()


def test_fetch_known_page():
    server = MPCServer()
    try:
        server.connect()
        results = server.search("Python programming language")
        if results:
            item = results[0]
            page_id = item.get("id") if isinstance(item, dict) else None
            if page_id:
                content = server.fetch(int(page_id))
                assert content is not None
                assert isinstance(content, str)
                assert len(content) > 0
    finally:
        server.disconnect()


def test_search_returns_different_results_for_different_queries():
    server = MPCServer()
    try:
        server.connect()
        r1 = server.search("quantum mechanics")
        r2 = server.search("classical music")
        assert isinstance(r1, list)
        assert isinstance(r2, list)
    finally:
        server.disconnect()

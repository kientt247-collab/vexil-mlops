import hashlib

import pytest

from vexil import hashing, smoke


def test_explorer_url_quotes_sql():
    url = smoke.explorer_url("SELECT 1 AS n")
    assert url == "https://api.opendota.com/api/explorer?sql=SELECT%201%20AS%20n"


def test_parse_count_reads_single_row():
    assert smoke.parse_count({"rows": [{"n": "42"}]}) == 42


def test_parse_count_rejects_error_and_extra_rows():
    with pytest.raises(RuntimeError):
        smoke.parse_count({"err": "syntax error", "rows": []})
    with pytest.raises(ValueError):
        smoke.parse_count({"rows": [{"n": 1}, {"n": 2}]})


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"vexil" * 1000)
    assert hashing.sha256_file(path) == hashlib.sha256(b"vexil" * 1000).hexdigest()


def test_render_page_shows_the_count():
    page = smoke.render_page({"run_at_utc": "2026-11-21 00:00:00", "matches_last_7_days": 7})
    assert "last 7 days: 7." in page

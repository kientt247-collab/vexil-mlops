import hashlib

from vexil import hashing


def test_sha256_file_matches_hashlib(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"vexil" * 1000)
    assert hashing.sha256_file(path) == hashlib.sha256(b"vexil" * 1000).hexdigest()

"""Thin wrapper around sqlite-vec for dense vector search.

Usage pattern confirmed against github.com/asg017/sqlite-vec's README
(2026-08-11): load the extension via `sqlite_vec.load(conn)`, create a
`vec0` virtual table sized to the embedding dimension, insert vectors as
JSON arrays, and query with `... WHERE embedding MATCH ? ORDER BY distance
LIMIT k`.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass


@dataclass
class SearchResult:
    chunk_id: str
    distance: float


class VectorStore:
    def __init__(self, dim: int, db_path: str = ":memory:") -> None:
        import sqlite_vec

        self.dim = dim
        self._conn = sqlite3.connect(db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)

        self._conn.execute(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
                embedding float[{dim}]
            )
            """
        )
        # rowid <-> chunk_id mapping, since vec0 tables are keyed by rowid.
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS chunk_ids (
                rowid INTEGER PRIMARY KEY,
                chunk_id TEXT UNIQUE NOT NULL
            )
            """
        )
        self._conn.commit()

    def _insert(self, chunk_id: str, vector: list[float]) -> None:
        """Insert without committing. Used by both add() (commits once
        itself) and add_many() (commits once for the whole batch).
        """
        if len(vector) != self.dim:
            raise ValueError(f"Expected vector of dim {self.dim}, got {len(vector)}")
        cur = self._conn.execute(
            "INSERT INTO chunk_ids (chunk_id) VALUES (?)", (chunk_id,)
        )
        rowid = cur.lastrowid
        self._conn.execute(
            "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)",
            (rowid, json.dumps(vector)),
        )

    def add(self, chunk_id: str, vector: list[float]) -> None:
        self._insert(chunk_id, vector)
        self._conn.commit()

    def add_many(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        # FIX (review #4): commit once for the whole batch instead of once
        # per row (previously add_many() looped calling add(), which
        # commits -- i.e. fsyncs -- after every single insert). At the
        # pilot's 54 chunks this was invisible; at the full 300-chunk
        # target it's ~2x more fsync-bound commits than necessary.
        for chunk_id, vector in zip(chunk_ids, vectors):
            self._insert(chunk_id, vector)
        self._conn.commit()

    def search(self, query_vector: list[float], k: int = 5) -> list[SearchResult]:
        if len(query_vector) != self.dim:
            raise ValueError(
                f"Expected query vector of dim {self.dim}, got {len(query_vector)}"
            )
        # This sqlite-vec version requires the k-nearest-neighbors count to
        # be expressed as a `k = ?` constraint inside the WHERE clause, not
        # a trailing SQL LIMIT -- and it must be run directly against the
        # vec0 table (a JOIN in the same query breaks its ability to detect
        # the KNN constraint). Look up chunk_id separately after.
        rows = self._conn.execute(
            """
            SELECT rowid, distance
            FROM vec_chunks
            WHERE embedding MATCH ? AND k = ?
            ORDER BY distance
            """,
            (json.dumps(query_vector), k),
        ).fetchall()

        results = []
        for rowid, distance in rows:
            chunk_id = self._conn.execute(
                "SELECT chunk_id FROM chunk_ids WHERE rowid = ?", (rowid,)
            ).fetchone()[0]
            results.append(SearchResult(chunk_id=chunk_id, distance=distance))
        return results

    def close(self) -> None:
        self._conn.close()

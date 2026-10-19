"""Download labelled sky cutouts into one resumable SQLite dataset.

The database stores one JPEG blob per object, rather than creating thousands of
files.  SQLite also keeps the download status and source metadata together,
which makes interrupted downloads safe to resume and the dataset auditable.
"""

from __future__ import annotations

import argparse
import ssl
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd
import certifi


SCHEMA = """
CREATE TABLE IF NOT EXISTS cutouts (
    object_id TEXT PRIMARY KEY,
    ra REAL NOT NULL,
    dec REAL NOT NULL,
    label INTEGER NOT NULL,
    split TEXT,
    size_arcsec REAL,
    provider TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    fov_arcsec REAL NOT NULL,
    url TEXT NOT NULL,
    mime_type TEXT,
    image BLOB,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    downloaded_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_cutouts_status ON cutouts(status);
CREATE INDEX IF NOT EXISTS idx_cutouts_label ON cutouts(label);
"""


@dataclass(frozen=True)
class DownloadResult:
    object_id: str
    image: bytes | None
    mime_type: str | None
    error: str | None


def cutout_url(
    ra: float,
    dec: float,
    fov_arcsec: float,
    width: int = 128,
    height: int = 128,
    provider: str = "hips2fits",
) -> str:
    """Build a JPEG request for one of the supplied cutout services."""
    if provider == "hips2fits":
        return (
            "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
            f"?hips=CDS%2FP%2FSDSS9%2Fcolor&ra={ra:.8f}&dec={dec:.8f}"
            f"&fov={fov_arcsec:.5f}&width={width}&height={height}"
            "&projection=TAN&format=jpg"
        )
    if provider == "sdss":
        # SDSS scale is expressed in arcsec/pixel.
        scale = fov_arcsec / max(width, height)
        return (
            "https://skyserver.sdss.org/dr19/SkyServerWS/ImgCutout/getjpeg"
            f"?ra={ra:.8f}&dec={dec:.8f}&scale={scale:.6f}"
            f"&width={width}&height={height}"
        )
    raise ValueError(f"Unsupported provider: {provider}")


def _download_one(
    object_id: str,
    url: str,
    timeout: float,
    retries: int,
) -> DownloadResult:
    """Download and minimally validate one image, with bounded retries."""
    last_error = "download failed"
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    for attempt in range(retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "galaxies-cnns/1.0"})
            with urlopen(request, timeout=timeout, context=ssl_context) as response:
                image = response.read()
                mime_type = response.headers.get_content_type()
            if not image:
                raise ValueError("empty response")
            if not (image.startswith(b"\xff\xd8") or image.startswith(b"\x89PNG")):
                raise ValueError("response is not a JPEG or PNG image")
            return DownloadResult(object_id, image, mime_type, None)
        except (HTTPError, URLError, TimeoutError, ValueError, OSError) as exc:
            last_error = str(exc)[:500]
            if attempt < retries:
                time.sleep(2**attempt)
    return DownloadResult(object_id, None, None, last_error)


def _connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.executescript(SCHEMA)
    return connection


def prepare_manifest(
    data: pd.DataFrame,
    database: str | Path,
    *,
    provider: str = "hips2fits",
    width: int = 128,
    height: int = 128,
    min_fov_arcsec: float = 16.0,
    max_fov_arcsec: float = 40.0,
    fov_multiplier: float = 3.0,
) -> int:
    """Insert candidate objects and return the number newly added."""
    required = {"ra", "dec", "spiral", "elliptical"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    object_column = "specobjid" if "specobjid" in data else "objid"
    if object_column not in data:
        raise ValueError("Data must contain specobjid or objid")

    connection = _connect(Path(database))
    added = 0
    try:
        for row in data.itertuples(index=False):
            values = row._asdict()
            object_id = str(values[object_column])
            if pd.isna(values["ra"]) or pd.isna(values["dec"]):
                continue
            size = values.get("size")
            size_arcsec = None if pd.isna(size) else float(size)
            fov = min_fov_arcsec if size_arcsec is None else max(
                min_fov_arcsec, min(max_fov_arcsec, fov_multiplier * size_arcsec)
            )
            label = int(values["spiral"] > values["elliptical"])
            split = values.get("split")
            split = None if split is None or pd.isna(split) else str(split)
            url = cutout_url(
                float(values["ra"]), float(values["dec"]), fov, width, height, provider
            )
            cursor = connection.execute(
                """INSERT OR IGNORE INTO cutouts
                (object_id, ra, dec, label, split, size_arcsec, provider, width,
                 height, fov_arcsec, url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    object_id, float(values["ra"]), float(values["dec"]), label,
                    split, size_arcsec, provider, width, height, fov, url,
                ),
            )
            added += cursor.rowcount
        connection.commit()
    finally:
        connection.close()
    return added


def download_pending(
    database: str | Path,
    *,
    workers: int = 16,
    timeout: float = 10.0,
    retries: int = 1,
    retry_errors: bool = False,
    progress_every: int = 10,
) -> tuple[int, int]:
    """Download pending rows in parallel; return ``(successful, failed)``."""
    connection = _connect(Path(database))
    statuses = ("pending", "error") if retry_errors else ("pending",)
    placeholders = ",".join("?" for _ in statuses)
    rows = connection.execute(
        f"SELECT object_id, url FROM cutouts WHERE status IN ({placeholders})",
        statuses,
    ).fetchall()
    connection.close()

    successful = failed = 0
    total = len(rows)
    connection = _connect(Path(database))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_one, object_id, url, timeout, retries): object_id
            for object_id, url in rows
        }
        for future in as_completed(futures):
            result = future.result()
            try:
                if result.image is None:
                    connection.execute(
                        """UPDATE cutouts
                        SET status='error', attempts=attempts+1, error=?
                        WHERE object_id=?""",
                        (result.error, result.object_id),
                    )
                    failed += 1
                else:
                    connection.execute(
                        """UPDATE cutouts
                        SET status='downloaded', attempts=attempts+1, error=NULL,
                            mime_type=?, image=?, downloaded_at=?
                        WHERE object_id=?""",
                        (
                            result.mime_type, sqlite3.Binary(result.image),
                            datetime.now(timezone.utc).isoformat(), result.object_id,
                        ),
                    )
                    successful += 1
                connection.commit()
            finally:
                completed = successful + failed
                if completed % progress_every == 0 or completed == total:
                    print(
                        f"Progress: {completed}/{total} "
                        f"({completed / total:.0%}); successful={successful}; failed={failed}",
                        flush=True,
                    )
    connection.close()
    return successful, failed


def status(database: str | Path) -> dict[str, int]:
    """Return counts by download status without downloading anything."""
    connection = _connect(Path(database))
    try:
        rows = connection.execute(
            "SELECT status, COUNT(*) FROM cutouts GROUP BY status"
        ).fetchall()
        return dict(rows)
    finally:
        connection.close()


def iter_images(
    database: str | Path,
    labels: Iterable[int] = (0, 1),
    split: str | None = None,
):
    """Yield ``(object_id, image_bytes, label)`` for training data loaders."""
    labels = tuple(labels)
    connection = sqlite3.connect(database)
    placeholders = ",".join("?" for _ in labels)
    try:
        if split is None:
            query = f"""SELECT object_id, image, label FROM cutouts
                        WHERE status='downloaded' AND label IN ({placeholders})
                        ORDER BY object_id"""
            parameters = labels
        else:
            query = f"""SELECT object_id, image, label FROM cutouts
                        WHERE status='downloaded' AND label IN ({placeholders})
                        AND split=? ORDER BY object_id"""
            parameters = (*labels, split)
        yield from connection.execute(query, parameters)
    finally:
        connection.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, nargs="?")
    parser.add_argument(
        "--database", type=Path, default=Path("datasets/images/cutouts.sqlite")
    )
    parser.add_argument("--provider", choices=("hips2fits", "sdss"), default="hips2fits")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        if not args.database.exists():
            raise SystemExit(f"Database not found: {args.database}")
        print(status(args.database))
        return
    if args.csv is None:
        parser.error("csv is required unless --status is used")
    data = pd.read_csv(args.csv)
    if args.limit is not None:
        data = data.head(args.limit)
    added = prepare_manifest(data, args.database, provider=args.provider)
    successful, failed = download_pending(
        args.database, workers=args.workers, retry_errors=args.retry_errors
    )
    print(f"Manifest: {added} new objects; downloaded={successful}; failed={failed}")


if __name__ == "__main__":
    main()

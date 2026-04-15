"""Modulo 0 — Download & Setup del dataset MEVID.

Scarica i video clip del dataset MEVID da S3.

Utilizzo::

    python src/00_setup_dataset.py                # tutti i 976 video (~127 GB)
    python src/00_setup_dataset.py --subset        # solo 3 video di test (~520 MB)
    python src/00_setup_dataset.py --dry-run       # mostra cosa farebbe
    python src/00_setup_dataset.py --urls file.txt  # URL personalizzati

Il download supporta resume automatico: i file parziali vengono
completati e i file già scaricati vengono saltati.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from tqdm import tqdm

from src.config import PROJECT_ROOT
from src.utils import setup_logging

log = logging.getLogger(__name__)

# ── Costanti ──────────────────────────────────────────────────

VIDEO_URLS_FILE: Path = PROJECT_ROOT / "data" / "raw" / "videos" / "mevid-v1-video-URLS.txt"
SUBSET_URLS_FILE: Path = PROJECT_ROOT / "data" / "raw" / "videos" / "mevid-subset-urls.txt"
VIDEO_OUTPUT_DIR: Path = PROJECT_ROOT / "data" / "raw" / "videos"
CHUNK_SIZE: int = 8 * 1024 * 1024  # 8 MB
REQUEST_TIMEOUT: int = 30


# ══════════════════════════════════════════════════════════════
# Download helpers
# ══════════════════════════════════════════════════════════════


def _fmt_size(n_bytes: int | float) -> str:
    """Formatta una dimensione in byte in stringa leggibile (KB/MB/GB)."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n_bytes) < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def download_file(url: str, dest_dir: Path, desc: str | None = None) -> bool:
    """
    Scarica un singolo file con resume e progress bar.

    Se il file esiste già con la dimensione corretta, viene saltato.
    Se esiste parzialmente e il server supporta Range, viene completato.

    Args:
        url:      URL remoto del file.
        dest_dir: Directory di destinazione.
        desc:     Etichetta per la progress bar.

    Returns:
        ``True`` se il file è stato scaricato o era già presente,
        ``False`` in caso di errore.
    """
    filename = Path(urlparse(url).path).name
    dest = dest_dir / filename
    display = desc or filename

    # HEAD per la dimensione remota
    try:
        head = requests.head(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        head.raise_for_status()
        remote_size = int(head.headers.get("Content-Length", 0))
    except requests.RequestException as exc:
        log.error("HEAD fallito per %s: %s", display, exc)
        return False

    # File già completo → skip
    if dest.exists():
        local_size = dest.stat().st_size
        if remote_size > 0 and local_size >= remote_size:
            log.info("Già scaricato: %s (%s)", display, _fmt_size(local_size))
            return True
    else:
        local_size = 0

    # Resume se possibile
    headers: dict[str, str] = {}
    mode = "wb"
    initial = 0
    if local_size > 0:
        accept_ranges = head.headers.get("Accept-Ranges", "none")
        if accept_ranges.lower() != "none":
            headers["Range"] = f"bytes={local_size}-"
            mode = "ab"
            initial = local_size
            log.info("Resume %s da %s", display, _fmt_size(local_size))

    # Download
    try:
        resp = requests.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()

        total = remote_size if remote_size > 0 else None
        with (
            open(dest, mode) as fout,
            tqdm(
                total=total,
                initial=initial,
                unit="B",
                unit_scale=True,
                desc=f"  {display}",
                leave=True,
            ) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                fout.write(chunk)
                pbar.update(len(chunk))

        return True

    except requests.RequestException as exc:
        log.error("Download fallito per %s: %s", display, exc)
        return False


# ══════════════════════════════════════════════════════════════
# Download video
# ══════════════════════════════════════════════════════════════


def download_videos(dry_run: bool = False, urls_file: Path | None = None) -> None:
    """
    Scarica i video MEVID elencati nel file di URL.

    I download sono sequenziali (ogni file ha la propria progress bar).
    I file già presenti vengono saltati automaticamente.

    Args:
        dry_run:   Se ``True``, mostra solo quanti file sarebbero scaricati.
        urls_file: File con gli URL da scaricare. Se omesso, usa il default.
    """
    if urls_file is None:
        urls_file = VIDEO_URLS_FILE

    if not urls_file.exists():
        log.error("File URL non trovato: %s", urls_file)
        sys.exit(1)

    urls = [
        line.strip()
        for line in urls_file.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not urls:
        log.error("Nessun URL trovato in %s", urls_file)
        return

    # Conta file già presenti
    existing = sum(1 for u in urls if (VIDEO_OUTPUT_DIR / Path(urlparse(u).path).name).exists())
    to_download = len(urls) - existing

    log.info(
        "Totale: %d video | già presenti: %d | da scaricare: %d",
        len(urls),
        existing,
        to_download,
    )

    if dry_run:
        log.info("(dry-run) Nessun download effettuato.")
        return

    VIDEO_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ok = 0
    fail = 0
    for i, url in enumerate(urls, 1):
        log.info("Video %d/%d", i, len(urls))
        if download_file(url, VIDEO_OUTPUT_DIR):
            ok += 1
        else:
            fail += 1

    log.info("Download completato: %d/%d ok, %d falliti", ok, len(urls), fail)


# ══════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Definisce e restituisce i parametri da riga di comando."""
    p = argparse.ArgumentParser(description="Modulo 0 — Download video MEVID")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Mostra quanti file sarebbero scaricati senza scaricarli",
    )

    source = p.add_mutually_exclusive_group()
    source.add_argument(
        "--subset",
        action="store_true",
        help="Scarica solo 4 video di test multi-camera (~520 MB invece di ~127 GB)",
    )
    source.add_argument(
        "--urls",
        type=str,
        default=None,
        help="File .txt con URL personalizzati (uno per riga)",
    )
    return p.parse_args()


def main() -> None:
    """Entry point: scarica i video MEVID."""
    setup_logging("setup_dataset")

    args = parse_args()
    t_start = time.time()

    # Determina sorgente URL
    if args.subset:
        urls_file = SUBSET_URLS_FILE
        log.info("═══ Download subset MEVID (4 video, ~520 MB) ═══")
    elif args.urls:
        urls_file = Path(args.urls)
        log.info("═══ Download video da %s ═══", urls_file)
    else:
        urls_file = None
        log.info("═══ Download video MEVID completo (~127 GB) ═══")

    download_videos(dry_run=args.dry_run, urls_file=urls_file)

    elapsed = time.time() - t_start
    log.info("Completato in %.1f s", elapsed)


if __name__ == "__main__":
    main()

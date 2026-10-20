"""Resume pending/error cutouts without touching fixed catalog splits."""
import argparse
from pathlib import Path
from scripts.cutouts import download_pending, status

if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--database', default='datasets/images/cutouts.sqlite')
    p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    # Require the prepared manifest: shared helpers may otherwise initialize an empty DB.
    if not Path(a.database).is_file():
        raise SystemExit('Image manifest does not exist. Run prepare_data or rebuild_images first.')
    # Resume the same object IDs and splits; do not rerun sampling to retry network failures.
    successful, failed = download_pending(a.database, workers=a.workers, retries=2, retry_errors=True)
    print(status(a.database))
    # Nonzero exit status stops the workflow before extraction hides incomplete downloads.
    if failed:
        raise SystemExit(f'{failed} downloads failed. Rerun this stage to retry.')

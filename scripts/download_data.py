"""Download large data directly over HTTPS, with atomic writes and SHA256 checks."""
import argparse
import hashlib
from pathlib import Path
import shutil
from urllib.request import urlopen


def sha256(path):
    """Return a file's SHA256 hex digest without loading it all into memory.

    path is any readable local file. Chunked hashing is also used to identify
    input files/source code for feature-cache reproducibility checks.
    """
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def download(url, output, expected):
    """Download a known HTTPS resource into output and verify its SHA256.

    expected is an independently supplied digest, not a checksum invented
    from the downloaded response. A matching existing file is reused; a
    mismatching file is preserved and rejected. Write via a temporary file
    and remove incomplete data on failure. This transfer does not resume.
    """
    if not url.startswith('https://'):
        raise ValueError('An HTTPS download URL is required.')
    output = Path(output)
    if output.exists():
        if sha256(output) == expected.lower():
            print(f'Reusing verified {output}')
            return
        raise ValueError(f'{output} exists with a different checksum; preserve or move it first.')
    output.parent.mkdir(parents=True, exist_ok=True)
    # A temporary pathname prevents an interrupted transfer looking like a complete input.
    temporary = output.with_suffix(output.suffix + '.part')
    try:
        with urlopen(url, timeout=120) as response, temporary.open('wb') as stream:
            if not response.geturl().startswith('https://'):
                raise ValueError('Download redirected away from HTTPS.')
            shutil.copyfileobj(response, stream)
        # Validate against the expected source checksum before accepting the final file.
        if sha256(temporary) != expected.lower():
            raise ValueError('SHA256 mismatch; download not accepted.')
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--sha256', required=True)
    args = parser.parse_args()
    download(args.url, args.output, args.sha256)

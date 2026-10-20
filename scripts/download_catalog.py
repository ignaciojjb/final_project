"""Download an explicitly joined DR19-hosted Galaxy Zoo/DR7 photometry sample.

Uses the public, read-only SkyServer SQL API (no SciServer account required).
The hosting release is DR19; this is legacy DR7 photometry with DR8 cross-match IDs.
"""
import argparse
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.request import Request, urlopen
import pandas as pd

ENDPOINT = 'https://skyserver.sdss.org/dr19/SkyServerWS/SearchTools/SqlSearch'
BANDS = 'ugriz'


def build_query(label, after, page_size):
    """Build one read-only SQL page for a single Galaxy Zoo class.

    label is 0 (elliptical) or 1 (spiral); after is the last exact specobjid,
    and page_size is bounded to protect the public query service. Both DR7
    and DR8 IDs participate in the join, avoiding every neighboring match.
    Return a SQL string with explicit columns, quality cuts and sorted IDs.
    No request is made here; first-ID pagination is not random sampling.
    """
    if label not in (0, 1) or after < 0 or not 1 <= page_size <= 1000:
        raise ValueError('Invalid label, cursor, or page size.')
    # Cast long IDs in SQL: floating-point conversion would lose object identity.
    columns = [
        'CONVERT(varchar(20),z.specobjid) AS specobjid',
        'CONVERT(varchar(20),z.objid) AS objid',
        'CONVERT(varchar(20),z.dr7objid) AS dr7objid',
        'CONVERT(varchar(20),p.dr8objid) AS dr8objid',
        'p.distance', 'p.ra', 'p.dec', 'p.modeDR7', 'p.modeDR8', 'p.nChild',
        'z.nvote','z.spiral','z.elliptical','z.uncertain',
        'p.petroMag_r','p.petroMagErr_r','p.petroR50_r','p.petroR90_r',
    ]
    columns += [f'p.{prefix}_{b}' for prefix in ['modelMag','modelMagErr','extinction'] for b in BANDS]
    # Use both ID systems; joining only dr7objid can return several DR8 neighbors.
    return f'''SELECT TOP {int(page_size)} {', '.join(columns)}
FROM ZooSpec z
JOIN PhotoObjDR7 p ON p.dr7objid=z.dr7objid AND p.dr8objid=z.objid
WHERE z.specobjid>{int(after)} AND z.spiral={int(label)} AND z.elliptical={1-int(label)}
AND z.uncertain=0 AND z.nvote>=20
AND p.modeDR7=1 AND p.nChild=0 AND p.petroMagErr_r>=0 AND p.petroMagErr_r<0.1
AND p.petroMag_r BETWEEN 10 AND 25
ORDER BY z.specobjid'''


def query_rows(sql):
    """Execute SQL against the public DR19 endpoint and return result-row dicts.

    The response contains both the data table and a SQL echo table. Select
    only Table1 so query metadata cannot accidentally become training rows.
    Network/JSON failures propagate; malformed responses raise ValueError.
    """
    request = Request(ENDPOINT + '?' + urlencode({'cmd':sql, 'format':'json'}),
                      headers={'User-Agent':'galaxy-ml-pipeline/2.0'})
    with urlopen(request, timeout=120) as response:
        payload = json.load(response)
    if not isinstance(payload, list):
        raise ValueError(f'SkyServer did not return tables: {str(payload)[:500]}')
    for table in payload:
        if table.get('TableName') == 'Table1':
            return table['Rows']
    raise ValueError(f'SkyServer returned no result table: {str(payload)[:500]}')


def download_catalog(output, per_class=1200, page_size=500):
    """Collect per_class qualifying rows for each class and persist the sample.

    output is a new CSV path; page_size controls network query batch size.
    Return the assembled DataFrame and save sibling provenance JSON with
    source, units and executed SQL. IDs remain decimal text in the CSV.
    Records stay in memory until collection finishes; queries do not resume
    after a failed process. Existing output CSVs are never overwritten.
    """
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; reuse it or choose a new output path.')
    if per_class < 1 or not 1 <= page_size <= 1000:
        raise ValueError('per-class must be positive; page-size must be 1..1000.')
    records, queries = [], []
    # Collect classes separately so one common class cannot exhaust the row budget.
    for label in (0, 1):
        cursor, collected = 0, 0
        while collected < per_class:
            sql = build_query(label, cursor, min(page_size, per_class-collected))
            rows = query_rows(sql)
            queries.append(sql)
            if not rows:
                raise ValueError(f'Only {collected} qualifying objects found for class {label}; requested {per_class}.')
            # Python integers keep all digits; ordered cursors must advance strictly.
            ids = [int(row['specobjid']) for row in rows]
            if len(set(ids)) != len(ids) or ids != sorted(ids) or ids[0] <= cursor:
                raise ValueError('Join produced duplicate IDs or an invalid page. Inspect the query; no records were arbitrarily discarded.')
            records.extend(rows)
            collected += len(rows)
            cursor = ids[-1]
            print(f'Class {label}: {collected}/{per_class}', flush=True)
            if collected < per_class:
                time.sleep(.2)
    frame = pd.DataFrame(records)
    if frame.specobjid.duplicated().any():
        raise ValueError('Duplicate specobjid across downloaded results.')
    output.parent.mkdir(parents=True, exist_ok=True)
    # Publish the CSV only after both classes finish; no half-sample is a final input.
    temporary = output.with_suffix(output.suffix + '.part')
    frame.to_csv(temporary,index=False)
    temporary.replace(output)
    output.with_suffix('.provenance.json').write_text(json.dumps({
        'endpoint':ENDPOINT, 'photometry':'PhotoObjDR7: legacy DR7, NOT DR19 photometric measurements',
        'selection':'First qualifying specobjid values per class; NOT a random survey sample',
        'join':'ZooSpec.dr7objid=PhotoObjDR7.dr7objid AND ZooSpec.objid=PhotoObjDR7.dr8objid',
        'distance_unit':'arcmin', 'coordinates_unit':'degrees', 'radii_unit':'arcsec',
        'queries':queries, 'rows':len(frame),
    },indent=2)+'\n')
    print(f'Saved {len(frame)} rows to {output}')
    return frame


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', default='datasets/raw/sdss_catalog.csv')
    p.add_argument('--per-class', type=int, default=1200)
    p.add_argument('--page-size', type=int, default=500)
    a = p.parse_args()
    download_catalog(a.output,a.per_class,a.page_size)

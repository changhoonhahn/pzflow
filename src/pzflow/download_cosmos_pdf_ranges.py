"""Accelerate an existing public archive download with validated byte ranges."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import urllib.request
import shutil

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data/cosmos2020_specz'
URL='https://irsa.ipac.caltech.edu/data/COSMOS/tables/cosmos2020/PZ/COSMOS2020_CLASSIC_R1_v2.1_LEPHARE_PZ_p3.fits.gz'
SOURCE=DATA/'COSMOS2020_CLASSIC_R1_v2.1_LEPHARE_PZ_p3.fits.gz'
OUTPUT=DATA/'classic_lephare_complete.fits.gz'
TOTAL=1827616655
PREFIX=600000000
if OUTPUT.exists() and OUTPUT.stat().st_size==TOTAL:
    print(f'Already downloaded: {OUTPUT}',flush=True)
    raise SystemExit(0)
if not SOURCE.exists():SOURCE=SOURCE.with_suffix(SOURCE.suffix+'.partial')
assert SOURCE.stat().st_size>=PREFIX
with SOURCE.open('rb') as src, OUTPUT.with_suffix('.prefix').open('wb') as dst:
    remaining=PREFIX
    while remaining:
        b=src.read(min(8*1024*1024,remaining)); assert b
        dst.write(b);remaining-=len(b)

def get_part(k):
    a=PREFIX+(TOTAL-PREFIX)*k//8
    b=PREFIX+(TOTAL-PREFIX)*(k+1)//8-1
    path=DATA/f'classic_pdf_part_{k}'
    for attempt in range(3):
        try:
            req=urllib.request.Request(URL,headers={'Range':f'bytes={a}-{b}'})
            with urllib.request.urlopen(req,timeout=180) as r,path.open('wb') as f:
                assert r.status==206
                assert r.headers['Content-Range']==f'bytes {a}-{b}/{TOTAL}'
                shutil.copyfileobj(r,f,length=1024*1024)
            assert path.stat().st_size==b-a+1
            print(f'Part {k+1}/8 complete',flush=True)
            return path
        except Exception:
            if attempt==2:raise

with ThreadPoolExecutor(max_workers=8) as pool: parts=list(pool.map(get_part,range(8)))
with OUTPUT.open('wb') as dst:
    for p in [OUTPUT.with_suffix('.prefix')]+parts:
        with p.open('rb') as src:shutil.copyfileobj(src,dst,length=8*1024*1024)
assert OUTPUT.stat().st_size==TOTAL
print(f'Complete: {OUTPUT}',flush=True)

"""Package the current self-contained manuscript sources and rendered figures."""
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[2]

def main():
    paper=ROOT/'paper'
    files=sorted(paper.glob('*.tex'))+[paper/n for n in
        ['Makefile','references.bib','aastex63.cls','aasjournal.bst','main.bbl']]
    files+=sorted((paper/'figures').glob('*.pdf'))
    out=ROOT/'output/cosmos_paper_tex.zip'
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,'cosmos_paper_tex/'+str(p.relative_to(paper)))
        z.writestr('cosmos_paper_tex/README.txt',
            'Current manuscript sources with Makefile and figure PDFs.\n'
            'Run make to produce main.pdf; requires latexmk, pdfLaTeX, and BibTeX.\n'
            'Figures use the SpenderQ methods-paper plotting style.\n'
            'Author metadata remains unset. Figure notebooks and their data are\n'
            'included in the separate full research package.\n')
    with zipfile.ZipFile(out) as z:assert z.testzip() is None
    print(out)

if __name__=='__main__':main()

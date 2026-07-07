"""
Triage per la eval MedSAM2 support_bbox multi-muscolo.

Legge lo scores.csv per-scan scritto da scripts/eval/eval_medsam2.py (colonne:
class,label,scan,dice,iou) e stampa:
  1. tabella per-classe (n, mean/median/min/max Dice) ordinata dal peggiore;
  2. gli scan sotto soglia (default 0.40) raggruppati per classe -> candidati al debug;
  3. i singoli scan peggiori in assoluto.

Solo lettura CSV, nessun modello caricato. Uso:
    python3 scripts/eval/analyze_scores.py results/all_muscles/scores.csv [--thr 0.40]
"""
import argparse
import csv
from collections import defaultdict
from statistics import mean, median


def load(path: str) -> list[dict]:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['dice'] = float(r['dice'])
        r['iou'] = float(r['iou'])
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('scores_csv', type=str)
    ap.add_argument('--thr', type=float, default=0.40,
                    help='soglia Dice sotto cui uno scan e\' segnalato come fallito')
    args = ap.parse_args()

    rows = load(args.scores_csv)
    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r['class']].append(r)

    # 1. tabella per-classe, peggiore in cima
    stats = []
    for cls, rs in by_class.items():
        d = [r['dice'] for r in rs]
        stats.append((cls, len(d), mean(d), median(d), min(d), max(d)))
    stats.sort(key=lambda s: s[2])

    print(f'\n=== Per-classe (n={len(rows)} scan, {len(by_class)} classi) ===')
    print(f'{"class":<8}{"n":>4}{"mean":>9}{"median":>9}{"min":>9}{"max":>9}')
    for cls, n, mn, md, lo, hi in stats:
        print(f'{cls:<8}{n:>4}{mn:>9.4f}{md:>9.4f}{lo:>9.4f}{hi:>9.4f}')
    overall = [r['dice'] for r in rows]
    print(f'{"ALL":<8}{len(overall):>4}{mean(overall):>9.4f}'
          f'{median(overall):>9.4f}{min(overall):>9.4f}{max(overall):>9.4f}')

    # 2. scan falliti per classe (candidati debug)
    print(f'\n=== Scan sotto soglia (Dice < {args.thr}) ===')
    any_fail = False
    for cls, _n, _mn, _md, _lo, _hi in stats:
        fails = sorted((r for r in by_class[cls] if r['dice'] < args.thr),
                       key=lambda r: r['dice'])
        if fails:
            any_fail = True
            ids = ', '.join(f'{r["scan"]}({r["dice"]:.3f})' for r in fails)
            print(f'  {cls:<8} {len(fails):>2}/{len(by_class[cls]):<2}  {ids}')
    if not any_fail:
        print('  nessuno.')

    # 3. worst assoluti
    print('\n=== 10 scan peggiori (assoluti) ===')
    for r in sorted(rows, key=lambda r: r['dice'])[:10]:
        print(f'  {r["class"]:<8} {r["scan"]:<20} Dice={r["dice"]:.4f} IoU={r["iou"]:.4f}')


if __name__ == '__main__':
    main()

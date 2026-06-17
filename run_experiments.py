"""
Run all Setting-1 experiments.

Train on T1 or T2, then test on both T1 and T2 for each trained model.
Final table: rows = (model, train_domain, fold), cols = test T1 / test T2.

Results are written as individual JSON files under RESULTS_DIR so that
parallel runs never race on the same file.  Use --merge to collect all
JSON files into a single CSV.

Usage:
  python run_experiments.py                              # all experiments
  python run_experiments.py --model qnet                 # only qnet
  python run_experiments.py --train_domain T1            # only T1-trained
  python run_experiments.py --fold 0                     # only fold 0
  python run_experiments.py --skip_train                 # test only
  python run_experiments.py --dry_run                    # print without executing
  python run_experiments.py --merge                      # merge JSONs → CSV
"""

import argparse
import copy
import csv
import glob
import json
import os

import torch

from train import train_from_cfg
from test  import test_from_cfg

DATA_DIRS = {
    'T1': 'data/datasets/CHAOS/processed/T1',
    'T2': 'data/datasets/CHAOS/processed/T2',
}

BASE_CFG = {
    'data': {
        'n_folds':       4,
        'n_shot':        1,
        'batch_size':    1,
        'num_workers':   4,
        'min_size':      200,
        'exclude_label': None,
        'label_names':   ['BG', 'LIVER', 'RK', 'LK', 'SPLEEN'],
    },
    'train': {
        'lr':           0.001,
        'lr_gamma':     0.95,
        'align_weight': 1.0,
        'max_epochs':   30,
    },
    'test': {
        'supp_idx': 0,
        'n_part':   3,
    },
    'domain': {
        'domain_map':    None,
        'source_domain': None,
        'target_domain': None,
    },
}

BG_LOSS = {'alpnet': 0.05, 'qnet': 0.1}

RESULTS_DIR = 'results'
RESULTS_CSV = os.path.join(RESULTS_DIR, 'results_s1.csv')

EXPERIMENTS = [
    {'model': model, 'train_domain': td, 'fold': fold}
    for model  in ('alpnet', 'qnet')
    for td     in ('T1', 'T2')
    for fold   in range(4)
]

RESULTS_CSV = 'results_s1.csv'
CSV_FIELDS  = [
    'run', 'test_domain',
    'LIVER_dice', 'LIVER_iou',
    'RK_dice',    'RK_iou',
    'LK_dice',    'LK_iou',
    'SPLEEN_dice','SPLEEN_iou',
    'MEAN_dice',  'MEAN_iou',
]

def _run_name(model: str, train_domain: str, fold: int) -> str:
    return f'{model}_{train_domain}_fold{fold}_s1'


def _ckpt_path(model: str, train_domain: str, fold: int) -> str:
    return os.path.join(
        'lightning_logs', _run_name(model, train_domain, fold), 'checkpoints', 'last.ckpt'
    )


def _build_cfg(model: str, train_domain: str, fold: int) -> dict:
    cfg = copy.deepcopy(BASE_CFG)
    cfg['model']                    = {'name': model}
    cfg['data']['data_dir']         = DATA_DIRS[train_domain]
    cfg['data']['fold']             = fold
    cfg['train']['bg_loss_weight']  = BG_LOSS[model]
    return cfg


def _result_path(run: str, test_domain: str) -> str:
    return os.path.join(RESULTS_DIR, f'{run}_{test_domain}.json')


def _save_result(run: str, test_domain: str, results: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    payload = {'run': run, 'test_domain': test_domain, 'results': results}
    with open(_result_path(run, test_domain), 'w') as f:
        json.dump(payload, f, indent=2)


def _already_done(run: str, test_domain: str) -> bool:
    return os.path.isfile(_result_path(run, test_domain))


def _row_from_results(run: str, test_domain: str, results: dict) -> dict:
    row = {'run': run, 'test_domain': test_domain}
    for organ in ('LIVER', 'RK', 'LK', 'SPLEEN', 'MEAN'):
        m = results.get(organ, {})
        row[f'{organ}_dice'] = f"{m['dice']:.4f}" if m else ''
        row[f'{organ}_iou']  = f"{m['iou']:.4f}"  if m else ''
    return row


def merge_results():
    """Collect all per-run JSONs in RESULTS_DIR into a single CSV."""
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, '*.json'))):
        with open(path) as f:
            payload = json.load(f)
        rows.append(_row_from_results(payload['run'], payload['test_domain'], payload['results']))

    if not rows:
        print('No result JSONs found — nothing to merge.')
        return

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(RESULTS_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Merged {len(rows)} results → {RESULTS_CSV}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',        type=str, default=None, choices=['alpnet', 'qnet'])
    parser.add_argument('--train_domain', type=str, default=None, choices=['T1', 'T2'])
    parser.add_argument('--fold',         type=int, default=None)
    parser.add_argument('--max_epochs',   type=int, default=None,
                        help='Override max_epochs (default: 30 for qnet, 100 for alpnet)')
    parser.add_argument('--skip_train',   action='store_true')
    parser.add_argument('--dry_run',      action='store_true')
    parser.add_argument('--merge',        action='store_true',
                        help='Merge all per-run JSONs in results/ into results_s1.csv')
    parser.add_argument('--device',       type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    if args.merge:
        merge_results()
        return

    exps = [
        e for e in EXPERIMENTS
        if (args.model        is None or e['model']        == args.model)
        and (args.train_domain is None or e['train_domain'] == args.train_domain)
        and (args.fold         is None or e['fold']         == args.fold)
        and os.path.isdir(DATA_DIRS[e['train_domain']])   # skip if data not available
    ]

    if not exps:
        print('No experiments to run (check --model / --train_domain / --fold and data dirs).')
        return

    for exp in exps:
        model        = exp['model']
        train_domain = exp['train_domain']
        fold         = exp['fold']
        run          = _run_name(model, train_domain, fold)
        ckpt         = _ckpt_path(model, train_domain, fold)
        cfg          = _build_cfg(model, train_domain, fold)
        if args.max_epochs is not None:
            cfg['train']['max_epochs'] = args.max_epochs

        print(f'\n{"="*52}')
        print(f'  {run}')
        print(f'{"="*52}')

        # --- TRAIN ---
        if not args.skip_train:
            if os.path.isfile(ckpt):
                print(f'[SKIP train] checkpoint exists: {ckpt}')
            else:
                print(f'[TRAIN] {run}')
                if args.dry_run:
                    print(f'  >> train_from_cfg(model={model}, train_domain={train_domain}, fold={fold})')
                else:
                    ckpt = train_from_cfg(cfg)
                    print(f'[DONE train] ckpt → {ckpt}')

        if not args.dry_run and not os.path.isfile(ckpt):
            print(f'[ERROR] checkpoint not found: {ckpt} — skipping tests')
            continue

        #TEST on each available domain
        for test_domain, test_dir in DATA_DIRS.items():
            if not os.path.isdir(test_dir):
                print(f'[SKIP test {test_domain}] {test_dir} not found')
                continue

            if _already_done(run, test_domain):
                print(f'[SKIP test {test_domain}] result exists: {_result_path(run, test_domain)}')
                continue

            label = 'same-domain' if test_domain == train_domain else 'cross-domain'
            print(f'[TEST {test_domain} ({label})] {run}')

            if args.dry_run:
                tgt = None if test_domain == train_domain else test_dir
                print(f'  >> test_from_cfg(cfg, {ckpt}, target_data_dir={tgt})')
                continue

            target_dir = None if test_domain == train_domain else test_dir
            results    = test_from_cfg(cfg, ckpt,
                                       target_data_dir=target_dir,
                                       device_str=args.device)
            _save_result(run, test_domain, results)
            mean_d = results.get('MEAN', {}).get('dice', float('nan'))
            print(f'[DONE test {test_domain}] MEAN Dice={mean_d:.4f} → {_result_path(run, test_domain)}')

    print(f'\n{"="*52}')
    print(f'  ALL DONE  —  results in {RESULTS_CSV}')
    print(f'{"="*52}')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Plot training/validation curves from Lightning's metrics.csv.

Safe to run mid-training -- reads whatever has been flushed so far.
Handles the sparse CSV layout Lightning produces (each metric only
populated on the rows where it was logged, NaN elsewhere).

Usage:
    python plot_curves.py --run-dir outputs/swahili_random_init
    python plot_curves.py --run-dir <dir> --out curves.png --watch 120
"""

import argparse
import glob
import os
import time

import matplotlib
matplotlib.use('Agg')          # headless -- required on cluster nodes
import matplotlib.pyplot as plt
import pandas as pd


# (column, label, which subplot)  -- subplot key groups metrics onto shared axes
METRICS = [
    ('trainer/loss', 'train loss (per step)', 'loss'),
    ('train/nll',    'train NLL (per epoch)', 'loss'),
    ('val/nll',      'val NLL',               'loss'),
    ('train/ppl',    'train perplexity',      'ppl'),
    ('val/ppl',      'val perplexity',        'ppl'),
    ('train/bpd',    'train bits/dim',        'bpd'),
    ('val/bpd',      'val bits/dim',          'bpd'),
]


def find_metrics_csv(run_dir):
    """Locate metrics.csv, preferring the highest version_N directory."""
    pattern = os.path.join(run_dir, 'csv_logs', 'version_*', 'metrics.csv')
    matches = sorted(glob.glob(pattern))
    if not matches:
        # fall back to a recursive search -- layout may differ if the run
        # was launched with a different save_dir
        matches = sorted(glob.glob(
            os.path.join(run_dir, '**', 'metrics.csv'), recursive=True))
    if not matches:
        raise FileNotFoundError(
            f'No metrics.csv found under {run_dir}. Has training started, '
            f'and has it reached the first flush (default: 100 steps)?')
    return matches[-1]


def load_series(csv_path):
    """Return {column: (steps, values)} for every metric present."""
    df = pd.read_csv(csv_path)
    if 'step' not in df.columns:
        raise ValueError(f'{csv_path} has no "step" column; got {list(df.columns)}')

    series = {}
    for col, label, group in METRICS:
        if col not in df.columns:
            continue
        # Each metric is sparse: drop the rows where it wasn't logged.
        sub = df[['step', col]].dropna()
        if sub.empty:
            continue
        series[col] = (sub['step'].to_numpy(), sub[col].to_numpy(), label, group)
    return series, df


def plot(series, out_path, title):
    groups = []
    for _, _, _, group in series.values():
        if group not in groups:
            groups.append(group)
    if not groups:
        raise ValueError('No plottable metrics found in the CSV.')

    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 4 * len(groups)),
                             squeeze=False)
    axes = axes[:, 0]

    for ax, group in zip(axes, groups):
        for col, (steps, vals, label, g) in series.items():
            if g != group:
                continue
            # per-step train loss is noisy -- thin the marker, keep the line
            style = dict(linewidth=1.0, alpha=0.65) if 'per step' in label \
                else dict(linewidth=1.8, marker='o', markersize=3)
            ax.plot(steps, vals, label=label, **style)
        ax.set_xlabel('step')
        ax.set_ylabel(group)
        ax.grid(True, alpha=0.3)
        ax.legend()
    axes[0].set_title(title)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def summarise(series):
    """Print the numbers you actually make decisions from."""
    print(f'{"metric":<24}{"last step":>10}{"last value":>14}'
          f'{"best value":>14}{"best step":>11}')
    print('-' * 73)
    for col, (steps, vals, label, _) in series.items():
        best_i = vals.argmin()
        print(f'{col:<24}{int(steps[-1]):>10}{vals[-1]:>14.4f}'
              f'{vals[best_i]:>14.4f}{int(steps[best_i]):>11}')

    # The decision signal: is val loss still improving, or turning up?
    if 'val/nll' in series:
        steps, vals, _, _ = series['val/nll']
        if len(vals) >= 3:
            best_i = int(vals.argmin())
            print()
            if best_i == len(vals) - 1:
                print('val NLL is still at its minimum on the last point -- '
                      'likely undertrained, consider more steps.')
            else:
                since = int(steps[-1] - steps[best_i])
                print(f'val NLL bottomed at step {int(steps[best_i])} '
                      f'({since} steps ago) and has not improved since -- '
                      f'check whether it has plateaued or is rising.')
        else:
            print('\nFewer than 3 val points so far -- too early to read a '
                  'trend. Lower trainer.val_check_interval for more.')


def run_once(args):
    csv_path = find_metrics_csv(args.run_dir)
    series, df = load_series(csv_path)
    title = args.title or os.path.basename(os.path.normpath(args.run_dir))
    out_path = args.out or os.path.join(args.run_dir, 'curves.png')
    plot(series, out_path, title)
    print(f'\nread : {csv_path}  ({len(df)} rows)')
    print(f'wrote: {out_path}\n')
    summarise(series)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run-dir', required=True,
                    help='Hydra run dir, e.g. outputs/swahili_random_init')
    ap.add_argument('--out', default=None,
                    help='Output PNG (default: <run-dir>/curves.png)')
    ap.add_argument('--title', default=None)
    ap.add_argument('--watch', type=int, default=0, metavar='SECONDS',
                    help='Re-plot every N seconds until interrupted')
    args = ap.parse_args()

    if args.watch:
        print(f'Watching {args.run_dir}, re-plotting every {args.watch}s. '
              f'Ctrl-C to stop.')
        while True:
            try:
                run_once(args)
            except (FileNotFoundError, ValueError, pd.errors.EmptyDataError,
                    pd.errors.ParserError) as e:
                # ParserError happens if we read mid-flush and catch a
                # partially-written line -- harmless, just retry next tick.
                print(f'[waiting] {e}')
            time.sleep(args.watch)
    else:
        run_once(args)


if __name__ == '__main__':
    main()
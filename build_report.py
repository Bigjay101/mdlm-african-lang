#!/usr/bin/env python3
"""
Consolidate every Swahili 2x2 result into one PDF: graphs and tables only.

Reads what already exists -- no GPU, safe on the login node:
  results/results_table.csv, tokenizer_table.csv, samples_table.csv, fig_*.png
                                         (from make_results.py -- run that first)
  outputs/<run>/csv_logs/version_*/metrics.csv   (validation curves, all versions stitched)
  analysis/continuation/prompts.jsonl + <cell>.jsonl   (from bleu_eval.py)

Writes:
  results/swahili_results.pdf
  results/continuation_ci.csv     (chrF/BLEU with 95% bootstrap intervals)
  results/continuation_paired.csv (paired bootstrap differences between systems)

Run from the repo root:
    python make_results.py      # refresh results/ first
    python build_report.py
Needs: matplotlib, pandas, sacrebleu (all already in the mdlm env).
"""

import argparse
import datetime
import glob
import json
import pathlib
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

ROOT = pathlib.Path(__file__).resolve().parent
RES = ROOT / 'results'
CONT = ROOT / 'analysis' / 'continuation'

CELLS = {  # display name -> (model, tokenizer, run dir, continuation file stem)
    'MDLM + Unigram': ('MDLM', 'Unigram', 'outputs/swahili_scratch',  'mdlm_unigram'),
    'AR + Unigram':   ('AR',   'Unigram', 'outputs/swahili_ar',       'ar_unigram'),
    'MDLM + BPE':     ('MDLM', 'BPE',     'outputs/swahili_mdlm_bpe', 'mdlm_bpe'),
    'AR + BPE':       ('AR',   'BPE',     'outputs/swahili_ar_bpe',   'ar_bpe'),
}
COLOUR = {'MDLM': '#2a78d6', 'AR': '#eb6834'}
DASH = {'Unigram': '-', 'BPE': '--'}
INK, INK2, GRID, SURFACE, HEAD = '#0b0b0b', '#52514e', '#e4e3df', '#ffffff', '#f1f0ec'
A4 = (11.69, 8.27)  # landscape
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.titlesize': 12, 'axes.titleweight': 'bold', 'legend.frameon': False,
})


# ------------------------------------------------------------------ helpers
def load_metrics(run_dir):
    """Stitch csv_logs/version_* (resumed runs), cutting each where the next starts."""
    def ver(p):
        m = re.search(r'version_(\d+)', p)
        return int(m.group(1)) if m else -1
    files = sorted(glob.glob(str(ROOT / run_dir / 'csv_logs' / 'version_*' / 'metrics.csv')), key=ver)
    frames = [f for f in (pd.read_csv(p) for p in files) if not f.empty]
    if not frames:
        return None
    kept = [f[f['step'] < frames[i + 1]['step'].min()] if i + 1 < len(frames) else f
            for i, f in enumerate(frames)]
    return pd.concat(kept, ignore_index=True)


def page(pdf, title, subtitle=None):
    fig = plt.figure(figsize=A4)
    fig.text(0.05, 0.94, title, fontsize=17, fontweight='bold', va='top')
    if subtitle:
        fig.text(0.05, 0.895, subtitle, fontsize=10, color=INK2, va='top', wrap=True)
    return fig


def finish(pdf, fig, note=None):
    if note:
        fig.text(0.05, 0.04, note, fontsize=8.5, color=INK2, va='bottom', wrap=True)
    pdf.savefig(fig)
    plt.close(fig)


def draw_table(fig, rect, df, col_widths=None, highlight_rows=(), fontsize=10):
    ax = fig.add_axes(rect)
    ax.axis('off')
    if col_widths is None:
        n = len(df.columns)
        first = 0.3 if n > 2 else 0.5
        col_widths = [first] + [(1 - first) / (n - 1)] * (n - 1)
    t = ax.table(cellText=df.values, colLabels=list(df.columns), loc='upper center',
                 cellLoc='center', colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.8)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_height(cell.get_height() * 1.6)
            cell.set_facecolor(HEAD)
            cell.set_text_props(fontweight='bold')
        elif (r - 1) in highlight_rows:
            cell.set_facecolor('#f7f7f4')
            cell.set_text_props(color=INK2, style='italic')
        if c == 0:
            cell.set_text_props(ha='left')
            cell._loc = 'left'
    return ax


def image_page(pdf, png, title, subtitle=None, note=None):
    if not png.exists():
        print(f'  skip: {png.name} not found')
        return
    fig = page(pdf, title, subtitle)
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.76])
    ax.imshow(plt.imread(png))
    ax.axis('off')
    finish(pdf, fig, note)


def fmt(v, d=3):
    return '' if v is None or (isinstance(v, float) and np.isnan(v)) else f'{v:.{d}f}'


# --------------------------------------------------------- continuation + CI
def continuation_stats(n_boot, seed):
    from sacrebleu.metrics import BLEU, CHRF
    prompts_f = CONT / 'prompts.jsonl'
    if not prompts_f.exists():
        return None, None
    prompts = {json.loads(l)['id']: json.loads(l) for l in prompts_f.read_text(encoding='utf-8').splitlines() if l.strip()}
    ids = sorted(prompts)
    refs = [prompts[i]['reference'] for i in ids]
    floor = [prompts[ids[(k + 1) % len(ids)]]['reference'] for k in range(len(ids))]
    systems = {'Random real continuation (floor)': floor}
    for cell, (_, _, _, stem) in CELLS.items():
        f = CONT / f'{stem}.jsonl'
        if f.exists():
            recs = {r['id']: r['hypothesis'] for r in
                    (json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip())}
            if set(recs) >= set(ids):
                systems[cell] = [recs[i] for i in ids]
    bleu, chrf = BLEU(), CHRF()
    rng = np.random.default_rng(seed)
    boots = [rng.integers(0, len(ids), len(ids)) for _ in range(n_boot)]

    def corpus(m, hyps, idx=None):
        if idx is None:
            return m.corpus_score(hyps, [refs]).score
        return m.corpus_score([hyps[i] for i in idx], [[refs[i] for i in idx]]).score

    rows, samples = [], {}
    for name, hyps in systems.items():
        b_s = np.array([corpus(bleu, hyps, ix) for ix in boots])
        c_s = np.array([corpus(chrf, hyps, ix) for ix in boots])
        samples[name] = c_s
        rows.append({'system': name, 'n': len(hyps),
                     'BLEU': corpus(bleu, hyps), 'BLEU_lo': np.percentile(b_s, 2.5), 'BLEU_hi': np.percentile(b_s, 97.5),
                     'chrF': corpus(chrf, hyps), 'chrF_lo': np.percentile(c_s, 2.5), 'chrF_hi': np.percentile(c_s, 97.5)})
    df = pd.DataFrame(rows)

    # Paired differences (same bootstrap resamples for both systems).
    pairs = [('MDLM + Unigram', 'AR + Unigram'), ('MDLM + BPE', 'AR + BPE'),
             ('MDLM + Unigram', 'MDLM + BPE'), ('AR + Unigram', 'AR + BPE')]
    pairs += [(c, 'Random real continuation (floor)') for c in CELLS]
    prow = []
    for a, b in pairs:
        if a in samples and b in samples:
            d = samples[a] - samples[b]
            prow.append({'comparison': f'{a}  vs  {b}',
                         'chrF_diff': float(df.set_index('system').loc[a, 'chrF'] - df.set_index('system').loc[b, 'chrF']),
                         'lo': np.percentile(d, 2.5), 'hi': np.percentile(d, 97.5),
                         'p_a_better': float((d > 0).mean())})
    return df, pd.DataFrame(prow)


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', type=pathlib.Path, default=RES / 'swahili_results.pdf')
    ap.add_argument('--n-boot', type=int, default=1000, help='bootstrap resamples for CIs')
    ap.add_argument('--author', default='Jordan Tshibumbu')
    args = ap.parse_args()

    res_f = RES / 'results_table.csv'
    if not res_f.exists():
        raise SystemExit('results/results_table.csv not found -- run `python make_results.py` first.')
    res = pd.read_csv(res_f).set_index('cell')

    # Stitched curves in BPC. k converts nats/token -> bits/char per tokenizer,
    # recovered from make_results' own numbers so both scripts agree exactly.
    curves, final = {}, {}
    for cell, (model, tok, run, _) in CELLS.items():
        if cell not in res.index:
            continue
        k = res.loc[cell, 'val_bpc'] / res.loc[cell, 'val_nll_nats_per_token']
        df = load_metrics(run)
        if df is None:
            continue
        val = df[['step', 'val/nll']].dropna()
        tr = df[['step', 'train/nll']].dropna() if 'train/nll' in df else pd.DataFrame()
        curves[cell] = (val['step'].to_numpy(), val['val/nll'].to_numpy() * k,
                        tr['step'].to_numpy() if len(tr) else None,
                        tr['train/nll'].to_numpy() * k if len(tr) else None)
        final[cell] = (int(val['step'].iloc[-1]), float(val['val/nll'].iloc[-1] * k),
                       float(tr['train/nll'].iloc[-1] * k) if len(tr) else np.nan)

    print('Bootstrapping continuation scores ...')
    cont, pairs = continuation_stats(args.n_boot, seed=0)
    if cont is not None:
        cont.to_csv(RES / 'continuation_ci.csv', index=False)
        if pairs is not None:
            pairs.to_csv(RES / 'continuation_paired.csv', index=False)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(args.out) as pdf:
        # ---- 1. title + headline table -------------------------------------
        fig = page(pdf, 'Masked diffusion vs autoregressive language models for Swahili',
                   f'{args.author}  ·  Results summary  ·  {datetime.date.today():%d %B %Y}\n'
                   'Design: {MDLM, AR} × {SentencePiece Unigram 16k, SentencePiece BPE 16k}, '
                   'parameter-matched (768 hidden, 12 blocks, 12 heads), context 1024, global batch 512, '
                   'trained from random initialisation on Maneno Yetu.')
        rows = []
        for cell in CELLS:
            if cell not in res.index:
                continue
            r = res.loc[cell]
            st, fv, ft = final.get(cell, (np.nan, np.nan, np.nan))
            chrf = cont.set_index('system').loc[cell] if cont is not None and cell in set(cont['system']) else None
            rows.append([cell, fmt(r['best_val_bpc']), f"{int(r['best_val_step']):,}",
                         fmt(fv), f'{st:,}' if st == st else '', fmt(fv - r['best_val_bpc']),
                         fmt(fv - ft if ft == ft else np.nan),
                         '' if chrf is None else f"{chrf['chrF']:.2f}"])
        head = pd.DataFrame(rows, columns=['Model', 'Best\nval BPC', 'Best\nat step', 'Final\nval BPC',
                                           'Final\nstep', 'Rise\nsince best', 'Final\nval−train gap',
                                           'chrF\n(test)'])
        draw_table(fig, [0.05, 0.45, 0.9, 0.38], head, fontsize=11,
                   col_widths=[0.2] + [0.8 / 7] * 7)
        if cont is not None:
            fl = cont.set_index('system').loc['Random real continuation (floor)', 'chrF']
            fig.text(0.05, 0.47, f'chrF of an unrelated real continuation (floor): {fl:.2f}',
                     fontsize=9.5, color=INK2)
        finish(pdf, fig, 'BPC = bits per character on the validation split; comparable across '
               'tokenizers (per-token loss is not). Lower is better. MDLM BPC is the diffusion '
               'NELBO, an upper bound on its true NLL; AR BPC is exact. "Rise since best" = how much '
               'validation BPC got worse after the best checkpoint; "Final val−train gap" is how far '
               'validation loss sits above training loss at the end. Both measure overfitting. chrF is on prompted '
               'continuation of 100 test-split passages (higher is better).')

        # ---- 2. validation curves -----------------------------------------
        if curves:
            fig = page(pdf, 'Validation loss over training',
                       'All four models on one axis, in bits per character. Dots mark each '
                       "model's best checkpoint.")
            ax = fig.add_axes([0.08, 0.12, 0.72, 0.72])
            for cell, (s, v, _, _) in curves.items():
                model, tok = CELLS[cell][:2]
                ax.plot(s, v, DASH[tok], color=COLOUR[model], lw=2, label=cell)
                i = int(np.argmin(v))
                ax.plot(s[i], v[i], 'o', color=COLOUR[model], ms=7, mec='white', mew=1.5)
            ymin = min(v.min() for _, v, _, _ in curves.values())
            ax.set_ylim(ymin - 0.05, ymin + 0.9)
            ax.set(xlabel='optimizer step', ylabel='validation bits per character (lower is better)')
            ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1))
            finish(pdf, fig, 'Y-axis clipped to the useful range; the AR curves continue upward '
                   'beyond it. Colour = model, line style = tokenizer (solid Unigram, dashed BPE).')

            # ---- 3. train vs val per cell ------------------------------------
            fig = page(pdf, 'Train vs validation, per model',
                       'The gap between the curves is overfitting. Same y-axis for all four panels.')
            axes = fig.subplots(2, 2, sharex=True, sharey=True,
                                gridspec_kw=dict(left=0.08, right=0.97, top=0.84, bottom=0.12,
                                                 hspace=0.25, wspace=0.08))
            for ax, cell in zip(axes.flat, CELLS):
                ax.set_title(cell, fontsize=11)
                if cell not in curves:
                    ax.text(0.5, 0.5, 'no log', ha='center', transform=ax.transAxes)
                    continue
                s, v, ts, tv = curves[cell]
                if ts is not None:
                    ax.plot(ts, tv, color='#2a78d6', lw=1.8, label='train')
                ax.plot(s, v, color='#eb6834', lw=1.8, label='validation')
            axes[0, 0].legend()
            for ax in axes[1]: ax.set_xlabel('optimizer step')
            for ax in axes[:, 0]: ax.set_ylabel('bits per character')
            finish(pdf, fig)

        # ---- 4. best-checkpoint bars --------------------------------------
        fig = page(pdf, 'Best validation BPC per model',
                   'Each model at its own best checkpoint (lowest validation loss).')
        ax = fig.add_axes([0.08, 0.14, 0.42, 0.68])
        xs = {'Unigram': 0, 'BPE': 1}
        for cell in CELLS:
            if cell not in res.index:
                continue
            model, tok = CELLS[cell][:2]
            x = xs[tok] + (-0.19 if model == 'MDLM' else 0.19)
            y = res.loc[cell, 'best_val_bpc']
            ax.bar(x, y, width=0.36, color=COLOUR[model], label=model if tok == 'Unigram' else None)
            ax.text(x, y, f'{y:.3f}\nstep {int(res.loc[cell, "best_val_step"]):,}',
                    ha='center', va='bottom', fontsize=9)
        lo = res['best_val_bpc'].min()
        ax.set_ylim(lo * 0.9, res['best_val_bpc'].max() * 1.04)
        ax.set_xticks(list(xs.values()), list(xs.keys()))
        ax.grid(axis='x', visible=False)
        ax.set_ylabel('best validation bits per character')
        ax.legend()
        tbl = res.reset_index()[['cell', 'best_val_bpc', 'best_val_step', 'step_used', 'val_bpc', 'last_step']]
        tbl = pd.DataFrame({'Model': tbl['cell'], 'Best\nBPC': tbl['best_val_bpc'].map(fmt),
                            'Best\nstep': tbl['best_val_step'].map(lambda v: f'{int(v):,}'),
                            f'BPC at\nstep {int(res["step_used"].min()):,}': tbl['val_bpc'].map(fmt),
                            'Last\nstep': tbl['last_step'].map(lambda v: f'{int(v):,}')})
        draw_table(fig, [0.55, 0.5, 0.42, 0.32], tbl, fontsize=10,
                   col_widths=[0.32, 0.17, 0.17, 0.17, 0.17])
        finish(pdf, fig, 'Y-axis does not start at zero, to make differences visible; they are '
               'small. The matched-step column compares all four at the same training step.')

        # ---- 5. continuation -----------------------------------------------
        if cont is not None:
            fig = page(pdf, 'Prompted continuation on the test split',
                       '100 unseen test passages: 150-word prompt, model writes the next ~50 words, '
                       'scored against the true continuation (sacrebleu). Error bars: 95% bootstrap '
                       f'intervals ({args.n_boot} resamples of the passages).')
            ax = fig.add_axes([0.2, 0.14, 0.32, 0.66])
            c = cont.copy()
            col = [COLOUR[CELLS[s][0]] if s in CELLS else '#9c9a92' for s in c['system']]
            y = np.arange(len(c))[::-1]
            ax.barh(y, c['chrF'], color=col, height=0.55,
                    xerr=[c['chrF'] - c['chrF_lo'], c['chrF_hi'] - c['chrF']],
                    error_kw=dict(ecolor=INK2, capsize=3, lw=1))
            ax.set_yticks(y, [s.replace('Random real continuation (floor)', 'Unrelated real text\n(floor)')
                              for s in c['system']])
            fl = c.loc[c['system'].str.contains('floor'), 'chrF'].iloc[0]
            ax.axvline(fl, color=INK, ls=':', lw=1.2)
            ax.set_xlim(fl - 6, c['chrF_hi'].max() + 2)
            ax.grid(axis='y', visible=False)
            ax.set_xlabel('chrF (higher is better)')
            t = pd.DataFrame({'System': c['system'].str.replace('Random real continuation', 'Unrelated real text'),
                              'BLEU [95% CI]': [f'{a:.2f} [{b:.2f}, {d:.2f}]' for a, b, d in zip(c['BLEU'], c['BLEU_lo'], c['BLEU_hi'])],
                              'chrF [95% CI]': [f'{a:.2f} [{b:.2f}, {d:.2f}]' for a, b, d in zip(c['chrF'], c['chrF_lo'], c['chrF_hi'])]})
            draw_table(fig, [0.56, 0.5, 0.42, 0.32], t, col_widths=[0.42, 0.29, 0.29],
                       highlight_rows=[0], fontsize=8.5)
            finish(pdf, fig, 'chrF scores character sequences, so partly correct Swahili words earn '
                   'credit; BLEU needs exact whole words. The floor pairs each passage with an unrelated '
                   'real continuation: it is the score of fluent, in-domain Swahili that ignores the prompt.')

            if pairs is not None and len(pairs):
                fig = page(pdf, 'Are the differences real? Paired bootstrap on chrF',
                           'Difference in chrF between two systems on the same resampled passages. '
                           'If the 95% interval excludes 0, the difference is unlikely to be noise.')
                p = pairs.copy()
                ax = fig.add_axes([0.32, 0.14, 0.62, 0.66])
                y = np.arange(len(p))[::-1]
                sig = (p['lo'] > 0) | (p['hi'] < 0)
                ax.errorbar(p['chrF_diff'], y, xerr=[p['chrF_diff'] - p['lo'], p['hi'] - p['chrF_diff']],
                            fmt='none', ecolor=INK2, capsize=3, lw=1.2)
                ax.scatter(p['chrF_diff'], y, c=['#2a78d6' if s else '#9c9a92' for s in sig], s=45, zorder=3)
                ax.axvline(0, color=INK, lw=1)
                ax.set_yticks(y, [f"{s}\n{'interval excludes 0' if g else 'interval includes 0'}  ·  "
                                  f"P(first better) = {q:.2f}" for s, g, q in
                                  zip(p['comparison'], sig, p['p_a_better'])], fontsize=8.5)
                ax.grid(axis='y', visible=False)
                ax.set_xlabel('chrF difference (first minus second)')
                finish(pdf, fig, 'Blue = 95% interval excludes zero; grey = includes zero '
                       '(not distinguishable with 100 passages).')

        # ---- 6. tokenizers ------------------------------------------------
        tok_f = RES / 'tokenizer_table.csv'
        if tok_f.exists() or (RES / 'fig_fertility.png').exists():
            fig = page(pdf, 'Tokenizers', 'Fertility (tokens per word) on the validation split. '
                       "GPT-2's English tokenizer is shown for reference.")
            png = RES / 'fig_fertility.png'
            if png.exists():
                ax = fig.add_axes([0.05, 0.42, 0.55, 0.45]); ax.imshow(plt.imread(png)); ax.axis('off')
            if tok_f.exists():
                td = pd.read_csv(tok_f)
                keep = [c for c in ['tokenizer', 'vocab', 'val_tokens', 'tokens_per_word', 'chars_per_token']
                        if c in td.columns]
                td = td[keep].copy()
                for c in td.columns[1:]:
                    td[c] = td[c].map(lambda v: f'{v:,.0f}' if isinstance(v, (int, np.integer)) or
                                      (isinstance(v, float) and v > 100) else
                                      (f'{v:.3f}' if isinstance(v, float) else v))
                draw_table(fig, [0.05, 0.1, 0.9, 0.28], td, fontsize=10)
            finish(pdf, fig)

        # ---- 7. sample statistics ------------------------------------------
        image_page(pdf, RES / 'fig_artifacts.png', 'Unprompted samples vs real text',
                   'Single-letter words and unseen words per model, against real validation text '
                   '(dotted line).')
        samp_f = RES / 'samples_table.csv'
        if samp_f.exists():
            sd = pd.read_csv(samp_f)
            keep = [c for c in ['cell', 'n_samples', 'words', 'single_letter_per_1k_words',
                                'oov_word_rate_%', 'distinct_2_per_500_words', 'token_entropy'] if c in sd.columns]
            sd = sd[keep].fillna('').rename(columns={
                'cell': 'Model', 'n_samples': 'Samples', 'words': 'Words',
                'single_letter_per_1k_words': 'Single-letter\nwords per 1k', 'oov_word_rate_%': 'Unseen\nwords %',
                'distinct_2_per_500_words': 'Distinct\nbigrams', 'token_entropy': 'Token\nentropy'})
            for c in sd.columns[1:]:
                sd[c] = sd[c].map(lambda v: (f'{v:,.0f}' if c in ('Samples', 'Words') else f'{v:.2f}')
                                  if isinstance(v, float) else v)
            fig = page(pdf, 'Unprompted sample statistics')
            draw_table(fig, [0.05, 0.4, 0.9, 0.45], sd, fontsize=9.5,
                       highlight_rows=[i for i, c in enumerate(sd['Model']) if 'Real' in str(c)])
            finish(pdf, fig)

    print(f'Wrote {args.out}')
    if cont is not None:
        print(f'Wrote {RES / "continuation_ci.csv"} and continuation_paired.csv')


if __name__ == '__main__':
    main()
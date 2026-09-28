#!/usr/bin/env python3
"""
Build the results package for the Swahili 2x2 experiment
({MDLM, AR} x {SentencePiece Unigram 16k, SentencePiece BPE 16k}).

Reads only what already exists on disk -- no GPU needed, safe on the login node:
  * outputs/<run>/csv_logs/version_*/metrics.csv   (training + validation logs)
  * configs/data/maneno-yetu/...                   (split manifest, corpus, tokenizers)
  * analysis/samples/<cell>.jsonl                  (optional: from sample_swahili.py)

Writes everything to results/:
  results_table.csv       one row per cell, every number in the summary
  tokenizer_table.csv     fertility per tokenizer on the val split
  samples_table.csv       sample statistics per cell vs real val text (if samples exist)
  summary.md              the tables, formatted, with the caveats that go with them
  fig_*.png               figures

Run from the repo root:
    python make_results.py
    python make_results.py --run "AR + BPE=outputs/my_other_dir"   # if a folder differs
    python make_results.py --step 7500                            # matched comparison step
"""

import argparse
import collections
import glob
import json
import math
import pathlib
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import transformers

# --------------------------------------------------------------------- config
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'configs' / 'data' / 'maneno-yetu'

CELLS = {  # cell name -> (model, tokenizer, default run dir, default samples file)
    'MDLM + Unigram': ('MDLM', 'Unigram', 'outputs/swahili_scratch',  'analysis/samples/mdlm_unigram.jsonl'),
    'AR + Unigram':   ('AR',   'Unigram', 'outputs/swahili_ar',       'analysis/samples/ar_unigram.jsonl'),
    'MDLM + BPE':     ('MDLM', 'BPE',     'outputs/swahili_mdlm_bpe', 'analysis/samples/mdlm_bpe.jsonl'),
    'AR + BPE':       ('AR',   'BPE',     'outputs/swahili_ar_bpe',   'analysis/samples/ar_bpe.jsonl'),
}
TOKENIZERS = {'Unigram': DATA / 'tokenizer', 'BPE': DATA / 'tokenizer-bpe'}
BLOCK = 1024           # model.length used in every run
SPECIALS = re.compile(r'<s>|</s>|<pad>|<unk>|\[MASK\]')
WORD = re.compile(r'[^\W\d_]+', re.UNICODE)

# Chart styling (dataviz reference palette): colour = model, line style = tokenizer.
COLOUR = {'MDLM': '#2a78d6', 'AR': '#eb6834'}
DASH = {'Unigram': '-', 'BPE': '--'}
INK, INK2, GRID, SURFACE = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE, 'savefig.facecolor': SURFACE,
    'axes.edgecolor': GRID, 'axes.labelcolor': INK2, 'xtick.color': INK2, 'ytick.color': INK2,
    'text.color': INK, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8,
    'axes.spines.top': False, 'axes.spines.right': False, 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'legend.frameon': False,
})


# ------------------------------------------------------------------- helpers
def load_split(name):
    manifest = json.loads((DATA / 'splits' / 'split_manifest.json').read_text())
    return [(DATA / 'data-raw' / 'cleaned' / f).read_text(encoding='utf-8').strip()
            for f in manifest['splits'][name]]


def load_metrics(run_dir):
    files = sorted(glob.glob(str(run_dir / 'csv_logs' / 'version_*' / 'metrics.csv')))
    if not files:
        files = sorted(glob.glob(str(run_dir / '**' / 'metrics.csv'), recursive=True))
    if not files:
        return None, []
    # Requeued runs start a new version_N; stitch them and keep the latest value per step.
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.sort_values('step'), files


def series(df, col):
    if col not in df.columns:
        return pd.DataFrame(columns=['step', col])
    s = df[['step', col]].dropna()
    return s.groupby('step', as_index=False).last()


def value_at(s, col, step):
    """Last logged value at or before `step`; returns (value, actual_step)."""
    s = s[s['step'] <= step]
    if s.empty:
        return float('nan'), None
    row = s.iloc[-1]
    return float(row[col]), int(row['step'])


def token_entropy(ids):
    counts = collections.Counter(ids)
    n = sum(counts.values())
    return -sum(c / n * math.log(c / n) for c in counts.values()) if n else float('nan')


def text_stats(texts, train_vocab):
    words = [w for t in texts for w in WORD.findall(t)]
    n = max(len(words), 1)
    lower = [w.lower() for w in words]
    single = sum(1 for w in words if len(w) == 1)
    oov = sum(1 for w in lower if w not in train_vocab)
    # distinct-2 depends on text length, so average it over equal 500-word chunks
    chunks = [lower[i:i + 500] for i in range(0, len(lower) - 499, 500)] or [lower]
    d2 = [len(set(zip(c, c[1:]))) / max(len(c) - 1, 1) for c in chunks]
    return {
        'words': len(words),
        'single_letter_per_1k_words': 1000 * single / n,
        'oov_word_rate_%': 100 * oov / n,
        'distinct_2_per_500_words': sum(d2) / len(d2),
    }


def save(fig, name, out):
    fig.tight_layout()
    fig.savefig(out / name, dpi=200)
    plt.close(fig)
    print(f'  wrote {out / name}')


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--step', type=int, default=7500, help='matched comparison step')
    ap.add_argument('--out', type=pathlib.Path, default=ROOT / 'results')
    ap.add_argument('--run', action='append', default=[],
                    help='override a run dir: "CELL=path" (repeatable)')
    ap.add_argument('--samples', action='append', default=[],
                    help='override a samples file: "CELL=path" (repeatable)')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    run_dirs = {c: ROOT / v[2] for c, v in CELLS.items()}
    sample_files = {c: ROOT / v[3] for c, v in CELLS.items()}
    for kv in args.run:
        c, p = kv.split('=', 1); run_dirs[c.strip()] = ROOT / p.strip()
    for kv in args.samples:
        c, p = kv.split('=', 1); sample_files[c.strip()] = ROOT / p.strip()

    # ---- 1. corpus and tokenizers ------------------------------------------
    print('Loading val split and train vocabulary ...')
    val_docs = load_split('val')
    val_text = '\n'.join(val_docs)
    n_chars, n_words = len(val_text), len(WORD.findall(val_text))
    train_vocab = {w.lower() for d in load_split('train') for w in WORD.findall(d)}

    tok_rows, toks, tok_per_char = [], {}, {}
    candidates = dict(TOKENIZERS)
    candidates['GPT-2 (English, reference only)'] = 'gpt2'
    for name, path in candidates.items():
        try:
            t = transformers.AutoTokenizer.from_pretrained(str(path))
        except Exception as e:  # gpt2 may not be cached; that row is optional
            print(f'  skipping {name}: {e.__class__.__name__}')
            continue
        ids = t(val_docs, add_special_tokens=False)['input_ids']
        n_tok = sum(len(x) for x in ids)
        tok_rows.append({'tokenizer': name, 'vocab': t.vocab_size, 'val_tokens': n_tok,
                         'tokens_per_word': n_tok / n_words, 'chars_per_token': n_chars / n_tok})
        if name in TOKENIZERS:
            toks[name] = t
            # One EOS per document is appended during training, so count it.
            tok_per_char[name] = (n_tok + len(val_docs)) / n_chars
            # Baseline entropy of real text, in the same 1024-token windows the models sample.
            flat = [i for x in ids for i in x]
            windows = [flat[k:k + BLOCK] for k in range(0, len(flat) - BLOCK + 1, BLOCK)]
            tok_rows[-1]['real_text_token_entropy'] = (
                sum(token_entropy(w) for w in windows) / max(len(windows), 1))
    tok_df = pd.DataFrame(tok_rows)
    tok_df.to_csv(args.out / 'tokenizer_table.csv', index=False)

    # ---- 2. training / validation metrics ----------------------------------
    rows, curves = [], {}
    for cell, (model, tok, _, _) in CELLS.items():
        df, files = load_metrics(run_dirs[cell])
        row = {'cell': cell, 'model': model, 'tokenizer': tok,
               'run_dir': str(run_dirs[cell].relative_to(ROOT)), 'log_files': len(files)}
        if df is None:
            print(f'  WARNING: no metrics.csv for {cell} under {run_dirs[cell]}')
            rows.append(row); continue
        val = series(df, 'val/nll')
        train = series(df, 'train/nll')
        k = tok_per_char.get(tok, float('nan')) / math.log(2)   # nats/token -> bits/char
        v_at, s_at = value_at(val, 'val/nll', args.step)
        t_at, _ = value_at(train, 'train/nll', args.step)
        best = val.loc[val['val/nll'].idxmin()] if not val.empty else None
        row.update({
            'last_step': int(df['step'].max()),
            'step_used': s_at,
            'val_nll_nats_per_token': v_at,
            'val_ppl_per_token': math.exp(v_at) if v_at == v_at else float('nan'),
            'val_bpc': v_at * k,
            'train_bpc': t_at * k,
            'train_val_gap_bpc': (v_at - t_at) * k,
            'best_val_bpc': float(best['val/nll']) * k if best is not None else float('nan'),
            'best_val_step': int(best['step']) if best is not None else None,
        })
        rows.append(row)
        curves[cell] = (val.assign(bpc=val['val/nll'] * k), train.assign(bpc=train['train/nll'] * k))

    # ---- 3. samples ---------------------------------------------------------
    samp_rows = []
    real = text_stats(val_docs, train_vocab)
    samp_rows.append({'cell': 'Real val text', **real})
    for cell, (model, tok, _, _) in CELLS.items():
        f = sample_files[cell]
        if not f.exists():
            continue
        recs = [json.loads(line) for line in f.read_text(encoding='utf-8').splitlines() if line.strip()]
        texts = [SPECIALS.sub(' ', r['text']) for r in recs]
        stats = text_stats(texts, train_vocab)
        stats['token_entropy'] = sum(token_entropy(r['ids']) for r in recs) / len(recs)
        stats['n_samples'] = len(recs)
        samp_rows.append({'cell': cell, **stats})
        for r in rows:
            if r['cell'] == cell:
                r.update({f'sample_{k}': v for k, v in stats.items()})
    samp_df = pd.DataFrame(samp_rows)
    if 'n_samples' in samp_df:
        samp_df['n_samples'] = samp_df['n_samples'].map(lambda v: '' if v != v else str(int(v)))
    if len(samp_rows) > 1:
        samp_df.to_csv(args.out / 'samples_table.csv', index=False)

    res = pd.DataFrame(rows)
    res.to_csv(args.out / 'results_table.csv', index=False)

    # ---- 4. figures ---------------------------------------------------------
    print('Figures:')
    if curves:
        fig, ax = plt.subplots(figsize=(7.5, 4.2))
        for cell, (val, _) in curves.items():
            model, tok = CELLS[cell][0], CELLS[cell][1]
            ax.plot(val['step'], val['bpc'], DASH[tok], color=COLOUR[model], lw=2, label=cell)
        ends = sorted(((v['bpc'].iloc[-1], v['step'].iloc[-1], c) for c, (v, _) in curves.items()))
        lo, hi = ax.get_ylim(); gap = 0.045 * (hi - lo); last = -1e9
        x_end = max(x for _, x, _ in ends)
        for y, x, c in ends:                      # push labels apart, bottom to top
            y_lab = max(y, last + gap); last = y_lab
            ax.text(x_end * 1.015, y_lab, c, va='center', fontsize=8, color=INK2,
                    clip_on=False)
        ax.axvline(args.step, color=INK2, lw=1, ls=':')
        ax.set(xlabel='optimizer step', ylabel='validation bits per character',
               title='Validation loss, all four cells (bits per character)')
        ax.legend(loc='upper right')
        save(fig, 'fig_val_curves.png', args.out)

        fig, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True, sharey=True)
        for ax, cell in zip(axes.flat, CELLS):
            ax.set_title(cell)
            if cell not in curves:
                ax.text(0.5, 0.5, 'no log found', ha='center', transform=ax.transAxes); continue
            val, train = curves[cell]
            ax.plot(train['step'], train['bpc'], color='#2a78d6', lw=2, label='train')
            ax.plot(val['step'], val['bpc'], color='#eb6834', lw=2, label='validation')
            ax.axvline(args.step, color=INK2, lw=1, ls=':')
        axes[0, 0].legend()
        for ax in axes[1]: ax.set_xlabel('optimizer step')
        for ax in axes[:, 0]: ax.set_ylabel('bits per character')
        fig.suptitle('Train vs validation per cell', fontweight='bold')
        save(fig, 'fig_train_val_gap.png', args.out)

    if 'val_bpc' in res:
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = {'Unigram': 0, 'BPE': 1}
        for _, r in res.dropna(subset=['val_bpc']).iterrows():
            x = xs[r['tokenizer']] + (-0.18 if r['model'] == 'MDLM' else 0.18)
            ax.bar(x, r['val_bpc'], width=0.34, color=COLOUR[r['model']],
                   label=r['model'] if r['tokenizer'] == 'Unigram' else None)
            ax.text(x, r['val_bpc'], f"{r['val_bpc']:.3f}", ha='center', va='bottom', fontsize=9)
        ax.set_xticks(list(xs.values()), list(xs.keys())); ax.grid(axis='x', visible=False)
        ax.set(ylabel='validation bits per character (lower is better)',
               title=f'Validation BPC at step {args.step}')
        ax.legend()
        save(fig, 'fig_val_bpc.png', args.out)

    if not tok_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.barh(tok_df['tokenizer'], tok_df['tokens_per_word'], color='#2a78d6', height=0.5)
        for y, v in enumerate(tok_df['tokens_per_word']):
            ax.text(v, y, f' {v:.2f}', va='center', fontsize=9)
        ax.invert_yaxis(); ax.grid(axis='y', visible=False)
        ax.set(xlabel='tokens per Swahili word (val split)', title='Tokenizer fertility')
        save(fig, 'fig_fertility.png', args.out)

    if len(samp_rows) > 1:
        fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
        for ax, col, label in [(axes[0], 'single_letter_per_1k_words', 'single-letter words per 1,000'),
                               (axes[1], 'oov_word_rate_%', 'words not seen in training (%)')]:
            sub = samp_df[samp_df['cell'] != 'Real val text']
            colours = [COLOUR[CELLS[c][0]] for c in sub['cell']]
            ax.barh(sub['cell'], sub[col], color=colours, height=0.5)
            for y, v in enumerate(sub[col]):
                ax.text(v, y, f' {v:.1f}', va='center', fontsize=9)
            ax.grid(axis='y', visible=False)
            ax.axvline(float(samp_df.loc[samp_df['cell'] == 'Real val text', col].iloc[0]),
                       color=INK, lw=1.5, ls=':', label='real val text')
            ax.invert_yaxis(); ax.set_xlabel(label); ax.legend(loc='lower right')
        fig.suptitle('Word-boundary artifact in generated samples', fontweight='bold')
        save(fig, 'fig_artifacts.png', args.out)

    # ---- 5. summary ---------------------------------------------------------
    def md(df, cols, fmt=3):
        df = df[[c for c in cols if c in df.columns]]
        try:
            return df.to_markdown(index=False, floatfmt=f'.{fmt}f')
        except ImportError:  # `tabulate` not installed: fall back to a plain table
            return '```\n' + df.to_string(index=False, float_format=lambda v: f'{v:.{fmt}f}') + '\n```'

    lines = [
        '# Swahili 2x2 results', '',
        f'Validation split: {len(val_docs)} documents, {n_words:,} words, {n_chars:,} characters.',
        f'Metrics compared at step {args.step} (the last validation point at or before it).', '',
        '## Validation loss', '',
        md(res, ['cell', 'step_used', 'val_nll_nats_per_token', 'val_ppl_per_token', 'val_bpc',
                 'train_bpc', 'train_val_gap_bpc', 'best_val_bpc', 'best_val_step', 'last_step']), '',
        'Bits per character (BPC) is the comparable number: per-token NLL and perplexity are not '
        'comparable across tokenizers. BPC = (val NLL / ln 2) x (val tokens / val characters), '
        'counting one EOS per document. For MDLM the validation NLL is the diffusion NELBO, so its '
        'BPC is an upper bound; for AR it is exact. A lower MDLM number is therefore a conservative '
        'result.', '',
        '## Tokenizers (val split)', '',
        md(tok_df, ['tokenizer', 'vocab', 'val_tokens', 'tokens_per_word', 'chars_per_token',
                    'real_text_token_entropy']), '',
    ]
    if len(samp_rows) > 1:
        lines += ['## Generated samples vs real text', '',
                  md(samp_df, ['cell', 'n_samples', 'words', 'single_letter_per_1k_words',
                               'oov_word_rate_%', 'distinct_2_per_500_words', 'token_entropy']), '',
                  'Real val text is the baseline: its single-letter and unseen-word rates are the '
                  'legitimate level, so the excess over it in a model row is the word-boundary '
                  'artifact. Token entropy is per sample over token ids; compare it to '
                  'real_text_token_entropy for the same tokenizer (a much lower value means '
                  'repetitive output).', '']
    else:
        lines += ['## Generated samples', '', 'No sample files found in analysis/samples/ yet.', '']
    (args.out / 'summary.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'\nWrote {args.out / "summary.md"}\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
#!/usr/bin/env python3
"""
Prompted-continuation evaluation (BLEU / chrF / chrF++) for the Swahili 2x2.

Task: take a passage from a TEST-split document, give the model the first part
(the prompt) and let it write the next part. Score what it wrote against what the
document actually says next.

  * The prompts are chosen ONCE, in words (not tokens), and saved to
    analysis/continuation/prompts.jsonl. Every cell -- both models, both
    tokenizers -- reuses that file, so all four are scored on identical text.
  * Each model generates as many tokens as the reference continuation has under
    that model's own tokenizer, so hypothesis and reference cover the same span.
  * AR:   standard left-to-right sampling from the prompt (Gumbel-max, temp 1,
          the same scheme as MDLM's own _ar_sampler).
  * MDLM: reverse diffusion starting from [prompt | masks]. MDLM's sampler never
          changes an unmasked token, so the prompt stays fixed and only the
          masked positions are filled -- the same update rule as ddpm_cache.
  * Scores are computed on detokenized text with sacrebleu, so they are
    comparable across tokenizers. A "random real continuation" baseline (each
    reference paired with a different prompt's true continuation) is reported too:
    it is the score fluent but unrelated Swahili gets, i.e. the floor.

Run from the repo root on a GPU node (one cell per call):
    python bleu_eval.py --cell ar_unigram --ckpt outputs/swahili_ar/checkpoints/best.ckpt \
        -- backbone=ar parameterization=ar model=small-ar data=swahili model.length=1024
Then, once all four cells have run, build the table (no GPU needed):
    python bleu_eval.py --summarise

Needs: pip install sacrebleu
"""

import argparse
import contextlib
import itertools
import json
import pathlib
import random
import re
import time

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / 'configs' / 'data' / 'maneno-yetu'
OUT = ROOT / 'analysis' / 'continuation'
PROMPTS = OUT / 'prompts.jsonl'
CELLS = ['mdlm_unigram', 'ar_unigram', 'mdlm_bpe', 'ar_bpe']
WORD = re.compile(r'[^\W\d_]+', re.UNICODE)


# ----------------------------------------------------------------- prompts
def build_prompts(n_prompts, prompt_words, cont_words, seed):
    """Pick passages from the test split, in word units, and save them once."""
    manifest = json.loads((DATA / 'splits' / 'split_manifest.json').read_text())
    docs = [(f, (DATA / 'data-raw' / 'cleaned' / f).read_text(encoding='utf-8').split())
            for f in manifest['splits']['test']]
    span = prompt_words + cont_words
    rng = random.Random(seed)
    slots = []                                   # every non-overlapping window
    for f, words in docs:
        for start in range(0, len(words) - span + 1, span):
            slots.append((f, start))
    rng.shuffle(slots)
    chosen = sorted(slots[:n_prompts])
    if len(chosen) < n_prompts:
        print(f'  only {len(chosen)} windows available; using all of them')
    words_by_doc = dict(docs)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(PROMPTS, 'w', encoding='utf-8') as fh:
        for i, (f, start) in enumerate(chosen):
            w = words_by_doc[f]
            fh.write(json.dumps({
                'id': i, 'doc': f, 'start_word': start,
                'prompt': ' '.join(w[start:start + prompt_words]),
                'reference': ' '.join(w[start + prompt_words:start + span]),
            }, ensure_ascii=False) + '\n')
    print(f'Wrote {len(chosen)} prompts to {PROMPTS}')


def load_prompts():
    return [json.loads(l) for l in PROMPTS.read_text(encoding='utf-8').splitlines() if l.strip()]


# ---------------------------------------------------------------- sampling
def gumbel_categorical(probs, fp64):
    """Same trick as MDLM's _sample_categorical: argmax(p / Exponential noise)."""
    import torch
    if fp64:
        probs = probs.double()
    noise = 1e-10 - (torch.rand_like(probs) + 1e-10).log()
    return (probs / noise).argmax(dim=-1)


@contextlib.contextmanager
def ema_weights(model):
    """Swap in the EMA weights, exactly as restore_model_and_sample does."""
    params = lambda: itertools.chain(model.backbone.parameters(), model.noise.parameters())
    if model.ema:
        model.ema.store(params()); model.ema.copy_to(params())
    model.backbone.eval(); model.noise.eval()
    try:
        yield
    finally:
        if model.ema:
            model.ema.restore(params())


def ar_continue(model, prefix_ids, n_new, eos_id):
    """Left-to-right sampling from a prompt (batch of 1)."""
    import torch
    x = torch.tensor([prefix_ids], device='cuda')
    for _ in range(n_new):
        logits = model.forward(x, None)[:, -1].float()
        gumbel = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
        nxt = (logits + gumbel).argmax(-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
    out = x[0, len(prefix_ids):].tolist()
    return out[:out.index(eos_id)] if eos_id in out else out


def mdlm_continue(model, batch, length, steps, eos_id, fp64, eps=1e-5):
    """Reverse diffusion from [prompt | masks]; prompts stay fixed throughout.

    batch: list of (prefix_ids, n_new). Rows can have different prompt lengths.
    Mirrors diffusion._ddpm_caching_update (loglinear noise: move chance == t).
    """
    import torch
    mask = model.mask_index
    x = torch.full((len(batch), length), mask, dtype=torch.long, device='cuda')
    for r, (prefix, _) in enumerate(batch):
        x[r, :len(prefix)] = torch.tensor(prefix, device='cuda')
    timesteps = torch.linspace(1, eps, steps + 1, device='cuda')
    dt = (1 - eps) / steps
    p_x0 = None
    for i in range(steps):
        t = timesteps[i] * torch.ones(x.shape[0], 1, device='cuda')
        if p_x0 is None:
            sigma_t, _ = model.noise(t)
            p_x0 = model.forward(x, sigma_t).exp()
        move_t = t[:, None]                      # (B,1,1)
        move_s = (t - dt)[:, None]
        q_xs = p_x0 * (move_t - move_s)
        q_xs[:, :, mask] = move_s[:, :, 0]
        sampled = gumbel_categorical(q_xs, fp64)
        keep = (x != mask).to(x.dtype)           # unmasked tokens never change
        x_next = keep * x + (1 - keep) * sampled
        if not torch.equal(x_next, x):
            p_x0 = None                          # cache only while nothing changes
        x = x_next
    # final noise removal: fill anything still masked greedily, as in _sample
    t = timesteps[-1] * torch.ones(x.shape[0], 1, device='cuda')
    final = model.forward(x, model.noise(t)[0]).argmax(dim=-1)
    x = torch.where(x == mask, final, x)
    outs = []
    for r, (prefix, n_new) in enumerate(batch):
        seq = x[r, len(prefix):len(prefix) + n_new].tolist()
        outs.append(seq[:seq.index(eos_id)] if eos_id in seq else seq)
    return outs


def run_cell(args):
    import hydra
    import lightning as L
    import torch
    import main as _mdlm_main  # noqa: F401  (registers the OmegaConf resolvers)
    import dataloader
    import diffusion

    if not torch.cuda.is_available():
        raise SystemExit('ERROR: no usable GPU on this node -- add it to --exclude.')
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'configs'), version_base=None):
        cfg = hydra.compose(config_name='config', overrides=list(args.overrides) + [
            f'eval.checkpoint_path={args.ckpt.resolve()}'])
    L.seed_everything(args.seed)
    tok = dataloader.get_tokenizer(cfg)
    model = diffusion.Diffusion.load_from_checkpoint(
        str(args.ckpt), tokenizer=tok, config=cfg).to('cuda')
    is_ar = model.parameterization == 'ar'
    length = cfg.model.length
    print(f'{args.cell}: {args.ckpt}  ({"AR" if is_ar else "MDLM"}, length {length})')

    prompts = load_prompts()
    jobs = []
    for p in prompts:
        prefix = [tok.bos_token_id] + tok(p['prompt'], add_special_tokens=False)['input_ids']
        n_new = len(tok(' ' + p['reference'], add_special_tokens=False)['input_ids'])
        if len(prefix) + n_new > length:
            prefix = prefix[:1] + prefix[-(length - n_new - 1):]   # keep the end of the prompt
        jobs.append((p, prefix, n_new))

    out_file = OUT / f'{args.cell}.jsonl'
    t0 = time.time()
    with ema_weights(model), torch.no_grad(), open(out_file, 'w', encoding='utf-8') as fh:
        if is_ar:
            results = []
            for k, (p, prefix, n_new) in enumerate(jobs):
                results.append(ar_continue(model, prefix, n_new, tok.eos_token_id))
                if (k + 1) % 10 == 0:
                    print(f'  {k + 1}/{len(jobs)}  {time.time() - t0:.0f}s')
        else:
            results = []
            for b in range(0, len(jobs), args.batch_size):
                chunk = jobs[b:b + args.batch_size]
                results += mdlm_continue(model, [(pre, n) for _, pre, n in chunk], length,
                                         args.steps, tok.eos_token_id, args.fp64)
                print(f'  {min(b + args.batch_size, len(jobs))}/{len(jobs)}  {time.time() - t0:.0f}s')
        for (p, prefix, n_new), ids in zip(jobs, results):
            fh.write(json.dumps({
                'id': p['id'], 'cell': args.cell, 'checkpoint': str(args.ckpt),
                'prompt_tokens': len(prefix), 'target_tokens': n_new,
                'hypothesis': tok.decode(ids, skip_special_tokens=True).strip(),
                'reference': p['reference'], 'hypothesis_ids': ids,
            }, ensure_ascii=False) + '\n')
    print(f'Wrote {out_file} in {time.time() - t0:.0f}s')


# ----------------------------------------------------------------- scoring
def single_letter_rate(texts):
    words = [w for t in texts for w in WORD.findall(t)]
    return 1000 * sum(len(w) == 1 for w in words) / max(len(words), 1)


METRICS = None


def score(hyps, refs):
    global METRICS
    from sacrebleu.metrics import BLEU, CHRF
    METRICS = METRICS or (BLEU(), CHRF(), CHRF(word_order=2))
    return tuple(m.corpus_score(hyps, [refs]) for m in METRICS)


def summarise():
    refs_all = {p['id']: p['reference'] for p in load_prompts()}
    rows = []
    # Floor: each reference paired with a DIFFERENT prompt's real continuation.
    ids = sorted(refs_all)
    shifted = [refs_all[ids[(k + 1) % len(ids)]] for k in range(len(ids))]
    gold = [refs_all[i] for i in ids]
    b, c, cpp = score(shifted, gold)
    rows.append(('Random real continuation (floor)', len(ids), b.score, c.score, cpp.score,
                 single_letter_rate(shifted)))
    for cell in CELLS:
        f = OUT / f'{cell}.jsonl'
        if not f.exists():
            continue
        recs = [json.loads(l) for l in f.read_text(encoding='utf-8').splitlines() if l.strip()]
        hyps = [r['hypothesis'] for r in recs]
        refs = [r['reference'] for r in recs]
        b, c, cpp = score(hyps, refs)
        rows.append((cell, len(recs), b.score, c.score, cpp.score, single_letter_rate(hyps)))
    rows.append(('Reference text itself', len(ids), 100.0, 100.0, 100.0, single_letter_rate(gold)))

    header = ('| Cell | Prompts | BLEU | chrF | chrF++ | Single-letter words per 1k |\n'
              '|---|---|---|---|---|---|')
    lines = [header] + [f'| {r[0]} | {r[1]} | {r[2]:.2f} | {r[3]:.2f} | {r[4]:.2f} | {r[5]:.1f} |'
                        for r in rows]
    text = '\n'.join([
        '# Prompted continuation on the test split', '',
        *lines, '',
        'BLEU and chrF are computed on detokenized text with sacrebleu, so they compare across '
        'tokenizers. The floor row pairs each reference with an unrelated real continuation: '
        'scores near it mean the model writes fluent but unrelated Swahili. The single-letter '
        'rate in the last column measures the word-boundary artifact; compare each model with '
        'the reference-text row.', '',
        'sacrebleu signatures: ' + ' / '.join(f'`{m.get_signature()}`' for m in METRICS),
    ])
    (OUT / 'summary.md').write_text(text, encoding='utf-8')
    print(text)


# -------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cell', choices=CELLS)
    ap.add_argument('--ckpt', type=pathlib.Path)
    ap.add_argument('--summarise', action='store_true', help='score all cells that have run')
    ap.add_argument('--n-prompts', type=int, default=100)
    ap.add_argument('--prompt-words', type=int, default=150)
    ap.add_argument('--cont-words', type=int, default=50)
    ap.add_argument('--steps', type=int, default=1000, help='MDLM denoising steps')
    ap.add_argument('--batch-size', type=int, default=8, help='MDLM prompts per batch')
    ap.add_argument('--fp64', action='store_true', help='float64 Gumbel noise for MDLM')
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('overrides', nargs='*', help='Hydra overrides, after a literal --')
    args = ap.parse_args()

    if not PROMPTS.exists():
        build_prompts(args.n_prompts, args.prompt_words, args.cont_words, seed=42)
    if args.summarise:
        summarise()
    elif args.cell and args.ckpt:
        if not args.ckpt.exists():
            raise SystemExit(f'ERROR: checkpoint not found: {args.ckpt}')
        run_cell(args)
    else:
        print(f'Prompts are in {PROMPTS}. Give --cell and --ckpt to run a model, '
              f'or --summarise to score.')


if __name__ == '__main__':
    main()
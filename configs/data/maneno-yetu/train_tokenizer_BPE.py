#!/usr/bin/env python3
"""
Train a SentencePiece BPE tokenizer on the train split only, and save it in
a format transformers.AutoTokenizer.from_pretrained() can load directly --
matching exactly how MDLM's dataloader.py loads tokenizers.

RQ2 counterpart to train_tokenizer.py (Unigram). Both are trained on the
same 664 train files at the same vocab size with the same special-token
ordering (pad=0, unk=1, bos=2, eos=3), so the only variable between them is
the segmentation algorithm:

  Unigram  starts from a large seed vocabulary and prunes probabilistically,
           scoring whole segmentations by likelihood.
  BPE      greedily merges the most frequent adjacent pair, repeatedly, with
           no notion of segmentation likelihood.

Hypothesis for agglutinative Swahili: BPE should over-merge across morpheme
boundaries exactly where prefix+root combinations are frequent (ame-, wana-,
ku-), giving worse morphological alignment than Unigram at equal vocab size.

Run from anywhere:
    python train_tokenizer_bpe.py
"""

import argparse
import pathlib

from tokenizers import SentencePieceBPETokenizer
import transformers

from split import load_split, DEFAULT_DATA_DIR, HERE

DEFAULT_MANIFEST = HERE / "splits" / "split_manifest.json"
DEFAULT_OUT_DIR = HERE / "tokenizer-bpe"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--vocab-size", type=int, default=16000,
                     help="Must match the Unigram tokenizer's vocab size, or "
                          "the RQ2 comparison confounds algorithm with "
                          "vocabulary capacity.")
    ap.add_argument("--min-frequency", type=int, default=2,
                     help="BPE-only: minimum pair frequency to consider a "
                          "merge. Unigram has no equivalent knob.")
    ap.add_argument("--limit-alphabet", type=int, default=1000,
                     help="BPE-only: max distinct base characters kept. "
                          "1000 is ample for Swahili's Latin alphabet; "
                          "characters beyond it become unreachable.")
    args = ap.parse_args()

    splits = load_split(args.manifest, data_dir=args.data_dir)
    train_files = [str(p) for p in splits["train"]]
    print(f"Training on {len(train_files)} train files (val/test untouched)")

    # NOTE: unlike Unigram, BPE takes unk_token in the CONSTRUCTOR, not in
    # train(). Passing it to train() is silently ignored (it is not even a
    # parameter there), which would leave the model without a fallback --
    # the same class of failure as the Unigram unk_id bug.
    tok = SentencePieceBPETokenizer(unk_token="<unk>")
    tok.train(
        files=train_files,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        limit_alphabet=args.limit_alphabet,
        # Order fixes the ids: pad=0, unk=1, bos=2, eos=3 -- identical to the
        # Unigram tokenizer, so only segmentation differs between the two.
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
    )
    # Deliberately NOT setting a post_processor -- an auto-wrapping one makes
    # MDLM's own `EOS = tokenizer.encode(tokenizer.eos_token)[0]` return the
    # wrong id whenever bos_token_id != eos_token_id. MDLM tokenizes each
    # document with add_special_tokens=False and adds BOS/EOS itself in
    # _group_texts, so a post-processor is unnecessary as well as harmful.

    hf_tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hf_tok.save_pretrained(args.out_dir)

    # Self-checks: reload exactly as MDLM's get_tokenizer() will.
    reloaded = transformers.AutoTokenizer.from_pretrained(args.out_dir)

    eos_extracted = reloaded.encode(reloaded.eos_token)
    bos_extracted = reloaded.encode(reloaded.bos_token)
    assert eos_extracted == [reloaded.eos_token_id], (
        f"EOS extraction broken: encode(eos_token)={eos_extracted}, "
        f"expected [{reloaded.eos_token_id}]")
    assert bos_extracted == [reloaded.bos_token_id], (
        f"BOS extraction broken: encode(bos_token)={bos_extracted}, "
        f"expected [{reloaded.bos_token_id}]")

    # Regression guard for the unk_id class of bug: characters absent from
    # the training split must map to <unk>, not raise. This is what crashed
    # the first training run during val-set tokenization.
    probe = reloaded("café © Žurnal", add_special_tokens=False)["input_ids"]
    assert len(probe) > 0, "unk fallback broken -- check unk_token placement"

    # Ids must line up with the Unigram tokenizer, or the two runs differ in
    # more than just segmentation.
    assert (reloaded.pad_token_id, reloaded.unk_token_id,
            reloaded.bos_token_id, reloaded.eos_token_id) == (0, 1, 2, 3), (
        "special token ids differ from the Unigram tokenizer's (0,1,2,3)")

    print(f"Saved to {args.out_dir}")
    print(f"vocab_size={reloaded.vocab_size}  "
          f"bos={reloaded.bos_token}({reloaded.bos_token_id})  "
          f"eos={reloaded.eos_token}({reloaded.eos_token_id})  "
          f"pad={reloaded.pad_token}({reloaded.pad_token_id})")
    print("Self-checks passed: BOS/EOS extraction, unk fallback, id alignment.")


if __name__ == "__main__":
    main()

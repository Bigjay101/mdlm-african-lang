#!/usr/bin/env python3
"""
Train a SentencePiece (Unigram) tokenizer on the train split only, and
save it in a format transformers.AutoTokenizer.from_pretrained() can load
directly -- matching exactly how MDLM's dataloader.py loads tokenizers.

Reads file membership from split_manifest.json via split.py's load_split(),
so training only ever touches the train set -- zero chance of val/test
leaking into the vocabulary.

Run from anywhere:
    python train_tokenizer.py
"""

import argparse
import pathlib

from tokenizers import SentencePieceUnigramTokenizer
import transformers

from split import load_split, DEFAULT_DATA_DIR, HERE

DEFAULT_MANIFEST = HERE / "splits" / "split_manifest.json"
DEFAULT_OUT_DIR = HERE / "tokenizer"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=pathlib.Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--data-dir", type=pathlib.Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--out-dir", type=pathlib.Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--vocab-size", type=int, default=16000,
                     help="Starting point for a corpus this size (~20-40M "
                          "tokens). GPT-2's OWT tokenizer uses ~50k on a "
                          "corpus orders of magnitude larger -- scale down "
                          "accordingly, adjust empirically if needed.")
    args = ap.parse_args()

    splits = load_split(args.manifest, data_dir=args.data_dir)
    train_files = [str(p) for p in splits["train"]]
    print(f"Training on {len(train_files)} train files (val/test untouched)")

    tok = SentencePieceUnigramTokenizer()
    tok.train(
        files=train_files,
        vocab_size=args.vocab_size,
        special_tokens=["<pad>", "<unk>", "<s>", "</s>"],
        unk_token="<unk>",          # <-- REQUIRED: sets unk_id on the Unigram model.
                                     # Listing <unk> in special_tokens is NOT enough;
                                     # without this, any character unseen during
                                     # training raises instead of mapping to <unk>.
    )
    # Deliberately NOT setting a post_processor. See module docstring --
    # an auto-wrapping post-processor makes MDLM's own
    #     EOS = tokenizer.encode(tokenizer.eos_token)[0]
    # silently return the wrong id whenever bos_token_id != eos_token_id.
    # MDLM's per-document tokenization already uses add_special_tokens=False
    # and adds BOS/EOS manually in _group_texts, so this is unnecessary
    # as well as harmful.

    hf_tok = transformers.PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>", eos_token="</s>",
        unk_token="<unk>", pad_token="<pad>",
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    hf_tok.save_pretrained(args.out_dir)

    # Self-check: reload exactly as MDLM's get_tokenizer() will, and
    # confirm the extraction line MDLM actually uses gives correct ids.
    reloaded = transformers.AutoTokenizer.from_pretrained(args.out_dir)
    eos_extracted = reloaded.encode(reloaded.eos_token)
    bos_extracted = reloaded.encode(reloaded.bos_token)
    assert eos_extracted == [reloaded.eos_token_id], (
        f"EOS extraction broken: encode(eos_token)={eos_extracted}, "
        f"expected [{reloaded.eos_token_id}]")
    assert bos_extracted == [reloaded.bos_token_id], (
        f"BOS extraction broken: encode(bos_token)={bos_extracted}, "
        f"expected [{reloaded.bos_token_id}]")

    print(f"Saved to {args.out_dir}")
    print(f"vocab_size={reloaded.vocab_size}  "
          f"bos={reloaded.bos_token}({reloaded.bos_token_id})  "
          f"eos={reloaded.eos_token}({reloaded.eos_token_id})  "
          f"pad={reloaded.pad_token}({reloaded.pad_token_id})")
    print("Self-check passed: MDLM's BOS/EOS extraction will work correctly.")


if __name__ == "__main__":
    main()
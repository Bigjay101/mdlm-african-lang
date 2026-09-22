import transformers, pathlib, json

root = pathlib.Path('configs/data/maneno-yetu')
m = json.loads((root/'splits/split_manifest.json').read_text())
d = root/'data-raw/cleaned'
text = "\n".join((d/f).read_text(encoding='utf-8') for f in m['splits']['val'])
words, chars = len(text.split()), len(text)
print(f"val split: {words:,} words, {chars:,} chars\n")

for name, p in [('Unigram 16k', root/'tokenizer'),
                ('BPE 16k',     root/'tokenizer-bpe'),
                ('GPT-2 (EN)',  'gpt2')]:
    t = transformers.AutoTokenizer.from_pretrained(str(p))
    n = len(t(text, add_special_tokens=False)['input_ids'])
    print(f"{name:12s} {n:>10,} tokens | {n/words:.3f} tok/word | {chars/n:.3f} chars/tok")
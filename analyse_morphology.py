import re, random
from collections import Counter

wc = Counter()
for f in m['splits']['train']:
    wc.update(w.lower() for w in re.findall(r"[^\W\d_]+", (d/f).read_text(encoding='utf-8'), re.UNICODE))

rare = [w for w, c in wc.items() if c <= 2]
common = [w for w, c in wc.most_common(500)]
print(f"{len(rare):,} rare types of {len(wc):,} total\n")

uni = transformers.AutoTokenizer.from_pretrained(str(root/'tokenizer'))
bpe = transformers.AutoTokenizer.from_pretrained(str(root/'tokenizer-bpe'))

random.seed(0)
for label, sample in [('RARE', random.sample(rare, 40)), ('COMMON', random.sample(common, 20))]:
    print(f"=== {label} ===")
    for w in sample:
        u = uni.convert_ids_to_tokens(uni(w, add_special_tokens=False)['input_ids'])
        b = bpe.convert_ids_to_tokens(bpe(w, add_special_tokens=False)['input_ids'])
        print(f"{w:22s} U: {'|'.join(u):32s} B: {'|'.join(b)}")
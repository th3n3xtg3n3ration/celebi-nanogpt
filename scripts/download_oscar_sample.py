import os
import argparse
from datasets import load_dataset


def main():
    """
    Downloads a streaming sample of the OSCAR dataset (unshuffled_deduplicated)
    for specified languages, and saves them as raw UTF-8 bytes (train.bin, val.bin).
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", nargs="+", default=["tr", "en"])
    ap.add_argument("--docs_per_lang", type=int, default=3000)
    ap.add_argument("--out_dir", default="data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = bytearray()
    val = bytearray()

    for lang in args.langs:
        ds = load_dataset(
            "oscar-corpus/oscar",
            f"unshuffled_deduplicated_{lang}",
            split="train",
            streaming=True,
        )
        i = 0
        for ex in ds:
            txt = ex.get("text", "")
            if not txt:
                continue
            b = txt.encode("utf-8", errors="ignore")
            if len(b) < 300:
                continue

            if (i % 10) == 0:
                val.extend(b + b"\n")
            else:
                train.extend(b + b"\n")

            i += 1
            if i >= args.docs_per_lang:
                break

        print("lang:", lang, "docs:", i)

    with open(os.path.join(args.out_dir, "train.bin"), "wb") as f:
        f.write(train)
    with open(os.path.join(args.out_dir, "val.bin"), "wb") as f:
        f.write(val)

    print("ok:", len(train), "train bytes |", len(val), "val bytes")


if __name__ == "__main__":
    main()

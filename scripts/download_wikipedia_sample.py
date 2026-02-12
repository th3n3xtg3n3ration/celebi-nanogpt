import os
import argparse
from datasets import load_dataset, get_dataset_config_names


def get_latest_config(lang):
    """
    Returns the latest available Wikipedia config for a given language.
    Configs are usually named like "20231101.en".
    We filter for ones ending in .{lang} and sort them to pick the latest date.
    """
    configs = get_dataset_config_names("wikimedia/wikipedia")
    # Filter for configs that match "YYYYMMDD.{lang}" pattern essentially
    # We look for configs ending with f".{lang}"
    lang_configs = [c for c in configs if c.endswith(f".{lang}")]
    
    if not lang_configs:
        raise ValueError(f"No Wikipedia config found for language: {lang}")
    
    # Sort to get the latest date (lexicographical sort works for YYYYMMDD)
    lang_configs.sort()
    return lang_configs[-1]


def main():
    ap = argparse.ArgumentParser(description="Download a sample of Wikipedia data.")
    ap.add_argument("--langs", nargs="+", default=["tr", "en"], help="Languages to download (e.g. tr en)")
    ap.add_argument("--docs_per_lang", type=int, default=3000, help="Number of documents to sample per language")
    ap.add_argument("--out_dir", default="data", help="Output directory for train.bin and val.bin")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    train = bytearray()
    val = bytearray()

    print(f"Downloading Wikipedia sample for languages: {args.langs}")

    for lang in args.langs:
        try:
            config_name = get_latest_config(lang)
            print(f"[{lang}] Using config: {config_name}")
            
            ds = load_dataset(
                "wikimedia/wikipedia",
                config_name,
                split="train",
                streaming=True,
                trust_remote_code=True
            )
            
            i = 0
            for ex in ds:
                txt = ex.get("text", "")
                if not txt:
                    continue
                
                b = txt.encode("utf-8", errors="ignore")
                # Skip very short documents to avoid noise
                if len(b) < 100:
                    continue

                if (i % 10) == 0:
                    val.extend(b + b"\n")
                else:
                    train.extend(b + b"\n")

                i += 1
                if i >= args.docs_per_lang:
                    break
            
            print(f"[{lang}] collected {i} documents.")
            
        except Exception as e:
            print(f"[{lang}] Error: {e}")

    # Write output files
    train_path = os.path.join(args.out_dir, "train.bin")
    val_path = os.path.join(args.out_dir, "val.bin")

    with open(train_path, "wb") as f:
        f.write(train)
    with open(val_path, "wb") as f:
        f.write(val)

    print(f"Done! Written to {args.out_dir}")
    print(f"train.bin: {len(train)} bytes")
    print(f"val.bin:   {len(val)} bytes")


if __name__ == "__main__":
    main()

import os
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="train.txt (utf-8)")
    ap.add_argument("--out", dest="out", required=True, help="data/train.bin")
    args = ap.parse_args()

    with open(args.inp, "rb") as f:
        b = f.read()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(b)

    print("wrote:", args.out, "bytes:", len(b))


if __name__ == "__main__":
    main()

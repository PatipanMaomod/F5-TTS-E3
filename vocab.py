import pandas as pd
from pathlib import Path


def build_vocab(data_path: str, vocab_path: str):
    """
    สร้าง vocab.txt จาก metadata.csv
    แต่ละบรรทัดคือ 1 character
    """
    df = pd.read_csv(
        Path(data_path) / "metadata.csv",
        sep="|", names=["path", "text"]
    )

    chars = set()
    for text in df["text"]:
        chars.update(list(str(text)))

    # เรียง + เพิ่ม special tokens
    vocab = ["<pad>", "<unk>", "<bos>", "<eos>"] + sorted(chars)

    with open(vocab_path, "w", encoding="utf-8") as f:
        for c in vocab:
            f.write(c + "\n")

    print(f"✅ Vocab size: {len(vocab)} → saved to {vocab_path}")
    return vocab


if __name__ == "__main__":
    build_vocab("./data/train", "./data/vocab.txt")
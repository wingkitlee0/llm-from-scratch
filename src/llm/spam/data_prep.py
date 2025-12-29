import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_DATAFILE = "data/sms_spam_collection/SMSSpamCollection.tsv"


def read_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", header=None, names=["label", "text"])
    return df


def create_balanced_dataset(df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    spam_subset: pd.DataFrame = df[df["label"] == "spam"]
    num_spam = spam_subset.shape[0]
    ham_subset = df[df["label"] == "ham"].sample(num_spam, random_state=random_state)

    balanced_df: pd.DataFrame = pd.concat([ham_subset, spam_subset], ignore_index=True)
    return balanced_df


def random_split_dataset(
    df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    train_endidx = int(train_frac * len(df))
    val_endidx = train_endidx + int(val_frac * len(df))
    train_df = df.iloc[:train_endidx]
    val_df = df.iloc[train_endidx:val_endidx]
    test_df = df.iloc[val_endidx:]
    return train_df, val_df, test_df


def main(datestr: str, datafile: str):
    df = read_data(datafile)
    print(df.head())
    bdf = create_balanced_dataset(df, random_state=42)
    bdf["label"] = bdf["label"].map({"ham": 0, "spam": 1})
    print(bdf.describe())

    train_df, val_df, test_df = random_split_dataset(
        bdf, train_frac=0.7, val_frac=0.1, random_state=42
    )

    output_dir = Path(datafile).parent / Path(datestr)
    output_dir.mkdir(parents=True, exist_ok=True)
    for df, filename in [
        (train_df, "train.csv"),
        (val_df, "val.csv"),
        (test_df, "test.csv"),
    ]:
        df.to_csv(output_dir / filename, index=False)

    print(f"Saved datasets to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datestr", type=str, default=datetime.now().strftime("%Y%m%d%H%M%S")
    )
    parser.add_argument(
        "--datafile", type=str, default=DEFAULT_DATAFILE, help="Data file"
    )
    args = parser.parse_args()
    main(datestr=args.datestr, datafile=args.datafile)

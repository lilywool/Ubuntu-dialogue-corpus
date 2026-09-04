"""
Read-only, one-time schema inspection of df_with_sentiment.pkl in the
Ubuntu checkout. Authorized scope: report columns, dtypes, shape, index
structure, and whether VADER fields exist. WRITES NOTHING -- no file is
created or modified by this script, in this repo or the source checkout.

Must be run natively (not through Claude's Linux-VM bridge -- that VM has
only ~3.5GB available RAM and no swap, which is not safe for a ~3GB pickle
deserialization). Run with the transformer clone's own .venm interpreter,
e.g.:

    ..\.venv\python.exe scripts\inspect_source_schema.py "C:\Users\lilli_37fwt34\Home\Projects\1. Github Root Repos\2. Ubuntu Dialogue Corpus\1. Ubuntu-dialogue-corpus\df_with_sentiment.pkl"

Before running: confirm available system RAM is comfortably above the
source file's size (ideally 3-4x the ~2.9GB pickle size, i.e. roughly
10-12GB free, to be safe for object-dtype overhead during unpickling).
"""
import sys
import psutil  # optional; script still runs without it, see except below

def check_memory(min_gb=8):
    try:
        import psutil
        avail_gb = psutil.virtual_memory().available / (1024 ** 3)
        print(f"Available RAM: {avail_gb:.1f} GB")
        if avail_gb < min_gb:
            print(
                f"WARNING: available RAM ({avail_gb:.1f} GB) is below the "
                f"recommended {min_gb} GB minimum for a ~2.9GB pickle. "
                "Consider closing other applications before continuing."
            )
    except ImportError:
        print("psutil not installed -- skipping automated memory check; "
              "confirm available RAM manually (Task Manager) before proceeding.")

def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python inspect_source_schema.py <path-to-df_with_sentiment.pkl>")
    source_path = sys.argv[1]

    check_memory()

    import pandas as pd
    print(f"Loading {source_path} ...")
    df = pd.read_pickle(source_path)

    print("\n=== shape ===")
    print(df.shape)

    print("\n=== index ===")
    print(f"type: {type(df.index).__name__}")
    print(f"name: {df.index.name}")
    print(df.index[:5])

    print("\n=== columns & dtypes ===")
    for col, dtype in df.dtypes.items():
        print(f"  {col!r}: {dtype}")

    print("\n=== VADER fields present? ===")
    vader_cols = [c for c in df.columns if "vader" in c.lower()]
    print(vader_cols if vader_cols else "None found")

    print("\n=== transformer fields present? ===")
    transformer_cols = [c for c in df.columns if any(k in c.lower() for k in ("transformer", "sentiment_score", "roberta"))]
    print(transformer_cols if transformer_cols else "None found")

    print("\n=== memory usage (deep) ===")
    print(df.memory_usage(deep=True).sum() / (1024 ** 3), "GB")

    # explicitly no df.to_csv / df.to_pickle / any write call anywhere.

if __name__ == "__main__":
    main()

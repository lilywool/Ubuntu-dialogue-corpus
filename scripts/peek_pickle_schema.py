"""
Memory-safe partial schema peek at a pandas .pkl file.

Why this exists: a full pd.read_pickle() of a multi-GB DataFrame pickle
needs many GB of RAM to deserialize. For schema purposes we don't need the
data -- only the column names and structural class names. This script never
deserializes: it inspects the pickle byte/opcode stream directly, reading
only a bounded slice of the file. Peak memory is bounded by the slice size
(default 4MB), not by the file size.

Two modes, because where the schema lives depends on the pandas version:

  head : walk the opcode stream from the start with pickletools.genops.
         Useful when the DataFrame's axes (axes[0] = columns) are written
         before the block data. On pandas 2.x this typically stops almost
         immediately -- the BlockManager reduces to _unpickle_block calls
         whose numpy buffers come FIRST, so the walk hits a multi-MB
         bytearray within the first few hundred bytes.

  tail : scan the last N bytes for pickle string records and report the
         printable ones in stream order. On pandas 2.x the axes (columns
         index, then row index) are written near the END of the stream,
         so this is where the column names actually are. This is a byte
         pattern scan, not a full opcode parse, so it can surface a few
         false positives from adjacent binary data -- read the results as
         candidates anchored by the pandas index class names around them.

Scanned records: SHORT_BINUNICODE (0x8c + 1-byte length) and BINUNICODE
(0x58 + 4-byte LE length).

Read-only: opens the file 'rb' and writes nothing, anywhere. Stdlib only
(no third-party imports), so the interpreter used is immaterial and no
packages are installed into any environment.

Limits: reveals column names and structural class names. Does NOT give
per-column dtypes or an exact row count -- those live in block descriptors
interleaved with the data arrays. A full load is still needed for those,
and for the actual sample extraction.

Usage:
    python scripts/peek_pickle_schema.py <path.pkl> [head|tail] [bytes]
"""
import io
import pickletools
import re
import string
import sys

STRING_OPS = {
    "SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE",
    "SHORT_BINSTRING", "BINSTRING",
}
COMPRESSION_MAGIC = {
    b"\x1f\x8b": "gzip", b"BZh": "bzip2", b"\xfd7zXZ": "xz",
    b"PK\x03\x04": "zip", b"\x28\xb5\x2f\xfd": "zstd",
}
PRINTABLE = set(string.printable) - set("\t\n\r\x0b\x0c")


def is_plausible(s, max_len=80):
    return 1 <= len(s) <= max_len and all(c in PRINTABLE for c in s)


def scan_head(path, nbytes):
    with open(path, "rb") as f:
        head = f.read(nbytes)
    for magic, name in COMPRESSION_MAGIC.items():
        if head.startswith(magic):
            sys.exit(f"File appears {name}-compressed; needs a raw pickle stream.")
    if not head.startswith(b"\x80"):
        sys.exit("File does not start with a pickle PROTO opcode.")
    print(f"Protocol: {head[1]}")

    found, n_ops, stopped = [], 0, None
    try:
        for opcode, arg, pos in pickletools.genops(io.BytesIO(head)):
            n_ops += 1
            if opcode.name in STRING_OPS and isinstance(arg, str):
                found.append((pos, arg))
    except Exception as exc:
        stopped = f"{type(exc).__name__}: {exc}"

    print(f"Walked {n_ops:,} opcodes; stopped: {stopped or 'end of prefix'}\n")
    for i, (pos, s) in enumerate(found[:400]):
        print(f"  [{i:3d}] @{pos:<12,} {s!r}")


def scan_tail(path, nbytes):
    with open(path, "rb") as f:
        f.seek(0, io.SEEK_END)
        size = f.tell()
        start = max(0, size - nbytes)
        f.seek(start)
        buf = f.read()

    print(f"File size: {size:,} bytes; scanning last {len(buf):,} bytes "
          f"(offset {start:,} onward)\n")

    found, i, n = [], 0, len(buf)
    while i < n:
        b = buf[i]
        if b == 0x8C and i + 1 < n:  # SHORT_BINUNICODE
            ln = buf[i + 1]
            if i + 2 + ln <= n:
                try:
                    s = buf[i + 2:i + 2 + ln].decode("utf-8")
                except UnicodeDecodeError:
                    i += 1
                    continue
                if is_plausible(s):
                    found.append((start + i, "SHORT_BINUNICODE", s))
                    i += 2 + ln
                    continue
        elif b == 0x58 and i + 5 <= n:  # BINUNICODE
            ln = int.from_bytes(buf[i + 1:i + 5], "little")
            if 0 < ln <= 200 and i + 5 + ln <= n:
                try:
                    s = buf[i + 5:i + 5 + ln].decode("utf-8")
                except UnicodeDecodeError:
                    i += 1
                    continue
                if is_plausible(s, max_len=200):
                    found.append((start + i, "BINUNICODE", s))
                    i += 5 + ln
                    continue
        i += 1

    print(f"=== {len(found):,} plausible string records, in stream order ===")
    for idx, (pos, op, s) in enumerate(found):
        print(f"  [{idx:4d}] @{pos:<14,} {op:<18} {s!r}")



def scan_find(path, needles, chunk=8 << 20, cap=40):
    """Stream the whole file, recording offsets of raw byte needles.

    Sequential read, bounded memory (one chunk + a small overlap). Used to
    locate where the axes actually live when neither head nor tail finds
    them -- e.g. pandas 2.x writes block data first, so the columns index
    can sit deep in the middle of a multi-GB stream.
    """
    import time
    hits = {n: [] for n in needles}
    t0, off, prev, overlap = time.time(), 0, b"", 64
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            window = prev + buf
            base = off - len(prev)
            for n in needles:
                start = 0
                while True:
                    i = window.find(n, start)
                    if i == -1:
                        break
                    if len(hits[n]) < cap:
                        hits[n].append(base + i)
                    start = i + 1
            off += len(buf)
            prev = window[-overlap:]
    print(f"scanned {off:,} bytes in {time.time() - t0:.1f}s\n")
    for n in needles:
        h = hits[n]
        print(f"  {n.decode():<26} {len(h):>3} hit(s)"
              f"{' (capped)' if len(h) == cap else ''}: {h[:12]}")


def scan_window(path, start, length):
    """Dump printable ASCII runs from an arbitrary byte window.

    Needed because pyarrow-backed string columns (ArrowStringArray /
    large_string) pack their values into one contiguous UTF-8 buffer rather
    than individual pickle string opcodes -- including, in this corpus, the
    columns index itself. So the column names are only visible as a raw
    ASCII run, not as SHORT_BINUNICODE records.
    """
    with open(path, "rb") as f:
        f.seek(start)
        buf = f.read(length)
    runs = re.findall(rb"[ -~]{3,}", buf)
    print(f"window @{start:,} +{length:,} -> {len(runs)} printable run(s)")
    for r in runs:
        print(f"  {r.decode()}")


def main():
    if len(sys.argv) < 2:
        sys.exit("Usage: python peek_pickle_schema.py <path.pkl> "
                 "[head|tail|find|window] [bytes | start length]")
    path = sys.argv[1]
    mode = sys.argv[2] if len(sys.argv) >= 3 else "tail"

    if mode == "head":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 4 * 1024 * 1024
        scan_head(path, n)
    elif mode == "tail":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 4 * 1024 * 1024
        scan_tail(path, n)
    elif mode == "find":
        needles = [a.encode() for a in sys.argv[3:]] or [
            b"pandas.core.indexes", b"_new_Index", b"RangeIndex",
            b"vader", b"conversation_id", b"word_count", b"text_length",
        ]
        scan_find(path, needles)
    elif mode == "window":
        if len(sys.argv) != 5:
            sys.exit("window mode: <path.pkl> window <start> <length>")
        scan_window(path, int(sys.argv[3]), int(sys.argv[4]))
    else:
        sys.exit("mode must be head, tail, find, or window")


if __name__ == "__main__":
    main()

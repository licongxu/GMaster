"""Disk-resident transform tables, so a table that cannot be *built* in a serving
process can still be used by one.

The Nside 1024 polarised layout is the case this exists for.  It is 73.48 GiB in
float32, which a widened pool holds comfortably, but building it in-process has never
worked: the per-chunk pieces and each window's join output are alive at the same time,
so the peak lands ~13 GiB above the layout and every pool tried refuses one 2.69 GiB
window block (``.qwen/tmp/score_n1024_spin2_halfmarch.log``: peak 86.8 of a 90.2 GiB
pool, ``largest_free_block_bytes: 0``, and the pipeline then runs s2fft's generic
scatter loop for 87 s -- 0.04x).

Splitting the two jobs removes the conflict:

* a **builder** process walks the march one chunk at a time, so its peak is one chunk
  table plus one finished window block (~3 GiB), and writes the finished blocks to a
  file;
* a **serving** process fills the final buffers with one host->device copy per block.
  There is no join, no transposed transient and no cuBLAS workspace, so the resident set
  is the layout and nothing else.

Loading is opt-in through :func:`gmaster.set_table_store`; nothing is read or written
until a directory is configured.  A file is identified by geometry *and* by a revision
hash of the recurrence source that produced it, so a change to ``_build`` invalidates
every cached table rather than silently serving stale values.
"""
import hashlib
import inspect
import json
import os

import jax.numpy as jnp
import numpy as np

_STORE_DIR = None

_FORMAT = "gmaster-table-v1"


def configure(path):
    """Set (or clear, with ``None``) the directory holding cached tables."""
    global _STORE_DIR
    if path is not None:
        path = os.path.abspath(os.path.expanduser(path))
        os.makedirs(path, exist_ok=True)
    _STORE_DIR = path


def path():
    return _STORE_DIR


def enabled():
    return _STORE_DIR is not None


def revision():
    """Hash of the source that generates the values, so a stale file never loads."""
    from . import _spin_slice

    src = "".join(inspect.getsource(f) for f in
                  (_spin_slice._windows, _spin_slice._march, _spin_slice._build))
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _files(nside, L, spin, layout, store):
    import jax

    name = f"n{nside}-L{L}-spin{spin}-{layout}-{jnp.dtype(store).name}-{revision()}"
    base = os.path.join(_STORE_DIR, name)
    return base + ".bin", base + ".json"


def save(blocks, layout, *, nside, L, spin, store, theta_rows):
    """Write one layout's blocks (window order) plus the header describing them."""
    bin_path, json_path = _files(nside, L, spin, layout, store)
    shapes = [tuple(np.asarray(b).shape) for b in blocks]
    tmp = bin_path + ".tmp"
    with open(tmp, "wb") as fh:
        for block in blocks:
            host = np.asarray(block)
            if host.dtype != np.asarray(jnp.empty((), dtype=store)).dtype:
                raise ValueError(f"block dtype {host.dtype} != store {store}")
            fh.write(memoryview(np.ascontiguousarray(host)).cast("B"))
    os.replace(tmp, bin_path)
    with open(json_path, "w") as fh:
        json.dump({"format": _FORMAT, "revision": revision(), "layout": layout,
                   "nside": nside, "L": L, "spin": spin, "store": str(store),
                   "ntheta": theta_rows, "shapes": shapes}, fh)
    return bin_path


def load(layout, *, nside, L, spin, store):
    """``(blocks, ntheta)`` from the store, or ``None`` when there is no usable file.

    Each block is read with one contiguous ``fromfile`` and moved to the device with one
    copy, so host scratch is one block and the device peak is the layout itself.
    """
    bin_path, json_path = _files(nside, L, spin, layout, store)
    if not (os.path.exists(bin_path) and os.path.exists(json_path)):
        return None
    with open(json_path) as fh:
        head = json.load(fh)
    if head.get("format") != _FORMAT or head.get("revision") != revision():
        return None
    itemsize = jnp.dtype(store).itemsize
    blocks = []
    offset = 0
    for shape in head["shapes"]:
        count = int(np.prod(shape))
        host = np.fromfile(bin_path, dtype=np.dtype(jnp.dtype(store).name),
                           count=count, offset=offset)
        if host.size != count:
            raise OSError(f"{bin_path} is truncated: wanted {count} items at "
                          f"offset {offset}, read {host.size}")
        offset += count * itemsize
        blocks.append(jnp.asarray(host.reshape(shape)))
    del host
    return blocks, head["ntheta"]


def describing():
    """Human-readable listing of what is stored, for ``build_table_store`` output."""
    if _STORE_DIR is None:
        return []
    out = []
    for name in sorted(os.listdir(_STORE_DIR)):
        if not name.endswith(".json"):
            continue
        with open(os.path.join(_STORE_DIR, name)) as fh:
            head = json.load(fh)
        out.append((name, head.get("revision"),
                    os.path.getsize(os.path.join(_STORE_DIR, name[:-5] + ".bin"))))
    return out

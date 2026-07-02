#!/usr/bin/env python3
"""Split a combined openPangu-2.0 (openpangu-v2) GGUF into:

  - a base GGUF without the MTP / NextN layers (smaller; use standalone), and
  - an MTP-only GGUF holding just the NextN head layers plus the shared
    embedding / output tensors, usable for self-speculative decoding via
    `--model-draft <mtp.gguf> --mtp`.

The MTP GGUF renumbers the NextN layers to blk.0.. and carries
block_count == nextn_predict_layers, so hparams.n_layer() == 0 and every
layer is an MTP layer. Its per-layer SWA window is set to the reference
value (sliding_window_list[-1], 2048 for openPangu-2.0-Flash) -- the split
model runs the MTP head with its trained sliding window, which the combined
GGUF cannot express next to the trunk's window.

Example:
  python conversion/split_pangu_mtp.py openPangu-2.0-Flash-BF16.gguf \
      --base openPangu-2.0-Flash-base-BF16.gguf \
      --mtp  openPangu-2.0-Flash-mtp-BF16.gguf
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

if 'NO_LOCAL_GGUF' not in __import__('os').environ:
    sys.path.insert(1, str(Path(__file__).parent.parent / 'gguf-py'))

import gguf
from gguf import GGUFReader, GGUFWriter, GGUFValueType  # noqa: E402

logger = logging.getLogger("split-pangu-mtp")

# non-blk tensors the MTP head graph reads (shared head/embedding fallbacks)
MTP_SHARED_TENSORS = {"token_embd.weight", "output.weight", "output_norm.weight"}

BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


def copy_kv(reader: GGUFReader, writer: GGUFWriter, overrides: dict, remove: set) -> None:
    for field in reader.fields.values():
        if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith('GGUF.'):
            continue
        if field.name in remove:
            logger.debug('removing %s', field.name)
            continue

        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == GGUFValueType.ARRAY else None

        value = field.contents()
        if field.name in overrides:
            value = overrides.pop(field.name)
            logger.info('overriding %s -> %s', field.name, str(value)[:80])

        writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

    for key, (value, val_type) in overrides.items():
        logger.info('adding %s = %s', key, str(value)[:80])
        writer.add_key_value(key, value, val_type)


def write_split(reader: GGUFReader, path: Path, arch: str, overrides: dict, remove: set,
                keep_tensor) -> None:
    writer = GGUFWriter(path, arch)
    copy_kv(reader, writer, dict(overrides), remove)

    tensors = []
    for tensor in reader.tensors:
        new_name = keep_tensor(tensor.name)
        if new_name is None:
            continue
        tensors.append((new_name, tensor))
        writer.add_tensor_info(new_name, tensor.data.shape, tensor.data.dtype, tensor.data.nbytes,
                               tensor.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_ti_data_to_file()

    for i, (new_name, tensor) in enumerate(tensors):
        logger.info('[%d/%d] writing %s (%s)', i + 1, len(tensors), new_name, path.name)
        writer.write_tensor_data(tensor.data, tensor_endianess=reader.endianess)

    writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="combined openpangu-v2 GGUF")
    parser.add_argument("--base", type=Path, help="output path for the base (no-MTP) GGUF")
    parser.add_argument("--mtp", type=Path, help="output path for the MTP-only GGUF")
    parser.add_argument("--mtp-swa-window", type=int, default=2048,
                        help="SWA window for the MTP layers (reference: sliding_window_list[-1], default 2048)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO)

    if not args.base and not args.mtp:
        parser.error("nothing to do: pass --base and/or --mtp")

    reader = GGUFReader(args.input)

    arch = reader.get_field(gguf.Keys.General.ARCHITECTURE).contents()
    if arch != "openpangu-v2":
        parser.error(f"expected an openpangu-v2 GGUF, got architecture '{arch}'")

    n_layer_all = reader.get_field(f"{arch}.block_count").contents()
    field_nextn = reader.get_field(f"{arch}.nextn_predict_layers")
    n_nextn     = field_nextn.contents() if field_nextn else 0
    if n_nextn <= 0:
        parser.error("this GGUF has no MTP / NextN layers (nextn_predict_layers missing or 0)")
    n_base = n_layer_all - n_nextn

    swa_field = reader.get_field(f"{arch}.attention.layer_swa_window")
    swa_all   = list(swa_field.contents()) if swa_field else [0] * n_layer_all

    logger.info("input: %d layers = %d base + %d MTP", n_layer_all, n_base, n_nextn)

    if args.base:
        def keep_base(name: str):
            m = BLK_RE.match(name)
            if m and int(m.group(1)) >= n_base:
                return None
            return name

        write_split(reader, args.base, arch,
                    overrides={
                        f"{arch}.block_count": n_base,
                        f"{arch}.attention.layer_swa_window": swa_all[:n_base],
                    },
                    remove={f"{arch}.nextn_predict_layers"},
                    keep_tensor=keep_base)
        logger.info("base GGUF written to %s", args.base)

    if args.mtp:
        def keep_mtp(name: str):
            m = BLK_RE.match(name)
            if m:
                il = int(m.group(1))
                if il < n_base:
                    return None
                return f"blk.{il - n_base}.{m.group(2)}"
            if name in MTP_SHARED_TENSORS:
                return name
            return None

        write_split(reader, args.mtp, arch,
                    overrides={
                        f"{arch}.block_count": n_nextn,
                        f"{arch}.nextn_predict_layers": n_nextn,
                        f"{arch}.leading_dense_block_count": 0,
                        f"{arch}.attention.layer_swa_window": [args.mtp_swa_window] * n_nextn,
                    },
                    remove=set(),
                    keep_tensor=keep_mtp)
        logger.info("MTP GGUF written to %s", args.mtp)


if __name__ == '__main__':
    main()

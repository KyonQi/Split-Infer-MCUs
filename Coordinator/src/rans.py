"""
rans_decoder.py — rANS byte encoder/decoder matching the Teensy C++ codec.
Uses a compiled C shared library (librans_codec.so) for speed,
with an automatic pure-Python fallback for decoding.

Build the C library once:
    gcc -O3 -shared -fPIC -o src/librans_codec.so src/rans_codec.c
"""

import ctypes
import logging
import struct
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RANS_MAGIC  = 0x72414E53
PROB_BITS   = 12
PROB_SCALE  = 1 << PROB_BITS   # 4096
RANS_L      = 1 << 23
NUM_SYMS    = 256
HEADER_FMT  = f'<III{NUM_SYMS}H'           # magic, orig_size, comp_size, freq[256]
HEADER_SIZE = struct.calcsize(HEADER_FMT)   # 524 bytes

# ── Try to load compiled C codec ────────────────────────────────────
_c_decompress = None
_c_compress   = None

def _find_and_load_lib():
    """Search for librans_codec.so (new combined lib) or librans_decode.so (old)."""
    global _c_decompress, _c_compress

    candidates_codec = [
        Path(__file__).parent / "librans_codec.so",
        Path.cwd() / "src" / "librans_codec.so",
        Path.cwd() / "librans_codec.so",
    ]
    # Try the new combined library first
    for p in candidates_codec:
        if p.is_file():
            try:
                lib = ctypes.CDLL(str(p))
                # Decoder
                fn_dec = lib.rans_decompress_c
                fn_dec.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),  # src
                    ctypes.c_uint32,                  # src_len
                    ctypes.POINTER(ctypes.c_uint8),   # dst
                    ctypes.c_uint32,                  # dst_cap
                ]
                fn_dec.restype = ctypes.c_uint32
                _c_decompress = fn_dec

                # Encoder
                fn_enc = lib.rans_compress_c
                fn_enc.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),  # src
                    ctypes.c_uint32,                  # src_len
                    ctypes.POINTER(ctypes.c_uint8),   # dst
                    ctypes.c_uint32,                  # dst_cap
                ]
                fn_enc.restype = ctypes.c_uint32
                _c_compress = fn_enc

                logger.info(f"[rANS]: Loaded C codec (enc+dec) from {p}")
                return
            except OSError as e:
                logger.warning(f"[rANS]: Failed to load {p}: {e}")

    # Fallback: try old decode-only library
    candidates_dec = [
        Path(__file__).parent / "librans_decode.so",
        Path.cwd() / "src" / "librans_decode.so",
        Path.cwd() / "librans_decode.so",
    ]
    for p in candidates_dec:
        if p.is_file():
            try:
                lib = ctypes.CDLL(str(p))
                fn = lib.rans_decompress_c
                fn.argtypes = [
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_uint32,
                    ctypes.POINTER(ctypes.c_uint8),
                    ctypes.c_uint32,
                ]
                fn.restype = ctypes.c_uint32
                _c_decompress = fn
                logger.info(f"[rANS]: Loaded C decoder (decode-only) from {p}")
                return
            except OSError as e:
                logger.warning(f"[rANS]: Failed to load {p}: {e}")

_find_and_load_lib()


# ── Public API ──────────────────────────────────────────────────────

def is_rans_compressed(data: bytes) -> bool:
    """Check if data starts with the rANS magic number."""
    if len(data) < 4:
        return False
    magic, = struct.unpack('<I', data[:4])
    return magic == RANS_MAGIC


def rans_compress(data: bytes) -> bytes:
    """
    Compress raw bytes using rANS, producing a stream decodable by the Teensy.
    Requires the C encoder (librans_codec.so).
    Returns the compressed bytes, or the original data if compression fails/expands.
    """
    if _c_compress is None:
        logger.warning("[rANS]: C encoder unavailable, sending uncompressed")
        return data
    return _rans_compress_c(data)


def rans_decompress(data: bytes) -> bytes:
    """
    Decompress a rANS-compressed stream produced by the Teensy encoder.
    Uses C fast-path if available, otherwise falls back to pure Python.
    """
    if _c_decompress is not None:
        return _rans_decompress_c(data)
    logger.warning("[rANS]: C decoder unavailable, using slow Python fallback")
    return _rans_decompress_py(data)


# ── C encoder ───────────────────────────────────────────────────────

def _rans_compress_c(data: bytes) -> bytes:
    src_len = len(data)
    # Worst case: header + data (compression may expand slightly)
    dst_cap = src_len + HEADER_SIZE + 1024
    src_buf = (ctypes.c_uint8 * src_len).from_buffer_copy(data)
    dst_buf = (ctypes.c_uint8 * dst_cap)()

    ret = _c_compress(src_buf, src_len, dst_buf, dst_cap)
    if ret == 0:
        logger.warning("[rANS]: C compression failed, sending uncompressed")
        return data
    
    # If compressed is larger than original, skip compression
    if ret >= src_len:
        logger.debug(f"[rANS]: Compression expanded data ({ret} >= {src_len}), sending uncompressed")
        return data

    return bytes(dst_buf[:ret])


# ── C fast-path ─────────────────────────────────────────────────────

def _rans_decompress_c(data: bytes) -> bytes:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Data too short for rANS header ({len(data)} < {HEADER_SIZE})")

    # peek original_size from header (offset 4, little-endian uint32)
    orig_size = struct.unpack_from('<I', data, 4)[0]

    src_buf = (ctypes.c_uint8 * len(data)).from_buffer_copy(data)
    dst_buf = (ctypes.c_uint8 * orig_size)()

    ret = _c_decompress(src_buf, len(data), dst_buf, orig_size)
    if ret == 0:
        raise RuntimeError("C rANS decompression failed")

    return bytes(dst_buf)


# ── Pure-Python fallback ────────────────────────────────────────────

def _rans_decompress_py(data: bytes) -> bytes:
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Data too short for rANS header ({len(data)} < {HEADER_SIZE})")

    fields = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
    magic       = fields[0]
    orig_size   = fields[1]
    comp_size   = fields[2]
    freq        = list(fields[3:])

    if magic != RANS_MAGIC:
        raise ValueError(f"Invalid rANS magic: 0x{magic:08X}")

    cum = [0] * (NUM_SYMS + 1)
    for i in range(NUM_SYMS):
        cum[i + 1] = cum[i] + freq[i]

    lut = [0] * PROB_SCALE
    for s in range(NUM_SYMS):
        for j in range(cum[s], cum[s + 1]):
            lut[j] = s

    payload = data[HEADER_SIZE : HEADER_SIZE + comp_size]
    if len(payload) < 4:
        raise ValueError("Compressed payload too short")

    state = (payload[0] << 24) | (payload[1] << 16) | (payload[2] << 8) | payload[3]
    p = 4
    plen = len(payload)

    output = bytearray(orig_size)
    mask = PROB_SCALE - 1

    for i in range(orig_size):
        c_val = state & mask
        sym   = lut[c_val]
        f     = freq[sym]
        c     = cum[sym]

        state = f * (state >> PROB_BITS) + c_val - c

        while state < RANS_L and p < plen:
            state = (state << 8) | payload[p]
            p += 1

        output[i] = sym

    return bytes(output)
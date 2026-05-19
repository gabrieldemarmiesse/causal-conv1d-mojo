"""Take the metallib extracted from the Mojo binary, load it onto
torch's own `MTLDevice` via Obj-C, build a compute pipeline state,
and dispatch the no-op kernel on torch's `MTLCommandQueue`.

If the per-call cost via torch's queue is dramatically lower than
Mojo's `enqueue_function`, that confirms the overhead is in Mojo's
AsyncRT command-buffer machinery — not in the underlying Metal API.
"""

import ctypes
import ctypes.util
import struct
import subprocess
import time
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parent
SRC = REPO / "modular_bug_repro.mojo"
BIN = Path("/tmp/repro_bin")
METALLIB = Path("/tmp/extracted.metallib")

# --- 1. Build the Mojo binary --------------------------------------------------
print("Building mojo binary...")
subprocess.run(
    ["uv", "run", "mojo", "build", str(SRC), "-o", str(BIN)],
    check=True,
    cwd=REPO,
    stdout=subprocess.DEVNULL,
)
print(f"  -> {BIN} ({BIN.stat().st_size} bytes)")

# --- 2. Extract the embedded metallib ----------------------------------------
data = BIN.read_bytes()
start = data.find(b"MTLB")
assert start >= 0, "no MTLB magic found"
size = struct.unpack_from("<Q", data, start + 0x10)[0]
METALLIB.write_bytes(data[start : start + size])
print(f"extracted metallib: {METALLIB} ({size} bytes)")

# Read the kernel symbol from the metallib.
sym = subprocess.run(
    ["xcrun", "--sdk", "macosx", "metal-nm", str(METALLIB)],
    capture_output=True,
    text=True,
).stdout.strip().split()[-1]
print(f"kernel symbol: {sym}")

# --- 3. Obj-C plumbing --------------------------------------------------------
libobjc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
libobjc.sel_registerName.restype = ctypes.c_void_p
libobjc.sel_registerName.argtypes = [ctypes.c_char_p]
libobjc.objc_getClass.restype = ctypes.c_void_p
libobjc.objc_getClass.argtypes = [ctypes.c_char_p]


def msg(restype, sender, sel_name, argtypes=(), args=()):
    # Re-bind argtypes per call — objc_msgSend is one C function pointer
    # shared by every selector signature, so persistent argtypes from a
    # previous call would mismatch the next one and crash.
    fn = libobjc.objc_msgSend
    fn.restype = restype
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, *argtypes]
    return fn(sender, libobjc.sel_registerName(sel_name.encode()), *args)


# --- 4. Grab torch's MTLDevice + MTLCommandQueue ------------------------------
_ = torch.zeros(1, device="mps")
torch.mps.synchronize()

torch_lib = ctypes.cdll.LoadLibrary(
    f"{torch.__file__[:-12]}/lib/libtorch_cpu.dylib"
)
torch_lib._ZN2at3mps19getCurrentMPSStreamEv.restype = ctypes.c_void_p
torch_lib._ZN2at3mps19getCurrentMPSStreamEv.argtypes = []
stream_ptr = torch_lib._ZN2at3mps19getCurrentMPSStreamEv()

# MPSStream layout: c10::Stream (16 bytes) + MTLCommandQueue_t (8 bytes).
mtl_queue = ctypes.c_void_p.from_address(stream_ptr + 16).value
mtl_device = msg(ctypes.c_void_p, mtl_queue, "device")
print(f"torch MTLDevice: 0x{mtl_device:x}")
print(f"torch MTLCommandQueue: 0x{mtl_queue:x}")

# --- 5. Load the metallib + build a pipeline state ----------------------------
ns_str_cls = libobjc.objc_getClass(b"NSString")
ns_url_cls = libobjc.objc_getClass(b"NSURL")

ns_path = msg(
    ctypes.c_void_p, ns_str_cls, "stringWithUTF8String:",
    argtypes=(ctypes.c_char_p,), args=(str(METALLIB).encode(),),
)
url = msg(
    ctypes.c_void_p, ns_url_cls, "fileURLWithPath:",
    argtypes=(ctypes.c_void_p,), args=(ns_path,),
)
err = ctypes.c_void_p(0)
library = msg(
    ctypes.c_void_p, mtl_device, "newLibraryWithURL:error:",
    argtypes=(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
    args=(url, ctypes.byref(err)),
)
assert library, "newLibraryWithURL failed"

fn_name_ns = msg(
    ctypes.c_void_p, ns_str_cls, "stringWithUTF8String:",
    argtypes=(ctypes.c_char_p,), args=(sym.encode(),),
)
mtl_function = msg(
    ctypes.c_void_p, library, "newFunctionWithName:",
    argtypes=(ctypes.c_void_p,), args=(fn_name_ns,),
)
assert mtl_function, f"newFunctionWithName failed for {sym!r}"
err = ctypes.c_void_p(0)
pso = msg(
    ctypes.c_void_p, mtl_device,
    "newComputePipelineStateWithFunction:error:",
    argtypes=(ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)),
    args=(mtl_function, ctypes.byref(err)),
)
assert pso, "newComputePipelineStateWithFunction failed"
print(f"MTLComputePipelineState: 0x{pso:x}")


# --- 6. Encode + commit one dispatch on torch's queue -------------------------
class MTLSize(ctypes.Structure):
    _fields_ = [("w", ctypes.c_ulong), ("h", ctypes.c_ulong), ("d", ctypes.c_ulong)]


SEL_CMD_BUF = libobjc.sel_registerName(b"commandBuffer")
SEL_ENCODER = libobjc.sel_registerName(b"computeCommandEncoder")
SEL_SET_PSO = libobjc.sel_registerName(b"setComputePipelineState:")
SEL_SET_BYTES = libobjc.sel_registerName(b"setBytes:length:atIndex:")
SEL_DISPATCH = libobjc.sel_registerName(
    b"dispatchThreadgroups:threadsPerThreadgroup:"
)
SEL_END = libobjc.sel_registerName(b"endEncoding")
SEL_COMMIT = libobjc.sel_registerName(b"commit")

# Single i64 dummy arg for the kernel — encoded once and reused via setBytes
# per call.
_dummy = (ctypes.c_int64 * 1)(0)
DUMMY_PTR = ctypes.addressof(_dummy)


def launch_noop():
    """Encode + commit one noop dispatch on torch's command queue."""
    send = libobjc.objc_msgSend
    send.restype = ctypes.c_void_p
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cmd_buf = send(mtl_queue, SEL_CMD_BUF)
    enc = send(cmd_buf, SEL_ENCODER)

    send.restype = None
    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p]
    send(enc, SEL_SET_PSO, pso)

    send.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_ulong, ctypes.c_ulong,
    ]
    send(enc, SEL_SET_BYTES, DUMMY_PTR, 8, 0)

    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p, MTLSize, MTLSize]
    send(enc, SEL_DISPATCH, MTLSize(1, 1, 1), MTLSize(1, 1, 1))

    send.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    send(enc, SEL_END)
    send(cmd_buf, SEL_COMMIT)


# Warmup + timed loop, single synchronize at the end (same as the Mojo repro).
for _ in range(500):
    launch_noop()
torch.mps.synchronize()

ITERS = 5000
t0 = time.perf_counter_ns()
for _ in range(ITERS):
    launch_noop()
torch.mps.synchronize()
us = (time.perf_counter_ns() - t0) / 1000.0 / ITERS
print()
print(f"Dispatch on torch's MTLCommandQueue (extracted metallib): {us:7.2f} μs/call")

import os
import shutil
Import("env")

build_flags = env.get("BUILD_FLAGS", [])
worker_id = None
for flag in build_flags:
    if isinstance(flag, str) and flag.startswith("-DWORKER_ID="):
        worker_id = flag.split("=")[1]
        break

if worker_id is None:
    print("WARNING: WORKER_ID not set in build flags, defaulting to 0")
    worker_id = 0

# Path to exported per-MCU headers from Python_Sim_Infer
# Adjust the path to your own workspace structure if necessary
HEADERS_SRC = os.path.abspath(
    os.path.join(env.get("PROJECT_DIR", "."),
                 "..", "..", "Python_Sim_Infer", "extractor_mcu", f"mcu_{worker_id}")
)
HEADERS_DST = os.path.join(env.get("PROJECT_DIR", "."), "include")
HEADER_FILES = ["weights.h", "quant_params.h", "layer_config.h"]

print(f"=== Prebuild worker {worker_id} ===")
print(f"Copying headers from {HEADERS_SRC} to {HEADERS_DST}...")

for hf in HEADER_FILES:
    src = os.path.join(HEADERS_SRC, hf)
    dst = os.path.join(HEADERS_DST, hf)
    if os.path.exists(src):
        shutil.copy2(src, dst)
    else:
        print(f"Warning: {src} does not exist and will be skipped.")

print("Prebuild step completed.")
# rnnoise.dll build (AVX2, with batch wrapper)

The root `rnnoise.dll` shipped with MeetRec is **not** stock — it is rnnoise built
with MSVC `/arch:AVX2` (SIMD, ~2.9x faster than scalar) plus one custom exported
function, `rnnoise_process_buffer`, that denoise.py uses for GIL-releasing
multithreaded processing.

## Source
- Upstream: https://github.com/xiph/rnnoise.git
- Pinned commit: `70f1d256acd4b34a572f999a05c87bf00b67730d`

## Reproduce
1. `git clone https://github.com/xiph/rnnoise.git .deps/rnnoise`
2. `cd .deps/rnnoise && git checkout 70f1d256acd4b34a572f999a05c87bf00b67730d`
3. Copy `build/rnnoise/rnnoise_buffer.c` into `.deps/rnnoise/src/`
4. Run `build/rnnoise/build_rnnoise_dll.bat` (needs VS2019 BuildTools w/ C++; edit
   the hard-coded `vcvars64.bat` path inside if yours differs)
5. It emits `rnnoise.dll` next to the script; copy it to the repo root, replacing
   the existing one. Keep a scalar backup if desired.

## Notes
- `/MT` statically links the CRT so the dll has no VC-runtime dependency.
- No x86 RTCD / SIMD source files are needed; global `/arch:AVX2` is enough for
  `vec.h` to pick the `vec_avx.h` path (the "no vectorization" warning disappears).
- `rnnoise_process_buffer` just loops `rnnoise_process_frame` in C; output is
  bitwise identical to calling per-frame from Python.

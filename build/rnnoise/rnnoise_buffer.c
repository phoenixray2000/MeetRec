/* Batch wrapper around rnnoise_process_frame so the per-frame loop runs in C.
   The single long C call releases the Python GIL for its whole duration, which
   is what lets denoise.py run segments in parallel threads. Kept in a separate
   file so the upstream rnnoise sources stay untouched. */
#include <stddef.h>
#include "rnnoise.h"

RNNOISE_EXPORT void rnnoise_process_buffer(DenoiseState *st, int nframes,
                                           float *out, const float *in) {
    int fs = rnnoise_get_frame_size();
    int i;
    for (i = 0; i < nframes; i++) {
        size_t off = (size_t)i * (size_t)fs;
        rnnoise_process_frame(st, out + off, in + off);
    }
}

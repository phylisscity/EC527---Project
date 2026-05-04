/*
 * EC527 Final Project: High-Performance Sorting
 * Team: Cynthia Young & Phyliss Darko
 *
 * GPU scaffolding inspired by Lab 7 (CUDA MMM) starter code from EC527 (Prof. Herbordt, BU)
 *
 * radix_sort_gpu.cu -- CUDA implementation of LSD radix sort
 *
 * Compile: nvcc -arch=compute_86 -code=sm_86 radix_sort_gpu.cu -o radix_sort_gpu
 * (A40 GPU on SCC is an example. adjusted based on chosen gpu)
 *
 * Branches from radix_sort_cpu.c. Input generation, validation, array sizes,
 * and output format are identical so CPU and GPU results can be compared directly.
 *
 * Three timing metrics are reported: per-kernel times for count, scan, and scatter
 * (CUDA events, excludes transfers), their sum as total kernel time, and end-to-end
 * time (host clock wrapping h2d + sort + d2h).
 *
 * The three GPU kernels are stubbed out below - implement them one at a time:
 *   1. count_kernel   - histogram per pass
 *   2. scan_kernel    - parallel prefix sum over 256 counts
 *   3. scatter_kernel - place each element at its output position
 *
 * The CPU reference sort is kept here for sanity checking during development.
 *
 * Output columns:
 *   dist, size, t_kernel_sec, t_e2e_sec, kernel_GB_per_sec, valid
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// ---------- configuration ----------

#define RADIX_BITS        8     // bits per pass
#define RADIX             256   // 2^RADIX_BITS buckets
#define PASSES            4     // 32-bit int / 8 bits per pass
#define THREADS_PER_BLOCK 1024  // kernel launch block size
#define WARP_SIZE         32    // CUDA warp size


#define NUM_SIZES   9
static long int test_sizes[NUM_SIZES] = {
    1000000,     //   1M
    2000000,     //   2M
    4000000,     //   4M
    8000000,     //   8M
    16000000,    //  16M
    32000000,    //  32M
    64000000,    //  64M
    128000000,   //  128M
    256000000    //  256M
};

#define NUM_DIST 3
static const char *dist_names[NUM_DIST] = { "random", "sorted", "reverse" };


// ---------- timing ----------
double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec  = end.tv_sec  - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0) {
        temp.tv_sec  -= 1;
        temp.tv_nsec += 1000000000;
    }
    return ((double)temp.tv_sec) + ((double)temp.tv_nsec) * 1.0e-9;
}

// forces CPU out of power-saving mode before timing starts - identical to lab files
double wakeup_delay()
{
    double meas = 0; int i, j;
    struct timespec time_start, time_stop;
    double quasi_random = 0;
    clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
    j = 100;
    while (meas < 1.0) {
        for (i = 1; i < j; i++)
            quasi_random = quasi_random * quasi_random - 1.923432;
        clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
        meas = interval(time_start, time_stop);
        j *= 2;
    }
    return quasi_random;
}

// ---------- input generation (identical to CPU version) ----------

// fill array with random 32-bit unsigned integers
void gen_random(unsigned int *arr, long int n)
{
    for (long int i = 0; i < n; i++)
        // combine two rand() calls to cover full 32-bit range
        arr[i] = ((unsigned int)rand() << 16) ^ (unsigned int)rand();
}

// fill array already sorted ascending
void gen_sorted(unsigned int *arr, long int n)
{
    for (long int i = 0; i < n; i++)
        arr[i] = (unsigned int)i;
}

// fill array sorted descending
void gen_reverse(unsigned int *arr, long int n)
{
    for (long int i = 0; i < n; i++)
        arr[i] = (unsigned int)(n - 1 - i);
}

// ---------- validation (identical to CPU version) ----------

// comparison function for qsort (unsigned int ascending)
int cmp_uint(const void *a, const void *b)
{
    unsigned int ua = *(unsigned int *)a;
    unsigned int ub = *(unsigned int *)b;
    return (ua > ub) - (ua < ub);
}

// compare our sorted output against qsort on same input
// returns 1 if correct, 0 if mismatch found
int validate(unsigned int *result, unsigned int *ref, long int n)
{
    qsort(ref, n, sizeof(unsigned int), cmp_uint);
    for (long int i = 0; i < n; i++) {
        if (result[i] != ref[i]) {
            printf("  VALIDATION FAILED at index %ld: got %u expected %u\n",
                   i, result[i], ref[i]);
            return 0;
        }
    }
    return 1;
}

// ---------- CPU reference - keep for sanity checking during GPU development ----------

void radix_sort_pass_cpu(unsigned int *in, unsigned int *out, long int n, int shift)
{
    long int count[RADIX];
    long int prefix[RADIX];

    memset(count, 0, sizeof(count));
    for (long int i = 0; i < n; i++)
        count[(in[i] >> shift) & 0xFF]++;

    prefix[0] = 0;
    for (int b = 1; b < RADIX; b++)
        prefix[b] = prefix[b-1] + count[b-1];

    for (long int i = 0; i < n; i++) {
        unsigned int bucket = (in[i] >> shift) & 0xFF;
        out[prefix[bucket]++] = in[i];
    }
}

void radix_sort_cpu(unsigned int *arr, unsigned int *scratch, long int n)
{
    radix_sort_pass_cpu(arr,     scratch, n,  0);
    radix_sort_pass_cpu(scratch, arr,     n,  8);
    radix_sort_pass_cpu(arr,     scratch, n, 16);
    radix_sort_pass_cpu(scratch, arr,     n, 24);
}


// ---------- GPU kernels ----------

/*
 * count_kernel: build a histogram of bucket indices for this pass.
 *
 * Each warp accumulates into its own shared-memory histogram to avoid
 * global-memory atomic contention, then one thread per bucket reduces
 * across warps and writes the final per-block count to global memory.
 */
__global__ void count_kernel(unsigned int *in, long int n,
                              int shift, unsigned int *d_block_count)
{
    const int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    int wid = threadIdx.x / WARP_SIZE;

    __shared__ unsigned int s_hist[NUM_WARPS * RADIX];

    for (int i = threadIdx.x; i < NUM_WARPS * RADIX; i += THREADS_PER_BLOCK)
        s_hist[i] = 0;
    __syncthreads();

    long int tid = (long int)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        unsigned int bucket = (in[tid] >> shift) & 0xFF;
        atomicAdd(&s_hist[wid * RADIX + bucket], 1u);
    }
    __syncthreads();

    // first RADIX threads each own one bucket; sum across warps
    if (threadIdx.x < RADIX) {
        unsigned int total = 0;
        for (int w = 0; w < NUM_WARPS; w++)
            total += s_hist[w * RADIX + threadIdx.x];
        d_block_count[(long int)blockIdx.x * RADIX + threadIdx.x] = total;
    }
}

/*
 * scan_kernel: three steps in one kernel, one block of RADIX threads.
 *
 * Step 1 — reduce: sum d_block_count[:,bucket] → global count per bucket.
 * Step 2 — global prefix: Blelloch exclusive scan over global counts → d_prefix[bucket],
 *           the output start position for each bucket.
 * Step 3 — block offsets: exclusive prefix down d_block_count[:,bucket] →
 *           d_block_offset[block*RADIX+bucket], so each block knows where in
 *           its bucket's output range its own elements belong.
 */
__global__ void scan_kernel(unsigned int *d_block_count, long int *d_prefix,
                             long int *d_block_offset, int num_blocks)
{
    int bucket = threadIdx.x;  // 0..255

    // Step 1: global count for this bucket
    long int global_count = 0;
    for (int b = 0; b < num_blocks; b++)
        global_count += (long int)d_block_count[(long int)b * RADIX + bucket];

    // Step 2: Blelloch exclusive prefix over the 256 global counts
    __shared__ long int s[RADIX];
    s[bucket] = global_count;
    __syncthreads();

    for (int stride = 1; stride < RADIX; stride <<= 1) {
        int i = (bucket + 1) * (stride << 1) - 1;
        if (i < RADIX) s[i] += s[i - stride];
        __syncthreads();
    }
    if (bucket == 0) s[RADIX - 1] = 0;
    __syncthreads();
    for (int stride = RADIX >> 1; stride >= 1; stride >>= 1) {
        int i = (bucket + 1) * (stride << 1) - 1;
        if (i < RADIX) {
            long int tmp  = s[i - stride];
            s[i - stride] = s[i];
            s[i]          = tmp + s[i];
        }
        __syncthreads();
    }
    d_prefix[bucket] = s[bucket];

    // Step 3: exclusive prefix down this bucket's column of d_block_count
    long int running = 0;
    for (int b = 0; b < num_blocks; b++) {
        d_block_offset[(long int)b * RADIX + bucket] = running;
        running += (long int)d_block_count[(long int)b * RADIX + bucket];
    }
}

/*
 * scatter_kernel (v1.3): two 4-bit warp-ballot passes sort the tile stably in
 * shared memory.  Each element's local rank within its bucket is then just its
 * position in the contiguous run (threadIdx.x - s_lp[bucket]) — no serial
 * counting loop.  The global write produces coalesced transactions for any
 * bucket segment spanning at least one full warp.
 */
__global__ void scatter_kernel(unsigned int *in, unsigned int *out,
                               long int n, int shift,
                               long int *d_prefix, long int *d_block_offset,
                               unsigned int *d_block_count)
{
    const int NUM_WARPS = THREADS_PER_BLOCK / WARP_SIZE;
    long int tid  = (long int)blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid  = threadIdx.x / WARP_SIZE;

    __shared__ unsigned int s_val[THREADS_PER_BLOCK];
    __shared__ unsigned int s_tmp[THREADS_PER_BLOCK];
    __shared__ unsigned int s_wh[THREADS_PER_BLOCK/WARP_SIZE][16];
    __shared__ unsigned int s_bp[16];
    __shared__ unsigned int s_lp[RADIX];   /* local excl. prefix over buckets */

    s_val[threadIdx.x] = (tid < n) ? in[tid] : ~0u;
    __syncthreads();

    /* ---- Pass 1: sort tile stably by bits [3:0] of the 8-bit bucket ---- */
    {
        unsigned int k = (s_val[threadIdx.x] >> shift) & 0xF;

        /* warp histogram via ballot; save own ballot for intra-warp rank */
        unsigned int my_ballot = 0;
        for (int v = 0; v < 16; v++) {
            unsigned int b = __ballot_sync(0xFFFFFFFF, k == (unsigned int)v);
            if (lane == 0) s_wh[wid][v] = __popc(b);
            if (k == (unsigned int)v) my_ballot = b;
        }
        __syncthreads();

        /* total per sub-bucket → exclusive scan → block start offsets */
        if (threadIdx.x < 16) {
            unsigned int tot = 0;
            for (int w = 0; w < NUM_WARPS; w++) tot += s_wh[w][threadIdx.x];
            s_bp[threadIdx.x] = tot;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int run = 0;
            for (int v = 0; v < 16; v++) { unsigned int c = s_bp[v]; s_bp[v] = run; run += c; }
        }
        __syncthreads();

        /* warp prefix: offset of this warp within its sub-bucket */
        unsigned int wp = 0;
        for (int w = 0; w < wid; w++) wp += s_wh[w][k];

        unsigned int ir   = __popc(my_ballot & ((1u << lane) - 1u));
        unsigned int rank = s_bp[k] + wp + ir;

        s_tmp[rank] = s_val[threadIdx.x];
        __syncthreads();
        s_val[threadIdx.x] = s_tmp[threadIdx.x];
        __syncthreads();
    }

    /* ---- Pass 2: sort tile stably by bits [7:4] (stable relative to pass 1) ---- */
    {
        unsigned int k = ((s_val[threadIdx.x] >> shift) >> 4) & 0xF;

        unsigned int my_ballot = 0;
        for (int v = 0; v < 16; v++) {
            unsigned int b = __ballot_sync(0xFFFFFFFF, k == (unsigned int)v);
            if (lane == 0) s_wh[wid][v] = __popc(b);
            if (k == (unsigned int)v) my_ballot = b;
        }
        __syncthreads();

        if (threadIdx.x < 16) {
            unsigned int tot = 0;
            for (int w = 0; w < NUM_WARPS; w++) tot += s_wh[w][threadIdx.x];
            s_bp[threadIdx.x] = tot;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            unsigned int run = 0;
            for (int v = 0; v < 16; v++) { unsigned int c = s_bp[v]; s_bp[v] = run; run += c; }
        }
        __syncthreads();

        unsigned int wp = 0;
        for (int w = 0; w < wid; w++) wp += s_wh[w][k];

        unsigned int ir   = __popc(my_ballot & ((1u << lane) - 1u));
        unsigned int rank = s_bp[k] + wp + ir;

        s_tmp[rank] = s_val[threadIdx.x];
        __syncthreads();
        s_val[threadIdx.x] = s_tmp[threadIdx.x];
        __syncthreads();
    }

    /* ---- Blelloch exclusive scan over per-bucket counts → run start offsets ---- */
    if (threadIdx.x < RADIX)
        s_lp[threadIdx.x] = d_block_count[(long int)blockIdx.x * RADIX + threadIdx.x];
    __syncthreads();
    for (int stride = 1; stride < RADIX; stride <<= 1) {
        int i = (threadIdx.x + 1) * (stride << 1) - 1;
        if (i < RADIX) s_lp[i] += s_lp[i - stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) s_lp[RADIX - 1] = 0;
    __syncthreads();
    for (int stride = RADIX >> 1; stride >= 1; stride >>= 1) {
        int i = (threadIdx.x + 1) * (stride << 1) - 1;
        if (i < RADIX) {
            unsigned int t   = s_lp[i - stride];
            s_lp[i - stride] = s_lp[i];
            s_lp[i]          = t + s_lp[i];
        }
        __syncthreads();
    }

    if (tid < n) {
        unsigned int val    = s_val[threadIdx.x];
        unsigned int bucket = (val >> shift) & 0xFF;
        out[d_prefix[bucket]
            + d_block_offset[(long int)blockIdx.x * RADIX + bucket]
            + (threadIdx.x - s_lp[bucket])] = val;
    }
}



// ---------- GPU sort driver ----------

/*
 * radix_sort_gpu: orchestrate 4 passes on the device.
 *
 * Ping-pongs between d_arr and d_scratch so the final sorted result
 * lands back in d_arr -- same convention as the CPU version.
 * d_count and d_prefix are reused each pass.
 */
// returns per-kernel times (summed over all passes) in seconds
void radix_sort_gpu(unsigned int *d_arr, unsigned int *d_scratch,
                    long int n, long int *d_prefix,
                    double *t_count_sec, double *t_scan_sec, double *t_scatter_sec)
{
    int blocks = (int)((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);

    int shifts[PASSES] = {0, 8, 16, 24};
    unsigned int *src = d_arr;
    unsigned int *dst = d_scratch;

    unsigned int *d_block_count;
    long int     *d_block_offset;
    cudaMalloc(&d_block_count,  (size_t)blocks * RADIX * sizeof(unsigned int));
    cudaMalloc(&d_block_offset, (size_t)blocks * RADIX * sizeof(long int));

    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);
    float ms_count = 0.0f, ms_scan = 0.0f, ms_scatter = 0.0f, ms_tmp;

    for (int p = 0; p < PASSES; p++) {
        cudaMemset(d_block_count, 0, (size_t)blocks * RADIX * sizeof(unsigned int));

        cudaEventRecord(ev_start);
        count_kernel<<<blocks, THREADS_PER_BLOCK>>>(src, n, shifts[p], d_block_count);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms_tmp, ev_start, ev_stop);
        ms_count += ms_tmp;

        cudaEventRecord(ev_start);
        scan_kernel<<<1, RADIX>>>(d_block_count, d_prefix, d_block_offset, blocks);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms_tmp, ev_start, ev_stop);
        ms_scan += ms_tmp;

        cudaEventRecord(ev_start);
        scatter_kernel<<<blocks, THREADS_PER_BLOCK>>>(src, dst, n, shifts[p], d_prefix, d_block_offset, d_block_count);
        cudaEventRecord(ev_stop);
        cudaEventSynchronize(ev_stop);
        cudaEventElapsedTime(&ms_tmp, ev_start, ev_stop);
        ms_scatter += ms_tmp;

        unsigned int *tmp = src; src = dst; dst = tmp;
    }

    *t_count_sec   = (double)ms_count   * 1.0e-3;
    *t_scan_sec    = (double)ms_scan    * 1.0e-3;
    *t_scatter_sec = (double)ms_scatter * 1.0e-3;
    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_block_count);
    cudaFree(d_block_offset);
}

// total bytes moved across all passes - each pass reads n and writes n
double bytes_per_sort(long int n)
{
    return (double)PASSES * 2.0 * (double)n * sizeof(unsigned int);
}

// ---------- main ----------

int main(int argc, char *argv[])
{
    double wd;
    struct timespec time_start, time_stop;

    printf("EC527 Final Project - Radix Sort (GPU)\n");
    printf("LSD radix sort: %d-bit radix, %d passes, 32-bit unsigned int\n\n",
           RADIX_BITS, PASSES);

    wd = wakeup_delay();
    srand(42);  // same fixed seed as CPU version -- ensures identical inputs

    // allocate device buffers once and reuse across all runs
    long int max_n = test_sizes[NUM_SIZES - 1];
    unsigned int *d_arr, *d_scratch;
    long int     *d_prefix;

    cudaMalloc(&d_arr,     max_n * sizeof(unsigned int));
    cudaMalloc(&d_scratch, max_n * sizeof(unsigned int));
    cudaMalloc(&d_prefix,  RADIX * sizeof(long int));

    printf("dist, size, t_count_sec, t_scan_sec, t_scatter_sec, t_kernel_sec, t_e2e_sec, kernel_GB_per_sec, valid\n");

    for (int d = 0; d < NUM_DIST; d++) {
        for (int s = 0; s < NUM_SIZES; s++) {
            long int n = test_sizes[s];

            unsigned int *h_arr = (unsigned int *)malloc(n * sizeof(unsigned int));
            unsigned int *h_ref = (unsigned int *)malloc(n * sizeof(unsigned int));
            unsigned int *h_out = (unsigned int *)malloc(n * sizeof(unsigned int));

            if (!h_arr || !h_ref || !h_out) {
                printf("ERROR: malloc failed for n=%ld\n", n);
                return 1;
            }

            if (d == 0) gen_random (h_arr, n);
            else if (d == 1) gen_sorted (h_arr, n);
            else  gen_reverse(h_arr, n);

            // save input before sorting for validation
            memcpy(h_ref, h_arr, n * sizeof(unsigned int));

            // e2e: h2d transfer + GPU sort + d2h transfer
            double t_count, t_scan, t_scatter;
            clock_gettime(CLOCK_MONOTONIC, &time_start);
            cudaMemcpy(d_arr, h_arr, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
            radix_sort_gpu(d_arr, d_scratch, n, d_prefix, &t_count, &t_scan, &t_scatter);
            cudaMemcpy(h_out, d_arr, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            clock_gettime(CLOCK_MONOTONIC, &time_stop);
            double t_e2e    = interval(time_start, time_stop);
            double t_kernel = t_count + t_scan + t_scatter;

            double gbps = bytes_per_sort(n) / t_kernel / 1.0e9;
            int ok = validate(h_out, h_ref, n);

            printf("%s, %ld, %.6f, %.6f, %.6f, %.6f, %.6f, %.3f, %s\n",
                   dist_names[d], n,
                   t_count, t_scan, t_scatter, t_kernel, t_e2e,
                   gbps, ok ? "PASS" : "FAIL");

            free(h_arr); free(h_ref); free(h_out);
        }
    }

    cudaFree(d_arr); cudaFree(d_scratch); cudaFree(d_prefix);

    printf("\nWakeup delay: %f\n", wd);
    return 0;
}

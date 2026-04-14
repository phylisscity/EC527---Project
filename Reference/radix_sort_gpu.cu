/*
 * radix_sort_gpu.cu -- CUDA implementation of LSD radix sort
 * EC527 Final Project: High-Performance Sorting
 * Team: Cynthia Young & Phyliss Darko
 *
 * Compile: nvcc -arch=compute_86 -code=sm_86 radix_sort_gpu.cu -o radix_sort_gpu
 * (A40 GPU on SCC is compute capability 8.6)
 *
 * Branches from radix_sort_cpu.c. Input generation, validation, array sizes,
 * and output format are identical so CPU and GPU results can be compared directly.
 *
 * Transfer timing is reported separately from compute timing so we can see
 * how much overhead the host <-> device copies add at each array size.
 *
 * The three GPU kernels are stubbed out below -- implement them one at a time:
 *   1. count_kernel   -- histogram per pass
 *   2. scan_kernel    -- parallel prefix sum over 256 counts
 *   3. scatter_kernel -- place each element at its output position
 *
 * The CPU reference sort is kept here for sanity checking during development.
 *
 * Output columns:
 *   dist, size, t_transfer_sec, t_compute_sec, t_total_sec,
 *   compute_cycles, compute_GB_per_sec, valid
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

// ---------- configuration ----------

#define RADIX_BITS  8           // bits per pass
#define RADIX       256         // 2^RADIX_BITS buckets
#define PASSES      4           // 32-bit int / 8 bits per pass

#define CPNS        3.0         // cycles per nanosecond -- adjust to your machine

#define NUM_SIZES   7
static long int test_sizes[NUM_SIZES] = {
    1000000,     //   1M
    2000000,     //   2M
    4000000,     //   4M
    8000000,     //   8M
    16000000,    //  16M
    32000000,    //  32M
    64000000     //  64M
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

// forces CPU out of power-saving mode before timing starts -- identical to lab files
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

// ---------- CPU reference -- keep for sanity checking during GPU development ----------

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
 * Each thread processes one element, extracts its 8-bit bucket index
 * at bit position 'shift', and atomically increments a per-block
 * histogram in shared memory. After all threads finish, the block
 * adds its local histogram into the global count array.
 *
 * Hint: declare __shared__ long int local_count[RADIX]
 * Hint: use atomicAdd() for both shared and global accumulation
 */
__global__ void count_kernel(unsigned int *in, long int n,
                              int shift, long int *count)
{
    // TODO: implement
}

/*
 * scan_kernel: exclusive prefix sum over the 256-entry count array.
 *
 * Converts raw bucket counts into starting output positions, so
 * count[b] becomes the index where the first element of bucket b
 * should be written in the output array.
 *
 * A single block with 256 threads is sufficient for 256 elements.
 *
 * Hint: Hillis-Steele or Blelloch scan both work here
 * Hint: use __shared__ memory and __syncthreads()
 */
__global__ void scan_kernel(long int *count, long int *prefix)
{
    // TODO: implement
}

/*
 * scatter_kernel: place each element at its correct output position.
 *
 * Each thread reads one element, computes its bucket index,
 * uses atomicAdd on prefix[bucket] to claim its output slot,
 * then writes the element to that position in the output array.
 */
__global__ void scatter_kernel(unsigned int *in, unsigned int *out,
                               long int n, int shift, long int *prefix)
{
    // TODO: implement
}

// ---------- GPU sort driver ----------

/*
 * radix_sort_gpu: orchestrate 4 passes on the device.
 *
 * Ping-pongs between d_arr and d_scratch so the final sorted result
 * lands back in d_arr -- same convention as the CPU version.
 * d_count and d_prefix are reused each pass.
 */
void radix_sort_gpu(unsigned int *d_arr, unsigned int *d_scratch,
                    long int n, long int *d_count, long int *d_prefix)
{
    int threads = 256;
    int blocks  = (n + threads - 1) / threads;

    int shifts[PASSES] = {0, 8, 16, 24};
    unsigned int *src = d_arr;
    unsigned int *dst = d_scratch;

    for (int p = 0; p < PASSES; p++) {
        cudaMemset(d_count, 0, RADIX * sizeof(long int));

        count_kernel  <<<blocks, threads>>>(src, n, shifts[p], d_count);
        scan_kernel   <<<1, RADIX>>>       (d_count, d_prefix);
        scatter_kernel<<<blocks, threads>>>(src, dst, n, shifts[p], d_prefix);

        // swap buffers for next pass
        unsigned int *tmp = src; src = dst; dst = tmp;
    }
}

// total bytes moved across all passes -- each pass reads n and writes n
double bytes_per_sort(long int n)
{
    return (double)PASSES * 2.0 * (double)n * sizeof(unsigned int);
}

// ---------- main ----------

int main(int argc, char *argv[])
{
    double wd;
    struct timespec time_start, time_stop;

    printf("EC527 Final Project -- Radix Sort (GPU)\n");
    printf("LSD radix sort: %d-bit radix, %d passes, 32-bit unsigned int\n\n",
           RADIX_BITS, PASSES);

    wd = wakeup_delay();
    srand(42);  // same fixed seed as CPU version -- ensures identical inputs

    // allocate device buffers once and reuse across all runs
    long int max_n = test_sizes[NUM_SIZES - 1];
    unsigned int *d_arr, *d_scratch;
    long int     *d_count, *d_prefix;

    cudaMalloc(&d_arr,     max_n * sizeof(unsigned int));
    cudaMalloc(&d_scratch, max_n * sizeof(unsigned int));
    cudaMalloc(&d_count,   RADIX * sizeof(long int));
    cudaMalloc(&d_prefix,  RADIX * sizeof(long int));

    printf("dist, size, t_transfer_sec, t_compute_sec, t_total_sec, "
           "compute_cycles, compute_GB_per_sec, valid\n");

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

            if      (d == 0) gen_random (h_arr, n);
            else if (d == 1) gen_sorted (h_arr, n);
            else             gen_reverse(h_arr, n);

            // save input before sorting for validation
            memcpy(h_ref, h_arr, n * sizeof(unsigned int));

            // time host -> device transfer
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
            cudaMemcpy(d_arr, h_arr, n * sizeof(unsigned int), cudaMemcpyHostToDevice);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
            double t_h2d = interval(time_start, time_stop);

            // time GPU compute only (excluding transfer)
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
            radix_sort_gpu(d_arr, d_scratch, n, d_count, d_prefix);
            cudaDeviceSynchronize();
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
            double t_compute = interval(time_start, time_stop);

            // time device -> host transfer
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
            cudaMemcpy(h_out, d_arr, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);
            double t_d2h = interval(time_start, time_stop);

            double t_transfer = t_h2d + t_d2h;
            double t_total    = t_transfer + t_compute;
            long int cyc      = (long int)(CPNS * 1.0e9 * t_compute);
            double gbps       = bytes_per_sort(n) / t_compute / 1.0e9;
            int ok            = validate(h_out, h_ref, n);

            printf("%s, %ld, %.6f, %.6f, %.6f, %ld, %.3f, %s\n",
                   dist_names[d], n,
                   t_transfer, t_compute, t_total,
                   cyc, gbps, ok ? "PASS" : "FAIL");

            free(h_arr); free(h_ref); free(h_out);
        }
    }

    cudaFree(d_arr); cudaFree(d_scratch);
    cudaFree(d_count); cudaFree(d_prefix);

    printf("\nWakeup delay: %f\n", wd);
    return 0;
}

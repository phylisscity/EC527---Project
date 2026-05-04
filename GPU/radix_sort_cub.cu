/*
 * EC527 Final Project: High-Performance Sorting
 * Team: Cynthia Young & Phyliss Darko
 *
 * radix_sort_cub.cu -- CUB DeviceRadixSort baseline
 *
 * Compile:
 *   nvcc -arch=compute_86 -code=sm_86 -O3 radix_sort_cub.cu -o radix_sort_cub
 *
 * Runs over the same input sizes and distributions as radix_sort_gpu.cu so
 * results can be compared directly.
 *
 * Timing:
 *   t_kernel_sec  -- CUDA-event time around SortKeys call only (excludes
 *                    temp-storage allocation and transfers)
 *   t_e2e_sec     -- wall-clock time: h2d transfer + sort + d2h transfer
 *   kernel_GB_per_sec -- same formula as custom GPU: 4 passes × 2 × n × 4 B
 *
 * Output columns:
 *   dist, size, t_kernel_sec, t_e2e_sec, kernel_GB_per_sec, valid
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// ---------- configuration ----------

#define RADIX_BITS   8
#define PASSES       4     // 32-bit key / 8 bits per pass

#define NUM_SIZES    9
static long int test_sizes[NUM_SIZES] = {
    1000000,
    2000000,
    4000000,
    8000000,
    16000000,
    32000000,
    64000000,
    128000000,
    256000000
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

// ---------- input generation ----------

void gen_random(unsigned int *arr, long int n)
{
    for (long int i = 0; i < n; i++)
        arr[i] = ((unsigned int)rand() << 16) ^ (unsigned int)rand();
}

void gen_sorted(unsigned int *arr, long int n)
{
    for (long int i = 0; i < n; i++)
        arr[i] = (unsigned int)i;
}

void gen_reverse(unsigned int *arr, long int n)
{
    for (long int i = 0; i < n; i++)
        arr[i] = (unsigned int)(n - 1 - i);
}

// ---------- validation ----------

int cmp_uint(const void *a, const void *b)
{
    unsigned int ua = *(unsigned int *)a;
    unsigned int ub = *(unsigned int *)b;
    return (ua > ub) - (ua < ub);
}

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

// ---------- throughput formula (same as custom GPU for apples-to-apples) ----------

double bytes_per_sort(long int n)
{
    return (double)PASSES * 2.0 * (double)n * sizeof(unsigned int);
}

// ---------- main ----------

int main(int argc, char *argv[])
{
    double wd;
    struct timespec time_start, time_stop;

    printf("EC527 Final Project - Radix Sort (CUB DeviceRadixSort)\n");
    printf("LSD radix sort: %d-bit radix, %d passes, 32-bit unsigned int\n\n",
           RADIX_BITS, PASSES);

    wd = wakeup_delay();
    srand(42);

    long int max_n = test_sizes[NUM_SIZES - 1];

    // device buffers
    unsigned int *d_in, *d_out;
    cudaMalloc(&d_in,  max_n * sizeof(unsigned int));
    cudaMalloc(&d_out, max_n * sizeof(unsigned int));

    // query temp-storage size for max_n once; reuse for all smaller n
    void   *d_temp = nullptr;
    size_t  temp_bytes = 0;
    cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, (int)max_n);
    cudaMalloc(&d_temp, temp_bytes);

    // CUDA events reused across all runs
    cudaEvent_t ev_start, ev_stop;
    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_stop);

    printf("dist, size, t_kernel_sec, t_e2e_sec, kernel_GB_per_sec, valid\n");

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

            memcpy(h_ref, h_arr, n * sizeof(unsigned int));

            // e2e: h2d + sort kernel + d2h
            clock_gettime(CLOCK_MONOTONIC, &time_start);
            cudaMemcpy(d_in, h_arr, n * sizeof(unsigned int), cudaMemcpyHostToDevice);

            // kernel-only timing via CUDA events
            float ms_kernel = 0.0f;
            cudaEventRecord(ev_start);
            cub::DeviceRadixSort::SortKeys(d_temp, temp_bytes, d_in, d_out, (int)n);
            cudaEventRecord(ev_stop);
            cudaEventSynchronize(ev_stop);
            cudaEventElapsedTime(&ms_kernel, ev_start, ev_stop);

            cudaMemcpy(h_out, d_out, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
            clock_gettime(CLOCK_MONOTONIC, &time_stop);

            double t_kernel = (double)ms_kernel * 1.0e-3;
            double t_e2e    = interval(time_start, time_stop);
            double gbps     = bytes_per_sort(n) / t_kernel / 1.0e9;
            int ok = validate(h_out, h_ref, n);

            printf("%s, %ld, %.6f, %.6f, %.3f, %s\n",
                   dist_names[d], n, t_kernel, t_e2e, gbps, ok ? "PASS" : "FAIL");

            free(h_arr); free(h_ref); free(h_out);
        }
    }

    cudaEventDestroy(ev_start);
    cudaEventDestroy(ev_stop);
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_temp);

    printf("\nWakeup delay: %f\n", wd);
    return 0;
}

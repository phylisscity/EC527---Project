/*
 * EC527 Final Project: High-Performance Sorting
 * Team: Cynthia Young & Phyliss Darko
 *
 * Inspired by and structured after Lab 2 starter code from EC527 (Prof. Herbordt, BU)
 *
 * radix_sort_simd.c - SIMD/AVX2 accelerated LSD radix sort
 *
 * Compile: gcc -O1 -mavx2 radix_sort_simd.c -lrt -o radix_sort_simd
 *
 * Optimization attempt: AVX2 intrinsics on the count step.
 * Load 8 unsigned ints at a time, shift and mask all 8 in parallel
 * to extract bucket indices, then increment counts serially.
 * The histogram increment itself cannot be vectorized (data-dependent
 * index = gather/scatter, not supported by AVX2), so gains are limited
 * to the extraction overhead reduction. Prof noted this may not help.
 *
 * Algorithm: LSD (least significant digit) radix sort
 *   - 8-bit radix: 256 buckets, 4 passes for 32-bit unsigned integers
 *   - Each pass: SIMD-assisted count -> scan (prefix sum) -> scatter
 *   - No comparisons ever - purely bucket-based
 *
 * Input distributions tested:
 *   0 = random
 *   1 = already sorted (ascending)
 *   2 = reverse sorted (descending)
 *
 * Output: comma-separated, to paste directly into Google Sheets
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

// ---------- configuration ----------

#define RADIX_BITS  8           // bits per pass
#define RADIX       256         // 2^RADIX_BITS buckets
#define PASSES      4           // 32-bit int / 8 bits per pass

#define CPNS        5.8         // cycles per nanosecond - adjust to machine
                                // check with lscpu

#define NUM_SIZES   9
static long int test_sizes[NUM_SIZES] = {
    1000000,     //   1M
    2000000,     //   2M
    4000000,     //   4M
    8000000,     //   8M
    16000000,    //  16M
    32000000,    //  32M
    64000000,    //  64M
    128000000,   // 128M
    256000000    // 256M
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


// ---------- input generation ----------

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


// ---------- validation ----------

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


// ---------- SIMD radix sort pass ----------

/*
 * radix_sort_pass: one pass of LSD radix sort with AVX2-assisted count step
 *
 * Steps:
 *   1. Count (SIMD-assisted): load 8 elements at a time into AVX2 register,
 *      shift and mask all 8 in parallel to extract bucket indices,
 *      extract back to scalar and increment counts.
 *      The increment itself is scalar - AVX2 has no gather-increment instruction.
 *   2. Scan:  prefix sum over counts -> starting position per bucket
 *   3. Scatter: place each element at its correct output position (serial)
 */
void radix_sort_pass(unsigned int *in, unsigned int *out, long int n, int shift)
{
    long int count[RADIX];
    long int prefix[RADIX];

    // step 1: SIMD-assisted count
    // process 8 elements per iteration using AVX2 256-bit registers
    memset(count, 0, sizeof(count));

    __m256i vshift = _mm256_set1_epi32(shift);   // broadcast shift amount to all 8 lanes
    __m256i vmask  = _mm256_set1_epi32(0xFF);    // broadcast 0xFF mask to all 8 lanes

    long int n8 = n & ~7L;  // round down to nearest multiple of 8
    for (long int i = 0; i < n8; i += 8) {
        // load 8 unsigned ints from input
        __m256i v = _mm256_loadu_si256((__m256i*)&in[i]);

        // shift right and mask to extract 8-bit bucket index from all 8 elements at once
        __m256i buckets = _mm256_and_si256(_mm256_srlv_epi32(v, vshift), vmask);

        // extract each bucket index and increment count - must be scalar
        // (no AVX2 instruction for scatter-increment to arbitrary indices)
        unsigned int b[8];
        _mm256_storeu_si256((__m256i*)b, buckets);
        count[b[0]]++;
        count[b[1]]++;
        count[b[2]]++;
        count[b[3]]++;
        count[b[4]]++;
        count[b[5]]++;
        count[b[6]]++;
        count[b[7]]++;
    }
    // handle leftover elements (when n is not a multiple of 8)
    for (long int i = n8; i < n; i++)
        count[(in[i] >> shift) & 0xFF]++;

    // step 2: exclusive prefix sum - prefix[b] = starting output index for bucket b
    prefix[0] = 0;
    for (int b = 1; b < RADIX; b++)
        prefix[b] = prefix[b-1] + count[b-1];

    // step 3: scatter - iterate forward to preserve stability
    for (long int i = 0; i < n; i++) {
        unsigned int bucket = (in[i] >> shift) & 0xFF;
        out[prefix[bucket]++] = in[i];
    }
}

/*
 * radix_sort: sort array in place using 4 passes of 8-bit LSD radix sort
 *
 * Ping-pongs between arr and scratch so the final result
 * always lands back in the original array.
 */
void radix_sort(unsigned int *arr, unsigned int *scratch, long int n)
{
    radix_sort_pass(arr,     scratch, n,  0);  // bits  0-7
    radix_sort_pass(scratch, arr,     n,  8);  // bits  8-15
    radix_sort_pass(arr,     scratch, n, 16);  // bits 16-23
    radix_sort_pass(scratch, arr,     n, 24);  // bits 24-31
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

    printf("EC527 Final Project - SIMD/AVX2 Radix Sort\n");
    printf("LSD radix sort: %d-bit radix, %d passes, AVX2 8-wide count\n\n",
           RADIX_BITS, PASSES);

    wd = wakeup_delay();
    srand(42);

    printf("version, dist, size, time_sec, cycles, GB_per_sec, valid\n");

    for (int d = 0; d < NUM_DIST; d++) {
        for (int s = 0; s < NUM_SIZES; s++) {
            long int n = test_sizes[s];

            unsigned int *arr     = (unsigned int *)malloc(n * sizeof(unsigned int));
            unsigned int *scratch = (unsigned int *)malloc(n * sizeof(unsigned int));
            unsigned int *ref     = (unsigned int *)malloc(n * sizeof(unsigned int));

            if (!arr || !scratch || !ref) {
                printf("ERROR: malloc failed for n=%ld\n", n);
                return 1;
            }

            if      (d == 0) gen_random (arr, n);
            else if (d == 1) gen_sorted (arr, n);
            else             gen_reverse(arr, n);

            memcpy(ref, arr, n * sizeof(unsigned int));

            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_start);
            radix_sort(arr, scratch, n);
            clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &time_stop);

            double t     = interval(time_start, time_stop);
            long int cyc = (long int)(CPNS * 1.0e9 * t);
            double gbps  = bytes_per_sort(n) / t / 1.0e9;
            int ok       = validate(arr, ref, n);

            printf("simd, %s, %ld, %.6f, %ld, %.3f, %s\n",
                   dist_names[d], n, t, cyc, gbps, ok ? "PASS" : "FAIL");

            free(arr); free(scratch); free(ref);
        }
    }

    printf("\nWakeup delay: %f\n", wd);
    return 0;
}
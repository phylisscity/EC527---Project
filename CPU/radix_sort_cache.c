/*
 * EC527 Final Project: High-Performance Sorting
 * Team: Cynthia Young & Phyliss Darko
 *
 * Inspired by and structured after Lab 2 starter code from EC527 (Prof. Herbordt, BU)
 *
 * radix_sort_cache.c - Cache-blocked LSD radix sort
 *
 * Compile: gcc -O1 radix_sort_cache.c -lrt -o radix_sort_cache
 *
 * Optimization over serial baseline: cache blocking on the scatter step.
 * The serial version scatters one element at a time across the full output
 * array — at large N every write is a cache miss. This version processes
 * input in BLOCK_SIZE chunks so active scatter writes stay in L2 cache.
 *
 * Algorithm: LSD (least significant digit) radix sort
 *   - 8-bit radix: 256 buckets, 4 passes for 32-bit unsigned integers
 *   - Each pass: count -> scan (prefix sum) -> blocked scatter
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

// ---------- configuration ----------

#define RADIX_BITS  8           // bits per pass
#define RADIX       256         // 2^RADIX_BITS buckets
#define PASSES      4           // 32-bit int / 8 bits per pass

// chunk size for blocked scatter
// 4096 * 4 bytes = 16KB input chunk + 256*8 = 2KB for counts fits comfortably in L2
#define BLOCK_SIZE  4096

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


// ---------- cache-blocked radix sort pass ----------
/*
 * radix_sort_pass: one cache-blocked pass of LSD radix sort
 *
 * Same count -> scan -> scatter structure as the serial baseline,
 * but scatter is broken into BLOCK_SIZE chunks.
 *
 * Phase 1 - global count: full pass over input to tally bucket sizes.
 *   Needed upfront so we can compute correct output positions.
 *
 * Phase 2 - blocked scatter: process BLOCK_SIZE elements at a time.
 *   Each block re-counts its own elements, computes local start positions
 *   on top of the global prefix, then scatters only that block's elements.
 *   Keeping the active working set small reduces scatter cache misses
 *   compared to the serial version, which scatters across the full array.
 */
void radix_sort_pass(unsigned int *in, unsigned int *out, long int n, int shift)
{
    long int global_count[RADIX];
    long int prefix[RADIX];
    long int block_count[RADIX];
    long int block_prefix[RADIX];
    long int placed[RADIX];

    // phase 1: global count
    memset(global_count, 0, sizeof(global_count));
    for (long int i = 0; i < n; i++)
        global_count[(in[i] >> shift) & 0xFF]++;

    // global prefix sum - prefix[b] = starting output index for bucket b
    prefix[0] = 0;
    for (int b = 1; b < RADIX; b++)
        prefix[b] = prefix[b-1] + global_count[b-1];

    // phase 2: blocked scatter
    memset(placed, 0, sizeof(placed));
    for (long int start = 0; start < n; start += BLOCK_SIZE) {
        long int end = start + BLOCK_SIZE;
        if (end > n) end = n;

        // count elements in this block per bucket
        memset(block_count, 0, sizeof(block_count));
        for (long int i = start; i < end; i++)
            block_count[(in[i] >> shift) & 0xFF]++;

        // local start = global bucket start + elements already placed from prior blocks
        for (int b = 0; b < RADIX; b++)
            block_prefix[b] = prefix[b] + placed[b];

        // scatter this block's elements into output
        for (long int i = start; i < end; i++) {
            unsigned int bucket = (in[i] >> shift) & 0xFF;
            out[block_prefix[bucket]++] = in[i];
        }

        // update placed count for next block
        for (int b = 0; b < RADIX; b++)
            placed[b] += block_count[b];
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

    printf("EC527 Final Project - Cache-Blocked Radix Sort\n");
    printf("LSD radix sort: %d-bit radix, %d passes, block size: %d\n\n",
           RADIX_BITS, PASSES, BLOCK_SIZE);

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

            printf("cache, %s, %ld, %.6f, %ld, %.3f, %s\n",
                   dist_names[d], n, t, cyc, gbps, ok ? "PASS" : "FAIL");

            free(arr); free(scratch); free(ref);
        }
    }

    printf("\nWakeup delay: %f\n", wd);
    return 0;
}
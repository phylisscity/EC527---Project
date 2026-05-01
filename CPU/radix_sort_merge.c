/*
 * EC527 Final Project: High-Performance Sorting
 * Team: Cynthia Young & Phyliss Darko
 *
 * Inspired by and structured after Lab 2 starter code from EC527 (Prof. Herbordt, BU)
 *
 * radix_sort_merge.c - Parallel partition sort + merge tree
 *
 * Compile: gcc -O1 -fopenmp radix_sort_merge.c -lrt -o radix_sort_merge
 *
 * Approach: each OpenMP thread independently sorts its own contiguous
 * partition of the input using a full radix sort. After all threads finish,
 * sorted partitions are combined using a parallel merge tree:
 *   round 1 - adjacent pairs merge in parallel (N/2 merges at once)
 *   round 2 - pairs of those results merge in parallel (N/4 merges)
 *   ... until one fully sorted array remains
 *
 * This differs from radix_sort_omp.c where threads cooperate on each pass.
 * Here threads are fully independent during sorting, trading per-pass
 * synchronization for a merge cost at the end.
 *
 * Algorithm: LSD (least significant digit) radix sort per partition
 *   - 8-bit radix: 256 buckets, 4 passes for 32-bit unsigned integers
 *   - Each pass: count -> scan (prefix sum) -> scatter
 *   - No comparisons ever during sort - purely bucket-based
 *   - Comparisons only appear in the final merge step
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
#include <omp.h>

// ---------- configuration ----------

#define RADIX_BITS  8           // bits per pass
#define RADIX       256         // 2^RADIX_BITS buckets
#define PASSES      4           // 32-bit int / 8 bits per pass

#define CPNS        5.8         // cycles per nanosecond - adjust to machine
                                // check with lscpu

// number of partitions = number of threads
// must be a power of 2 for the merge tree to work cleanly
#define NUM_THREADS 4

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


// ---------- serial radix sort (runs on each partition) ----------

/*
 * radix_sort_pass: one pass of LSD radix sort
 * Identical to serial baseline - each thread runs this independently
 * on its own partition with no synchronization needed.
 */
void radix_sort_pass(unsigned int *in, unsigned int *out, long int n, int shift)
{
    long int count[RADIX];
    long int prefix[RADIX];

    // step 1: count
    memset(count, 0, sizeof(count));
    for (long int i = 0; i < n; i++)
        count[(in[i] >> shift) & 0xFF]++;

    // step 2: exclusive prefix sum
    prefix[0] = 0;
    for (int b = 1; b < RADIX; b++)
        prefix[b] = prefix[b-1] + count[b-1];

    // step 3: scatter
    for (long int i = 0; i < n; i++) {
        unsigned int bucket = (in[i] >> shift) & 0xFF;
        out[prefix[bucket]++] = in[i];
    }
}

void radix_sort_partition(unsigned int *arr, unsigned int *scratch, long int n)
{
    radix_sort_pass(arr,     scratch, n,  0);
    radix_sort_pass(scratch, arr,     n,  8);
    radix_sort_pass(arr,     scratch, n, 16);
    radix_sort_pass(scratch, arr,     n, 24);
}


// ---------- merge ----------

/*
 * merge_sorted: merge two adjacent sorted segments into tmp, copy back
 * left segment:  arr[0 .. mid-1]
 * right segment: arr[mid .. n-1]
 */
void merge_sorted(unsigned int *arr, unsigned int *tmp, long int n, long int mid)
{
    long int i = 0, j = mid, k = 0;
    while (i < mid && j < n) {
        if (arr[i] <= arr[j])
            tmp[k++] = arr[i++];
        else
            tmp[k++] = arr[j++];
    }
    while (i < mid) tmp[k++] = arr[i++];
    while (j < n)   tmp[k++] = arr[j++];
    memcpy(arr, tmp, n * sizeof(unsigned int));
}


// ---------- parallel partition sort + merge tree ----------

/*
 * radix_sort_merge: parallel sort + merge tree
 *
 * Phase 1 - parallel sort:
 *   Each thread gets a contiguous chunk of size n/NUM_THREADS.
 *   Threads run radix_sort_partition independently with no communication.
 *
 * Phase 2 - parallel merge tree:
 *   Round 1: threads pair up, each pair merges its two adjacent sorted chunks.
 *   Round 2: half as many active threads, each merges two chunks twice as large.
 *   Continues until one sorted array remains.
 *   Active threads halve each round - mirrors a tournament bracket.
 */
void radix_sort_merge(unsigned int *arr, unsigned int *scratch, long int n)
{
    long int chunk = (n + NUM_THREADS - 1) / NUM_THREADS;

    // phase 1: each thread sorts its own partition independently
    #pragma omp parallel num_threads(NUM_THREADS)
    {
        int tid = omp_get_thread_num();
        long int start = tid * chunk;
        long int end   = start + chunk;
        if (end > n) end = n;
        long int len = end - start;

        if (len > 0)
            radix_sort_partition(arr + start, scratch + start, len);
    }
    // all threads finish before merge begins
    // (implicit barrier at end of parallel region)

    // phase 2: parallel merge tree
    // active = number of merges happening this round
    // step = size of each sorted segment entering this round
    long int step = chunk;
    for (int active = NUM_THREADS / 2; active >= 1; active /= 2) {
        #pragma omp parallel for num_threads(active) schedule(static)
        for (int t = 0; t < active; t++) {
            long int start = t * 2 * step;
            long int mid   = start + step;
            long int end   = mid + step;
            if (mid > n) mid = n;
            if (end > n) end = n;
            if (mid < end)
                merge_sorted(arr + start, scratch + start, end - start, mid - start);
        }
        step *= 2;
    }
}

double bytes_per_sort(long int n)
{
    return (double)PASSES * 2.0 * (double)n * sizeof(unsigned int);
}


// ---------- main ----------

int main(int argc, char *argv[])
{
    double wd;
    struct timespec time_start, time_stop;

    printf("EC527 Final Project - Parallel Partition Sort + Merge Tree\n");
    printf("LSD radix sort per partition: %d-bit radix, %d passes, %d threads\n\n",
           RADIX_BITS, PASSES, NUM_THREADS);

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

            clock_gettime(CLOCK_MONOTONIC, &time_start);
            radix_sort_merge(arr, scratch, n);
            clock_gettime(CLOCK_MONOTONIC, &time_stop);

            double t     = interval(time_start, time_stop);
            long int cyc = (long int)(CPNS * 1.0e9 * t);
            double gbps  = bytes_per_sort(n) / t / 1.0e9;
            int ok       = validate(arr, ref, n);

            printf("merge, %s, %ld, %.6f, %ld, %.3f, %s\n",
                   dist_names[d], n, t, cyc, gbps, ok ? "PASS" : "FAIL");

            free(arr); free(scratch); free(ref);
        }
    }

    printf("\nWakeup delay: %f\n", wd);
    return 0;
}

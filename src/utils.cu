/* Some misc. utility functions for things like error checking, array generation, and device querying.
 */

#include "utils.h"
#include <stdio.h>
#include <time.h>

// Reads the value of i from the command line array and returns n = 2^i
int parse_args(int argc, char *argv[])
{
    if (argc != 2)
    {
        printf("Usage: ./vec_add <i, where n = 2^i>\n");
        exit(1);
    }
    
    return (int) pow(2, atoi(argv[1]));
}

// Fills a vector with random floats in the range [0, 1]
void init_vec(float *vec, int len)
{
    static int seeded = 0;
    if (!seeded)
    {
        srand(time(NULL));
        seeded = 1;
    }
    
    int i;
    for (i = 0; i < len; i++)
    {
        vec[i] = (float) rand() / RAND_MAX;
    }    
}

// Prints the given vector to stdout
void print_vec(const char *label, float *vec, int len)
{
#if PRINT_VECS
    printf("%s", label);
    
    int i;
    for (i = 0; i < len; i++)
    {
        printf("%f ", vec[i]);
    }
    printf("\n\n");
#endif
}

// Checks if an error occurred using the given status.
// If so prints the given message and halts.
void check_error(cudaError_t status, const char *msg)
{
    if (status != cudaSuccess)
    {
        const char *errorStr = cudaGetErrorString(status);
        printf("%s:\n%s\nError Code: %d\n\n", msg, errorStr, status);
        exit(status); // bail out immediately (makes debugging easier)
    }
}

// Queries the device to get the max number of threads that can be run per SM.
// Note: You can query many different h/w properties this way. For more info see
// the NVidia API documentation linked at the top of vec_add.cu
int get_max_block_threads()
{
    int dev_num;
    int max_threads;
    cudaError_t status;

    status = cudaGetDevice(&dev_num);
    check_error(status, "Error querying device number.");

    status = cudaDeviceGetAttribute(&max_threads, cudaDevAttrMaxThreadsPerBlock, dev_num);
    check_error(status, "Error querying max block threads.");

    return max_threads;
}

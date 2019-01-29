#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "utils.h"
#include "julia.h"
extern "C" {

#include "julia-c.h"
#include "qdbmp.h"

}
// Go through each pixel in the output image and tweak its colour value
// (such that when we're done, the colour values in the data array have a uniform distribution)

float * shared_cache;
int cache_set = 0;

// Takes an integer colour value and splits it into its RGB component parts.
// val (a 32-bit unsigned integer type) is expected to contain a 24-bit unsigned integer.
void toRGB(unsigned int val,
           unsigned char *r, unsigned char *g, unsigned char *b)
{
    // intentionally mixed up the order here to make the colours a little nicer...
    *r = (val & 0xff);
    val >>= 8;
    *b = (val & 0xff);
    val >>= 8;
    *g = (val & 0xff);
}

// Returns the sum of the elements in the given array.
unsigned int sum_array(unsigned int *array, int len)
{
    unsigned int total = 0;
    for (int i = 0; i < len; i++)
    {
        total += array[i];
    }

    return total;
}

__global__ void normalize_col(float * data, float * cache)
{
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int val;
    float hue;
    // expand the value's range from [0, 1] to [0, MAX_COLOUR]
    if(global_id < WIDTH * HEIGHT){
        //data[global_id] = __float2uint_rd(cache[__float2uint_rd(data[global_id])] * MAX_COLOUR);

        val = data[global_id];
        hue = cache[val];

        // expand the value's range from [0, 1] to [0, MAX_COLOUR]
        data[global_id] = (unsigned int) (hue * MAX_COLOUR);
    }
}

// Perform "histogram colour equalization" on the data array, using the
// information in the histogram array.
// This just ensures that the colours get nicely distributed to different
// values in the data array (i.e. makes sure that if the data array only contains values
// in a narrow range (between 100 and 200), the colours won't all be the same.
void hist_eq(unsigned int *data, unsigned int *hist)
{
    float hue = 0.0;
    if(!cache_set){
        float * cache = (float *)malloc(sizeof(float)*MAX_ITER);
        cache_set = 1;
        unsigned int total = sum_array(hist, MAX_ITER);
        for (unsigned int i = 0; i < MAX_ITER; i++)
        {
            cache[i] = hue;
            hue += (float) hist[i] / total;
        }
        cudaMalloc(&shared_cache, MAX_ITER);
        cudaMemcpy(shared_cache, cache, MAX_ITER, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        free(cache);
    }

    // Go through each pixel in the output image and tweak its colour value
    // (such that when we're done, the colour values in the data array have a uniform distribution)
    float * dev_data;
    int block_size = get_max_block_threads();
    int n = HEIGHT * WIDTH;
    int blocks = n / block_size + (n % block_size > 0 ? 1 : 0);
    cudaMalloc(&dev_data, HEIGHT*WIDTH);
    cudaMemcpy(dev_data,data,HEIGHT*WIDTH, cudaMemcpyHostToDevice);
    normalize_col<<<blocks,block_size>>>(dev_data,shared_cache);
    cudaMemcpy(data,dev_data,HEIGHT*WIDTH, cudaMemcpyDeviceToHost);
    cudaFree(dev_data);
    cudaFree(shared_cache);
    cache_set = false;
    cudaDeviceSynchronize();
}

// Writes the given data to a bitmap (.bmp) file with the given name.
// To do this, it interprets each value in the data array as an RGB colour
// (by calling toRGB()).
void write_bmp(unsigned int *data, char *fname)
{
    BMP *bmp = BMP_Create((UINT) WIDTH, (UINT) HEIGHT, (USHORT) DEPTH);
    unsigned char r, g, b;
    for (int y = 0; y < HEIGHT; y++)
    {
        for (int x = 0; x < WIDTH; x++)
        {
            toRGB(data[y * WIDTH + x], &r, &g, &b);
            BMP_SetPixelRGB(bmp, (UINT) x, (UINT) y,
                            (UCHAR) r, (UCHAR) g, (UCHAR) b);
        }
    }
        BMP_WriteFile(bmp, fname);
        BMP_Free(bmp);
}

int main(int argc, char *argv[])
{
    printf("Beginning julia set computation...\n");

    for(int frame = 0; frame < FRAME_COUNT; frame++){

        // The data array below stores the pixel data as a 1D array.
        // Each element represents colour of 1 pixel in the output image.
        // The hist (histogram) array the frequencies of the values in the data array.
        // E.g. If hist[2] == 30, that means the number appears 30 times in the data array.

        unsigned int * data = (unsigned int *)malloc(sizeof(int)*HEIGHT * WIDTH);
        unsigned int * hist = (unsigned int *)malloc(sizeof(int)*MAX_ITER);

        for(int i = 0; i < MAX_ITER; i++){
            hist[i] = 0;
        }

        drawFrame(data,hist,frame);

        hist_eq(data, hist);

        char filename [50];

        sprintf(filename, "bmpout/%05d_%s",frame,FNAME);

        printf("%s\n",filename);

        write_bmp(data, filename);

        free(data);
        free(hist);
    }
    printf("Done.\n");

    return EXIT_SUCCESS;
}

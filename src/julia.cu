#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <cuComplex.h>
#include "utils.h"
#include "julia.h"
extern "C" {

#include "julia-c.h"
#include "qdbmp.h"

}
// Go through each pixel in the output image and tweak its colour value
// (such that when we're done, the colour values in the data array have a uniform distribution)

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
/*unsigned int sum_array(unsigned int *array, int len)
{
    unsigned int total = 0;
    for (int i = 0; i < len; i++)
    {
        total += array[i];
    }

    return total;
}*/

__global__ void normalize_col(unsigned int * data, float * cache)
{
    unsigned int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int val;
    float hue;
    // expand the value's range from [0, 1] to [0, MAX_COLOUR]
    if(global_id < WIDTH*HEIGHT){
        //data[global_id] = __float2uint_rd(cache[__float2uint_rd(data[global_id])] * MAX_COLOUR);

        val = data[global_id];
        hue = cache[val];

        // expand the value's range from [0, 1] to [0, MAX_COLOUR]
        data[global_id] = (unsigned int) (hue * MAX_COLOUR);
    }
}

/*__device__ cuComplex cExp( cuComplex z ){
    cuComplex res;
    float s, c;
    float e = expf(z.x);
    sincosf(z.y, &s, &c);
    res.x = c * e;
    res.y = s * e;
    return res;
}

__device__ cuComplex csin( cuComplex z ){

    cuComplex i = make_cuComplex(0.0,1.0);
    cuComplex ni = make_cuComplex(0.0,-1.0);

    return cuCdivf(

        cuCsubf( cExp( cuCmulf(i,z) ) , cExp( cuCmulf(ni,z) )) 
        , ( cuCmulf(i,make_cuComplex(2,0)) ) 
    
    );
}

/*__device__ unsigned int julia_iters(cuComplex z)
{
    unsigned int iter = 0;
    cuComplex c = make_cuComplex( CR , CI );

    while (fabsf(cuCimagf(z)) < LIMIT && iter < MAX_ITER - 1)
    {
        z = cuCmulf(c , csin(z));
        iter++; 
    }

    //this value will be used to colour a pixel on the screen
    return iter;
}

__global__ void drawFrame(unsigned int * data, unsigned int * hist, int frame){

    int pixel = blockIdx.x * blockDim.x + threadIdx.x;

    // Build the output image one pixel at a time.
    if(pixel < WIDTH * HEIGHT){
        cuComplex z;
        float series_row;
        float series_col;
        unsigned int iters;
    
        int row = pixel/WIDTH;
        int col = pixel%WIDTH;

        series_row = row - HEIGHT / 2;
        series_col = col - WIDTH / 2;

        z = cuCaddf(
            make_cuComplex((series_col / RES_FACTOR ), 0) , 
            cuCmulf(
                cuCdivf(make_cuComplex(0,1) , make_cuComplex(RES_FACTOR,0)) 
                , make_cuComplex(series_row,0)));
        z.x *= SCALE;
        z.y *= SCALE;

        iters = julia_iters(z);
        data[pixel] = iters;
        atomicAdd(&hist[iters],1);
    }
}*/

// Perform "histogram colour equalization" on the data array, using the
// information in the histogram array.
// This just ensures that the colours get nicely distributed to different
// values in the data array (i.e. makes sure that if the data array only contains values
// in a narrow range (between 100 and 200), the colours won't all be the same.
void hist_eq(unsigned int *data, unsigned int *hist)
{
    float hue = 0.0;
    float * cache = (float *)malloc(sizeof(float)*MAX_ITER);
    float * shared_cache;
    unsigned int total = sum_array(hist, MAX_ITER);
    for (unsigned int i = 0; i < MAX_ITER; i++)
    {
        cache[i] = hue;
        hue += (float) hist[i] / total;
    }
    cudaMalloc(&shared_cache, MAX_ITER * sizeof(float));
    cudaMemcpy(shared_cache, cache, MAX_ITER, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    free(cache);

    // Go through each pixel in the output image and tweak its colour value
    // (such that when we're done, the colour values in the data array have a uniform distribution)
    unsigned int * dev_data;
    int block_size = get_max_block_threads();
    int n = HEIGHT * WIDTH;
    cudaError_t status;
    int blocks = n / block_size + (n % block_size > 0 ? 1 : 0);
    printf("block size: %d\tn: %d\tblocks: %d\tblock size * blocks: %d\n",block_size,n,blocks,block_size*blocks);
    cudaMalloc(&dev_data, WIDTH*HEIGHT*sizeof(int));

    cudaMemcpy(dev_data,data, WIDTH*HEIGHT*sizeof(int), cudaMemcpyHostToDevice);
    normalize_col<<<blocks,block_size>>>(dev_data,shared_cache);
    cudaMemcpy(data,dev_data, WIDTH*HEIGHT*sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_data);
    cudaFree(shared_cache);
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

        unsigned int * dev_data;
        unsigned int * dev_hist;


        int block_size = get_max_block_threads();
        int n = HEIGHT * WIDTH;
        int blocks = n / block_size + (n % block_size > 0 ? 1 : 0);

        /*cudaMalloc(&dev_hist, MAX_ITER);
        cudaMemcpy(dev_hist,hist,MAX_ITER, cudaMemcpyHostToDevice);

        cudaMalloc(&dev_data, HEIGHT*WIDTH);
        cudaMemcpy(dev_data,data,HEIGHT*WIDTH, cudaMemcpyHostToDevice);
        drawFrame<<<blocks,block_size>>>(dev_data,dev_hist,frame);
        cudaMemcpy(data,dev_data,HEIGHT*WIDTH, cudaMemcpyDeviceToHost);
        cudaFree(dev_data);

        cudaMemcpy(hist,dev_hist,MAX_ITER, cudaMemcpyDeviceToHost);
        cudaFree(dev_hist);

        cudaDeviceSynchronize();*/

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

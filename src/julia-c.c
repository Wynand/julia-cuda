#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "qdbmp.h"
#include "julia-c.h"

unsigned int julia_iters(float complex z);

void drawFrame(unsigned int * data, unsigned int * hist, int frame){
    // Build the output image one pixel at a time.
    for (int pixel = 0; pixel < HEIGHT*WIDTH; pixel++)
    {
        float complex z;
        float series_row;
        float series_col;
        unsigned int iters;
    
        int row = pixel/WIDTH;
        int col = pixel%WIDTH;

        series_row = row - HEIGHT / 2;
        series_col = col - WIDTH / 2;

        z = series_col / RES_FACTOR + (I / RES_FACTOR) * series_row;
        z *= SCALE;

        iters = julia_iters(z);
        data[pixel] = iters;

        hist[iters]++;
    }
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

// Perform "histogram colour equalization" on the data array, using the
// information in the histogram array.
// This just ensures that the colours get nicely distributed to different
// values in the data array (i.e. makes sure that if the data array only contains values
// in a narrow range (between 100 and 200), the colours won't all be the same.

/*void normalize_col(unsigned int * data, float * cache)
{
    //int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int val;
    float hue;
    // Go through each pixel in the output image and tweak its colour value
    // (such that when we're done, the colour values in the data array have a uniform distribution)
    for (int xy = 0; xy < WIDTH*HEIGHT; xy++)
    {
        val = data[xy];
        hue = cache[val];

        // expand the value's range from [0, 1] to [0, MAX_COLOUR]
        data[xy] = (unsigned int) (hue * MAX_COLOUR);
    }
}*/

// Generates terms of the Julia fractal sequence (starting with the given complex number)
// until either the imaginary part exceeds LIMIT or we hit MAX_ITER iterations.
unsigned int julia_iters(float complex z)
{
    unsigned int iter = 0;
    while (fabsf(cimag(z)) < LIMIT && iter < MAX_ITER - 1)
    {
        z = C * csin(z);
        iter++; 
    }

    //this value will be used to colour a pixel on the screen
    return iter;
}

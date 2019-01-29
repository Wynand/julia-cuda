#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "qdbmp.h"
#include "julia-c.h"

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
        z *= SCALE * pow(STEP_FACTOR,frame);

        iters = julia_iters(z);
        data[pixel] = iters;
        hist[iters]++;
    }
}

#ifndef _JULIA_C_H
#define _JULIA_C_H

#define WIDTH 600
#define HEIGHT 600
#define FRAME_COUNT 1000
#define STEP_FACTOR 0.99
// Controls how close we "zoom in" on the fractal pattern (lower = closer)
#define SCALE 2.0
// Complex number contant that controls the pattern that's generated (try changing it!)
// Note that values closer to 0.0 may significantly increase the computation time.
// You can set either of the terms to anything in the range -1.0 to 1.0. Just make sure to
// keep the trailing 'I'.
#define C (1.0 + 0.5I)
// Name of the file to same the output image in
#define FNAME "out.bmp"

/* Image Generation Constants - messing with these ones might break stuff */

#define DEPTH 24
#define MAX_COLOUR ((1L << DEPTH) - 1)
#define MAX_ITER ((1L << 16) - 1)
#define RES_FACTOR (sqrt(pow(WIDTH / 2, 2) + pow(HEIGHT / 2, 2)))
#define LIMIT 50
#define PI 3.14159265358979323846

/* Prototype Declarations */

void drawFrame(unsigned int * data, unsigned int * hist, int frame);

#endif
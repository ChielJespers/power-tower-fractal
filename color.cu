#include <gd.h>
#include <stdio.h>

#include <fenv.h>
#include <math.h>
#include <errno.h>

// ------------------------
// TO BE CUSTOMIZED BY USER
// ------------------------

// RENDERING PARAMETERS
#define sharpness     2000                                          // number of pixels specifying PNG pngWidth
#define maxIter       20000                                          // set higher for highly zoomed-in pictures
#define precision     3
// ------------------------
// COMPLEX DOMAIN
double reStart = -0.193;
double reEnd = -0.183;
double imStart = 0.23;
double imEnd = 0.24;

// See the bottom of this code for a discussion of some output possibilities.
char*   filename =   "ZoomIslesharp.png";

__global__
void fillColor(int n, int H, int W, int* color,
    int red, int green, int blue, int yellow, int purple, int orange, int black, int grey, int white, int maroon, int olive,
    double reStart, double reEnd, double imStart, double imEnd) {

  double epsilon = 1;
  double conv_radius = pow(10, -precision);

  int T = blockIdx.x*blockDim.x + threadIdx.x;
  if (T >= n) return;

  int x = T % H;
  int y = T / H;
  double re = reStart + ((double) x / W * (reEnd - reStart));
  double im = imEnd - ((double) y / H * (imEnd - imStart));

  double nextRe, nextIm, nextNextRe, nextNextIm, logRe, logIm, powerRe, powerIm;

  int toggleOverflow = 0;                                          
  int numberOfIterations = 0;                                      

  logRe = .5*log(re*re + im*im);
  logIm = atan2(im, re);
  nextRe = re;
  nextIm = im;
  while (numberOfIterations < maxIter && epsilon > conv_radius && toggleOverflow == 0)
  {
    // calculate current iteration
    powerRe = (nextRe * logRe - nextIm * logIm);
    powerIm = (nextRe * logIm + nextIm * logRe);

    if (powerRe > 700) {
        toggleOverflow = 1;
    }

    nextRe = exp(powerRe) * cos(powerIm);
    nextIm = exp(powerRe) * sin(powerIm);

    powerRe = (nextRe * logRe - nextIm * logIm);
    powerIm = (nextRe * logIm + nextIm * logRe);

    if (powerRe > 700) {
        toggleOverflow = 1;
    }

    // calculate iteration after, compare with current iteration
    nextNextRe = exp(powerRe) * cos(powerIm);
    nextNextIm = exp(powerRe) * sin(powerIm);

    epsilon = sqrt(pow(nextRe - nextNextRe, 2) + pow(nextIm - nextNextIm, 2));
    
    numberOfIterations += 1;
  }

  if (epsilon < conv_radius) {
    color[T] = black;
  }
  else if (numberOfIterations == maxIter) {
    int k = 1;
    while (sqrt(pow(nextRe - nextNextRe, 2) + pow(nextIm - nextNextIm, 2)) > conv_radius and k < maxIter + 1) {
        powerRe = (nextNextRe * logRe - nextNextIm * logIm);
        powerIm = (nextNextRe * logIm + nextNextIm * logRe);

        nextNextRe = exp(powerRe) * cos(powerIm);
        nextNextIm = exp(powerRe) * sin(powerIm);
    
        k += 1;
    }
    if (k == maxIter) {
        color[T] = black; // Assumed convergence
    }
    else {
        k = k % 8;
        if (k == 0)
          color[T] = maroon;
        else if (k == 1)
          color[T] = olive;
        else if (k == 2)
          color[T] = red;
        else if (k == 3)
          color[T] = green;
        else if (k == 4)
          color[T] = blue;
        else if (k == 5)
          color[T] = yellow;
        else if (k == 6)
          color[T] = purple; 
        else if (k == 7)
          color[T] = orange;
      }
  }
  else {
    color[T] = white;
  }
}

int main(){

  FILE*       outfile;                               // defined in stdio
  gdImagePtr  image;                                 // a GD image object
  int         T, x, y;                            // array subscripts
  int         red, green, blue, yellow, purple, orange, black, grey, white, maroon, olive; //all possible colors

  int pngWidth = sharpness;
  int pngHeight = pngWidth * (imEnd - imStart) / (reEnd - reStart);
  int N = pngWidth * pngHeight;

  //int** color = make2DintArray(pngWidth, pngHeight);
  int* color = (int*) malloc(N*sizeof(int));
  int* d_color;

  printf("width: %i\n", pngWidth);
  printf("height: %i\n", pngHeight);

  image = gdImageCreate(pngWidth, pngHeight);

  red    = gdImageColorAllocate(image, 255, 0, 0);
  green  = gdImageColorAllocate(image, 0, 255, 0);
  blue   = gdImageColorAllocate(image, 0, 0, 255);
  yellow = gdImageColorAllocate(image, 255, 255, 0);
  purple = gdImageColorAllocate(image, 127, 0, 255);
  orange = gdImageColorAllocate(image, 255, 128, 0);
  black  = gdImageColorAllocate(image, 0, 0, 0);
  grey   = gdImageColorAllocate(image, 127, 127, 127);
  white  = gdImageColorAllocate(image, 255, 255, 255);
  maroon = gdImageColorAllocate(image, 128, 0, 0);
  olive  = gdImageColorAllocate(image, 128, 128, 0);
  
  cudaMalloc(&d_color, N*sizeof(int));

  // Calculate power tower convergence / divergence
  fillColor<<<(pngWidth*pngHeight+255)/256, 256>>>(N, pngHeight, pngWidth, d_color,
    red, green, blue, yellow, purple, orange, black, grey, white, maroon, olive,
    reStart, reEnd, imStart, imEnd);
  cudaMemcpy(color, d_color, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (T=0; T<pngWidth*pngHeight; T++) {
    x = T % pngHeight;
    y = T / pngHeight;
    gdImageSetPixel(image, x, y, color[T]);
  }

  // Free 2D array
  free(color);
  cudaFree(d_color);
  // Finally, write the image out to a file.
  printf("Creating output file '%s'.\n", filename);
  outfile = fopen(filename, "wb");
  gdImagePng(image, outfile);
  fclose(outfile);
}
#include <gd.h>
#include <stdio.h>

#include <fenv.h>
#include <math.h>
#include <errno.h>

// ------------------------
// TO BE CUSTOMIZED BY USER
// ------------------------

// RENDERING PARAMETERS
#define sharpness     1000                                          // number of pixels specifying PNG pngWidth
#define maxIter       150                                          // set higher for highly zoomed-in pictures

// ------------------------
// COMPLEX DOMAIN
double reStart = -0.193;
double reEnd = -0.183;
double imStart = 0.23;
double imEnd = 0.24;

#define M_PI 3.14159265358979323846

// See the bottom of this code for a discussion of some output possibilities.
char*   filename =   "ZoomIslesharp.png";

int** make2DintArray(int arraySizeX, int arraySizeY);
void free2DintArray(int** myArray, int arraySizeX);
double getArg(double im);

int main(){
    // printf("MATH_ERRNO is %s\n", math_errhandling & MATH_ERRNO ? "set" : "not set");
    // printf("MATH_ERREXCEPT is %s\n",
    //        math_errhandling & MATH_ERREXCEPT ? "set" : "not set");
    // feclearexcept(FE_ALL_EXCEPT);
    // errno = 0;
    // printf("log(0) = %f\n", log(0));
    // if(errno == ERANGE)
    //     perror("errno == ERANGE");
    // if(fetestexcept(FE_DIVBYZERO))
    //     puts("FE_DIVBYZERO (pole error) reported");

  FILE*       outfile;                               // defined in stdio
  gdImagePtr  image;                                 // a GD image object
  int         i, T, x, y;                               // array subscripts
  int         blue, grey[256];                       // red, all possible shades of grey
  int         shade;

  double re, im, nextRe, nextIm, logRe, logIm, powerRe, powerIm;

  int toggleOverflow = 0;                            // BECOMES 1 AFTER OVERFLOW ERROR, STOPPING THE WHILE-LOOP
  int numberOfIterations = 0;                        // COUNTS THE NUMBER OF ITERATIONS ALREADY EXECUTED FOR CURRENT COMPLEX NUMBER

  int pngWidth = sharpness;
  int pngHeight = pngWidth * (imEnd - imStart) / (reEnd - reStart);

  //int** color = make2DintArray(pngWidth, pngHeight);
  int* color = (int*) malloc(pngWidth*pngHeight*sizeof(int));

  printf("width: %i\n", pngWidth);
  printf("height: %i\n", pngHeight);

  image = gdImageCreate(pngWidth, pngHeight);

  blue  = gdImageColorAllocate(image, 0, 0, 255);
  
  for (i=0; i<256; i++){
    grey[i] = gdImageColorAllocate(image, i,i,i);
  }

  // Calculate power tower convergence / divergence
  for (T=0; T<pngWidth*pngHeight; T++){
    x = T % pngHeight;
    y = T / pngHeight;
    re = reStart + ((double) x / pngWidth * (reEnd - reStart));
    im = imEnd - ((double) y / pngHeight * (imEnd - imStart));
        
    toggleOverflow = 0;                                          // BECOMES 1 AFTER OVERFLOW ERROR, STOPPING THE WHILE-LOOP
    numberOfIterations = 0;                                      // COUNTS THE NUMBER OF ITERATIONS ALREADY EXECUTED FOR CURRENT COMPLEX NUMBER
    if (re == 0 && im == 0){
        color[T] = blue;
    }
    else{
        logRe = log(sqrt(re*re + im*im));
        logIm = getArg(im);
        nextRe = re;
        nextIm = im;
        while (numberOfIterations < maxIter && toggleOverflow == 0)
        {
            powerRe = (nextRe * logRe - nextIm * logIm);
            powerIm = (nextRe * logIm + nextIm * logRe);

            if (powerRe > 700) {
                printf("ga werken kk ding\n");
                toggleOverflow = 1;
            }

            nextRe = exp(powerRe) * cos(powerIm);
            nextIm = exp(powerRe) * sin(powerIm);

            if (nextRe > 10) {
                printf("ga werken kk ding\n");
                toggleOverflow = 1;
            }
            
            numberOfIterations += 1;
        }
    }

    shade = 255 - ((numberOfIterations * 255) / maxIter);
    color[T] = grey[shade];
  }

  // Generate image in memory
  // for (x=0; x<pngWidth; x++){
  //   for (y=0; y<pngHeight; y++){
  //     gdImageSetPixel(image, x, y, color[x][y]);
  //   }
  // }
  for (T=0; T<pngWidth*pngHeight; T++) {
    x = T % pngHeight;
    y = T / pngHeight;
    gdImageSetPixel(image, x, y, color[T]);
  }

  // Free 2D array
  //free2DintArray(color, pngWidth);
  free(color);
  // Finally, write the image out to a file.
  printf("Creating output file '%s'.\n", filename);
  outfile = fopen(filename, "wb");
  gdImagePng(image, outfile);
  fclose(outfile);
}

int** make2DintArray(int arraySizeX, int arraySizeY) {  
  int** theArray;
  theArray = (int**) malloc(arraySizeX*sizeof(int*));
  for (int i = 0; i < arraySizeX; i++) {
    theArray[i] = (int*) malloc(arraySizeY*sizeof(int));
  }
  return theArray;  
}

void free2DintArray(int** myArray, int arraySizeX) {
  for (int i = 0; i < arraySizeX; i++){  
    free(myArray[i]);  
  }  
  free(myArray);    
}

double getArg(double im) {
    while (im > M_PI) {
        im -= 2*M_PI;
    }
    while (im <= -M_PI ) {
        im += 2*M_PI;
    }
    return im;
}
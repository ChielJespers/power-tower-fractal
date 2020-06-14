#include <complex.h>    /* Standard Library of Complex Numbers */
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

// See the bottom of this code for a discussion of some output possibilities.
char*   filename =   "ZoomIslesharp.png";

int** make2DintArray(int arraySizeX, int arraySizeY);
void free2DintArray(int** myArray, int arraySizeX);

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

  double re, im, nextRe, nextIm;
  double complex firstIterate, firstLog, nextIterate, power;

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
  //for (x=0; x<pngWidth; x++){
  //  for (y=0; y<pngHeight; y++){
  //    re = reStart + ((double) x / pngWidth * (reEnd - reStart));
  //    im = imEnd - ((double) y / pngHeight * (imEnd - imStart));
      x = T % pngHeight;
      y = T / pngHeight;
      re = reStart + ((double) x / pngWidth * (reEnd - reStart));
      im = imEnd - ((double) y / pngHeight * (imEnd - imStart));
      firstIterate = re + I * im;           // SCALES CURRENT PNG-COORDINATE TO A COMPLEX NUMBER IN THE SPECIFIED DOMAIN
          
      toggleOverflow = 0;                                          // BECOMES 1 AFTER OVERFLOW ERROR, STOPPING THE WHILE-LOOP
      numberOfIterations = 0;                                      // COUNTS THE NUMBER OF ITERATIONS ALREADY EXECUTED FOR CURRENT COMPLEX NUMBER
      if (firstIterate == 0){
        // color[x][y] = blue;
        color[T] = blue;
      }
      else{
          firstLog = clog(firstIterate);
          nextIterate = firstIterate;
          while (numberOfIterations < maxIter && toggleOverflow == 0)
          {
            feclearexcept(FE_ALL_EXCEPT);
            
            power = nextIterate * firstLog;
            nextIterate = cexp(power);
            if(fetestexcept(FE_OVERFLOW)) {
              toggleOverflow = 1;
            }

            numberOfIterations += 1;
          }
      }

      shade = 255 - ((numberOfIterations * 255) / maxIter);
      color[T] = grey[shade];
      // color[x][y] = grey[shade];
  //  }
  //}
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

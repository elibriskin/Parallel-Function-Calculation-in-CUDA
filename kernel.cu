#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>

# define M_PI           3.14159265358979323846  /* pi */

#define CHECK(call) \
{ \
 const cudaError_t error = call; \
 if (error != cudaSuccess) \
 { \
 printf("Error: %s:%d, ", __FILE__, __LINE__); \
 printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
 exit(1); \
 } \
}

float* linspace(float start, float stop, int size) {

    //Allocate memory for array of floats
    float* arr = new float[size];

    //Determine step increment based off of the range of the array divided by the size
    float arr_increment = (stop - start) / size;

    //Create array of linearly spaced points
    for (int i = 0; i < size; i++) {
        arr[i] = start + i * arr_increment;
    };
    return arr;
};

void xmesh(float* ip,
    float start,
    float stop,
    int nx,
    int ny) {
    int id = 0;

    //Allocate array memory of linearly spaced points in x dimension
    float* arr = linspace(start, stop, nx);

    //Will instantiate a 2D matrix where each row is an array of linearly spaced
    //points - this is the 'x' dimension of the mesh grid
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            ip[id] = arr[j];
            id++;
        }
    }
    delete[] arr;
}

void ymesh(float* ip,
    float start,
    float stop,
    int nx,
    int ny) {
    int id = 0;

    //Allocate array memory of linearly spaced points in y dimension
    float* arr = linspace(start, stop, nx);

    //Will instantiate a 2D matrix where each column is an array of linearly spaced
    //points - this is the 'y' dimension of the mesh grid
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            ip[id] = arr[i];
            id++;
        }
    }
    delete[] arr;
}


void printMatrix(float* C, const int nx, const int ny) {

    //Allocate matrix
    float* ic = C;
    printf("\nMatrix: (%d.%d)\n", nx, ny);

    //For each element in matrix, print the element
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            printf("%0f", ic[ix]);
        }
        ic += nx;
        printf("\n");
    }
    printf("\n");
}

void writeCSV(std::string filename, float* C, const int nx, const int ny) {

    //Instantiate a file to write data to
    std::ofstream file(filename);
    float* ic = C;
    if (!file.is_open()) {
        std::cerr << "Failed to open file!" << std::endl;
    }

    //For each row in matrix, write each row as a row in the CSV file
    for (int iy = 0; iy < ny; iy++) {
        for (int ix = 0; ix < nx; ix++) {
            file << ic[ix];
            file << ",";
        }
        ic += nx;

        //Write new row in CSV file
        file << "\n";
    }
}

__global__ void funcMatrix(float* matB, float*X, float*Y, int nx, int ny) {

    //Get ID for blocks and threads in blocks for x and y dimensions 
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;

    if (ix < nx && iy < ny) {

        //The width of the 2D gaussian function
        float width = 1;

        //Two dimensional gaussian function for each matrix element
        float val = (1 / (2 * M_PI * pow(width, 2))) * exp(-1*(pow(X[idx], 2) + pow(Y[idx], 2))/(2*pow(width, 2)));
        matB[idx] = val;
    }
}

int main(int argc, char** argv)
{
    printf("Starting application %s...", argv[0]);

    //Get device properties
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    //Set up matrix size
    float matrix_dim = 10;
    float matrix_start = -matrix_dim / 2;
    float matrix_stop = matrix_dim / 2;

    //Set up matrix with x and y dimensions
    int nx = 5000;
    int ny = 5000;

    //Total size of matrix
    int nxy = nx * ny;
    int nBytes = nxy * sizeof(float);

    //Initialize unified memory matrices
    float* X, *Y, * gpuRef;
    printf("The size of your matrix is % d by % d, with an area of % d.\n", nx, ny, nxy);

    //Allocate memory on host
    cudaMallocManaged((float**)&X, nBytes);
    cudaMallocManaged((float**)&Y, nBytes);
    cudaMallocManaged((float**)&gpuRef, nBytes);

    //Initialize data on matrices on host side
    xmesh(X, matrix_start, matrix_stop, nx, ny);
    ymesh(Y, matrix_start, matrix_stop, nx, ny);

    //Specify block and grid dimensions for thread managemenet and invoke the GPU kernel
    int dimx = 25;
    int dimy = 25;
    dim3 block(dimx, dimy);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    //Execute matrix function on GPU
    funcMatrix <<< grid, block >>> (gpuRef, X, Y, nx, ny);
    cudaDeviceSynchronize();

    //Write out matrix as csv file
    std::string filepath = "matrix_data.csv";
    writeCSV(filepath, gpuRef, nx, ny);

    ////Free unified memory
    cudaFree(X);
    cudaFree(Y);
    cudaFree(gpuRef);

    //Reset GPU
    cudaDeviceReset();

    return 0;
}



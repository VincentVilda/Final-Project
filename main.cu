//By Vincent Vilda and Jacob Osorio

#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "support.h"


using namespace std;
using namespace cv;


void Histogram(int Height, int Width, unsigned char * I_dev, int * bin_dev);


int main(){

    ///Timer
    Timer timer;

    ///read the Im in grayscale 0 to 255. 8bit
    // Mat I = imread("Lenna.png", 0); 
    Mat I = imread("cameraman.png", 0);

    cout << "Height= " << I.rows << ", Width= " << I.cols << ", Channels= " << I.channels() << endl;

    ///Take the histogram using OpenCV Function use for comparason.
    printf("Running CPU histogram"); fflush(stdout);
    startTime(&timer); //Timer start for OpenCV Histogram

    float range[] = { 0, 256 };
    const float* histRange[] = { range };
    int histSize = 256;
    bool uniform = true, accumulate = false;
    Mat opencvHist;
    calcHist( &I, 1, 0, cv::Mat(), opencvHist, 1, &histSize, histRange, uniform, accumulate );
 
    stopTime(&timer); printf("%f s\n", elapsedTime(timer)); //Timer Stops


    int bin_h[256] = { 0 };

    unsigned char* I_d = NULL;
    int* bin_d = NULL;



    ///allocate cuda memory
    cudaMalloc((void**)&I_d, I.rows * I.cols * I.channels());
    cudaMalloc((void**)&bin_d, 256 * sizeof(int));




    ///copy CPU to GPU
    cudaMemcpy(I_d, I.data, I.rows*I.cols * I.channels(), cudaMemcpyHostToDevice);
    cudaMemcpy(bin_d, bin_h, 256 * sizeof(int), cudaMemcpyHostToDevice);
    



    //call the histogram function
    printf("Running GPU Histogram"); fflush(stdout);
    startTime(&timer); //Timer starts for GPU histogram

    Histogram(I.rows, I.cols, I_d,  bin_d); //start kernel

    stopTime(&timer); printf("%f s\n", elapsedTime(timer)); //Timer Stops





    //copy memory back to CPU from GPU
    cudaMemcpy(bin_h, bin_d, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    
 


    //Print Result
    for (int i = 0; i < 256; i++){
        cout << "bin[" << i << "]: " <<"CUDA = "<< bin_h[i]/32 << " ,     CPU =  "<< opencvHist.at<float>(i)<<endl;
    }




    //free up the memory from GPU
    cudaFree(bin_d);
    cudaFree(I_d);
    system("pause");
    return 0;
}


///////////////////////////////Below are the 3 Kernel Used for Performace test///////////////////////////////////



/////////////////////////Initial Code with block dimension equal to 1/////////////////
// __global__ void kernel(unsigned char* Im, int* Hist);

// void Histogram(int Height, int Width, unsigned char * I_dev, int * bin_dev){

//     dim3 dimGrid(Width, Height);
//     kernel<<<dimGrid, 1 >>>(I_dev, bin_dev);

// }



// __global__ void kernel(unsigned char* Im, int* Hist){
//     int x = blockIdx.x;
//     int y = blockIdx.y;

//     int Im_Idx = x + y * gridDim.x;

//     atomicAdd(&Hist[Im[Im_Idx]], 1);
// }
////////////////////////////////////////////////////////////////////////////////////



////////////////////////Kernel with optimized block dimension ///////////////////////
// __global__ void kernel(unsigned char* Im, int* Hist, int Height, int Width);

// void Histogram(int Height, int Width, unsigned char * I_dev, int * bin_dev){
//     dim3 blockDim(32, 32); 
//     dim3 gridDim((Width + blockDim.x - 1) / blockDim.x, (Height + blockDim.y - 1) / blockDim.y);

//     kernel<<<gridDim, blockDim>>>(I_dev, bin_dev, Height, Width);
// }


// __global__ void kernel(unsigned char* Im, int* Hist, int Height, int Width){
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;

//     if (x < Width && y < Height) {
//         int Im_Idx = x + y * Width;
//         atomicAdd(&Hist[Im[Im_Idx]], 1);
//     }
// }
////////////////////////////////////////////////////////////////////////////////////////////



////////////////////////Kernel with optimized dim block and shared memory bins///////////////////////
__global__ void kernel(unsigned char* Im, int* Hist, int Height, int Width){
    __shared__ int local_hist[256]; // Shared memory for local histograms
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Initialize local hist
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        local_hist[i] = 0;
    }
    __syncthreads();

    //Start to bin in shared hist
    if (x < Width && y < Height) {
        int Im_Idx = x + y * Width;
        atomicAdd(&local_hist[Im[Im_Idx]], 1); 
    }
    __syncthreads();

    // Bin in host hist
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        atomicAdd(&Hist[i], local_hist[i]);
    }
}

void Histogram(int Height, int Width, unsigned char * I_dev, int * bin_dev){
    dim3 blockDim(32, 32); 
    dim3 gridDim((Width + blockDim.x - 1) / blockDim.x, (Height + blockDim.y - 1) / blockDim.y);

    // Allocate mem for hist device
    cudaMemset(bin_dev, 0, 256 * sizeof(int));

    kernel<<<gridDim, blockDim>>>(I_dev, bin_dev, Height, Width);
}

///////////////////////////////////////////////////////////////////////////////////////////////////





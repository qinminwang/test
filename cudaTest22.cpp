#include<stdio.h>
#include<stdlib.h>

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
void GenerateNumbers(int *number,int size)
 {   
     for(int i = 0;i<size;i++)
     {
         number[i] = rand()%10;
     }
 }

bool InitCUDA()
{
    int count;
    cudaGetDeviceCount(&count);
    printf("threr are %d device!",count);
    if(count == 0){
        fprintf(stderr,"there is no device.\n");
        return false;
    }
    int i;
    for(i = 0;i<count;i++)
    {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            if (prop.major >= 1) {
                break;
            }
        }
    }
    if (i == count) {
        fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
        return false;
    }

    cudaSetDevice(i);

    return true;
 }

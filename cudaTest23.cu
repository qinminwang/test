// __global__ 函数 (GPU上运行) 计算立方和

#include<stdio.h>
#define DATA_SIZE 1024*1024
__global__ static void sumOfSquares(int *num, int* result)
{
    int sum = 0;

    int i;

    for (i = 0; i < DATA_SIZE; i++) {

        sum += num[i] * num[i] * num[i]-num[i]*num[i]+
	num[i]/(num[i]+1) * num[i]/(num[i]+2) * num[i]/(num[i]+3);
    }

    *result = sum;

}

__global__ static void sum(int *sum,int* result)
{
    __shared__ int sum_number[1024];
    int number = 0;
    for(int i = 0;i<blockDim.x;i++)
	number += sum[i+threadIdx.x*blockDim.x];
    sum_number[threadIdx.x] = number;
    __syncthreads(); 
    number = 0;
    if(0 == threadIdx.x)
    {
        for(int i = 0;i<blockDim.x;i++)
	{
    	    number += sum_number[i];
	}
	*result = number;
    }
}

__global__ static void sprintf_gpu(int a)
{
	printf("result:%d",a*blockIdx.x*blockDim.x+threadIdx.x);
}

__global__ static void squares(int* num, int *squ)
{
    int i = blockIdx.x*blockDim.x+threadIdx.x;
   // if(i == j)
	squ[i] = num[i] * num[i] * num[i]-num[i]*num[i]+
num[i]/(num[i]+1) * num[i]/(num[i]+2) * num[i]/(num[i]+3);
	
}
extern "C" void fun(int *num,int *squ, int* result)
{
    //sumOfSquares << <64,64 , 0 >> >(num, result);

    squares<<<1024,1024>>>(num,squ);
    //cudaThreadSynchronize();
	
    sum<<<1,1024,0>>>(squ,result);
}


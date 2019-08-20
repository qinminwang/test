#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include<cublas.h>
#include<cublas_v2.h>
#include<distance_gpu.h>

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#define DATA_SIZE 1024*1024

int data[DATA_SIZE];


void GenerateNumbers(int *number,int size);
bool InitCUDA();
extern "C" void fun(int *num,int *squ, int* result);


int main()
{

float *mat_B = (float*)malloc(sizeof(float)*100);
for(int i = 0;i<100;i++)
mat_B[i]=0.1*(i+1);
float *vec_A= (float*)malloc(sizeof(float)*10);
for(int j = 0;j<10;j++)
vec_A[j]=0.1*(1+j);
//float *dist = (float*)malloc(sizeof(float)*10);
float dist[10];
initDist(9,9);
computeDist1(vec_A,mat_B, &dist[0]);
for(int i = 0 ;i< 10;i++)
std::cout<<dist[i]<<"\t";
std::cout<<std::endl;
destoryDist();
initDist(8,8);
computeDist1(vec_A,mat_B, &dist[0]);
for(int i = 0 ;i< 10;i++)
std::cout<<dist[i]<<"\t";
std::cout<<std::endl;
destoryDist();
return 0;
}
int main_old()
{
//clock_t  start_gpu, stop_gpu;
cudaEvent_t start_GPU, stop_GPU;
float time_GPU;    
//CUDA 初始化
//    if (!InitCUDA()) {
//	printf("initcuda error");
//        return 0;
//    }
int const m = 5;
    int const n = 3;
    int const k = 2;
float *A ,*B,*C;
    float *d_A,*d_B,*d_C;
A = (float*)malloc(sizeof(float)*m*k); //在内存中开辟空间 
B = (float*)malloc(sizeof(float)*n*k); //在内存中开辟空间
 C = (float*)malloc(sizeof(float)*m*n); //在内存中开辟空间
for(int i = 0; i< m*k; i++){
        A[i] = i;
      }
for(int i = 0; i< n*k; i++){
        B[i] = i;
      }
float alpha = 1.0;
    float beta = 0.0;
cublasAlloc(m*k, sizeof(float), (void**)&d_A);
        cublasAlloc(n*k, sizeof(float), (void**)&d_B);
        cublasAlloc(m*n, sizeof(float), (void**)&d_C);
//cudaMalloc((void**)&d_A,sizeof(float)*m*k);
//    cudaMalloc((void**)&d_B,sizeof(float)*n*k);
//    cudaMalloc((void**)&d_C,sizeof(float)*m*n);
cudaMemcpy(d_A,A,sizeof(float)*m*k,cudaMemcpyHostToDevice);
 //   cudaMemcpy(d_B,B,sizeof(float)*n*k,cudaMemcpyHostToDevice);
    cublasSetVector(n*k, sizeof(float), B, 1, d_B, 1);
cublasHandle_t handle;
    cublasCreate(&handle);
//cudathreadsynchronize();
//cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, d_A, m, d_B, k, &beta, d_C, m);
cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);
//cudaMemcpy(C,d_C,sizeof(float)*m*n,cudaMemcpyDeviceToHost);
cublasGetVector(m*n, sizeof(float), d_C, 1, C, 1);
//cudathreadsynchronize();
for (int i = 0; i< m*n;i++){
	if(!(i%n))
	    printf("\n");
        printf("%f\t",C[i]);
    }
printf("\n");
free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
cublasDestroy(handle);
    //生成随机数
    GenerateNumbers(data, DATA_SIZE);

    /*把数据拷贝到显卡内存中*/

printf("GPUsum: begin------ \n");

    int* gpudata, *result;
    int *squares;
    int cpusum = 0;
    //cudaMalloc 取得一块显卡内存 ( 当中result用来存储计算结果 )
    cudaMalloc((void**)&gpudata, sizeof(int)* DATA_SIZE);
    cudaMalloc((void**)&squares, sizeof(int)* DATA_SIZE);
    cudaMalloc((void**)&result, sizeof(int));

    //cudaMemcpy 将产生的随机数拷贝到显卡内存中 
    //cudaMemcpyHostToDevice - 从内存拷贝到显卡内存
    //cudaMemcpyDeviceToHost - 从显卡内存拷贝到内存
    cudaMemcpy(gpudata, data, sizeof(int)* DATA_SIZE, cudaMemcpyHostToDevice);

    // 在CUDA 中运行函数 语法：函数名称<<<block 数目, thread 数目, shared memory 大小>>>(參数...);
//创建Event
  cudaEventCreate(&start_GPU); cudaEventCreate(&stop_GPU);
  cudaEventRecord(start_GPU, 0);
 // start_gpu = clock();
   
fun(gpudata,squares, result);


cudaEventRecord(stop_GPU, 0);
cudaEventSynchronize(start_GPU);
cudaEventSynchronize(stop_GPU); 
 cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU); 
printf("The time for GPU:\t%f(ms)\n", time_GPU);
cudaEventDestroy(start_GPU);
cudaEventDestroy(stop_GPU);
    /*把结果从显示芯片复制回主内存*/

    int sum;

    //cudaMemcpy 将结果从显存中复制回内存
    cudaMemcpy(&sum, result, sizeof(int), cudaMemcpyDeviceToHost);

    //Free
    cudaFree(gpudata);
    cudaFree(result);

    printf("GPUsum: %d \n", sum);

    sum = 0;

    for (int i = 0; i < DATA_SIZE; i++) {
        sum += data[i] * data[i] * data[i]-data[i]*data[i]+data[i]/(data[i]+1) * data[i]/(data[i]+2) * data[i]/(data[i]+3);
    }

    printf("CPUsum: %d \n", sum);

    return 0;
}

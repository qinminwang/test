#include<stdio.h>
#include<iostream>
#include<stdlib.h>
#include<vector>
#include<bitset>

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#define DATA_SIZE 2000
#define DOUBLE_DATA_SIZE 4000
static float look_up_table[64];
static float* look_up_table_gpu;

struct Entries_gpu0
  {
    int imageId;
    int idxId;
    bool HMBM[64];
  };
struct inverted_gpu0
  {
    float thresholds[64];
    float idf_weight;
    int indexInmap; //    map_begin
  };
static float *mat_gpu;
static int *sum_mat_gpu,*result_gpu,*wordsID_gpu;
static inverted_gpu0 *inverted_gpu;
static Entries_gpu0 *Entries_gpu;
static int *image_index_gpu;
static float *descriptor_gpu;
static float *d_A;
static float *d_B;
static float *d_Result;

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


__global__ static void gpu_function(float * mat_g, int *sum_mat_g,int *result_g)
{    
    int indexId = blockIdx.x;
    float *mat = mat_g + indexId * DATA_SIZE * DATA_SIZE;
    int * sum_mat = sum_mat_g + indexId * DOUBLE_DATA_SIZE; 
    __shared__ int smallIndex1[256];
    __shared__ int sum_small1[256]; 
    // __shared__ int sum_mat_shared_g[DOUBLE_DATA_SIZE];
    // for(int index_query = threadIdx.x; index_query < DOUBLE_DATA_SIZE; index_query += blockDim.x )
    // {
    //     sum_mat_shared_g[index_query] = sum_mat_g[indexId * DOUBLE_DATA_SIZE + index_query];
    //     sum_mat_g[indexId * DOUBLE_DATA_SIZE + index_query] = 0;
    // }

    while(true)
    { 
        __shared__ int sum_small;
        sum_small = 4000; 
         __shared__ int smallIndex;
         smallIndex = -1;
        smallIndex1[threadIdx.x] = -1;
        sum_small1[threadIdx.x] = 4000;
        for(int index_query = threadIdx.x; index_query < DOUBLE_DATA_SIZE; index_query += blockDim.x )
        {
            if(sum_mat[index_query] > 0 && sum_mat[index_query] < sum_small1[threadIdx.x])
            {
              smallIndex1[threadIdx.x] = index_query;
              sum_small1[threadIdx.x] = sum_mat[index_query];//最小个数
            }
            // if(sum_mat[index_query] == 1)
            // {
            //     atomicExch(&sum_small, 1);
            //     atomicExch(&smallIndex, index_query);
            // }
        }
         __syncthreads();
        if(sum_small1[threadIdx.x] == 1)
        {
            atomicExch(&sum_small, 1);
            atomicExch(&smallIndex, smallIndex1[threadIdx.x]);
        }

        __syncthreads();
        if(sum_small != 1)
        {
            if(threadIdx.x<16)
            for(int index = threadIdx.x; index < blockDim.x; index += 16)
            {                
                if(sum_small1[index]>0 && sum_small1[threadIdx.x] > sum_small1[index]){
                    smallIndex1[threadIdx.x] = smallIndex1[index];
                    sum_small1[threadIdx.x] = sum_small1[index];
                }
            }
            __syncthreads();
            if(threadIdx.x==0)
            for(int index = 0; index < 16; index++)
            {
                if(sum_small1[index]>0 && sum_small > sum_small1[index])
                {
                    smallIndex = smallIndex1[index];
                    sum_small = sum_small1[index];
                }
            }
        }
        
        __syncthreads();
        if(sum_small == 4000)
        {
            return;
        }
        sum_mat[smallIndex] = 0;
        int bigDistIndex = -1;
        float distNumber = 0;
        __shared__ int bigDistI[256];
        __shared__ float distNum[256];
        bigDistI[threadIdx.x] = -1;
        distNum[threadIdx.x] = 0;
        __syncthreads();
        if(smallIndex < DATA_SIZE)
        {
            for(int index = threadIdx.x; index < DATA_SIZE; index += blockDim.x)
            {
                int temp1 = smallIndex * DATA_SIZE + index;

                if(mat[temp1] > 0)
                {
                    if(mat[temp1] > distNum[threadIdx.x])
                    {
                        bigDistI[threadIdx.x] = index;
                        distNum[threadIdx.x] = mat[temp1];
                    }
                    mat[temp1] = 0;
                    sum_mat[DATA_SIZE+index] --;
                }
            }
            __syncthreads();
            if(threadIdx.x<16)
            for(int index = threadIdx.x; index < blockDim.x ; index += 16)
            {
                if(distNum[index] > distNum[threadIdx.x])
                {
                    distNum[threadIdx.x] = distNum[index];
                    bigDistI[threadIdx.x] = bigDistI[index];
                }
            }
             __syncthreads();
            #pragma unroll
            for(int index = 0; index < 16; index++)
            {
                if(distNum[index] > distNumber)
                {
                    bigDistIndex = bigDistI[index];
                    distNumber = distNum[index];
                }
            }
            __syncthreads();

            sum_mat[bigDistIndex+DATA_SIZE] = 0;
	        result_g[indexId * DATA_SIZE + smallIndex] = bigDistIndex + 1;

            for(int indexIdx = threadIdx.x; indexIdx < DATA_SIZE; indexIdx += blockDim.x)
            {
               int temp2 = indexIdx * DATA_SIZE + bigDistIndex;
               if (mat[temp2] > 0)
               {
                 mat[temp2] = 0;
                 sum_mat[indexIdx]--;
               }
            }
        }
        else
        {

            for(int index = threadIdx.x; index < DATA_SIZE; index += blockDim.x)
            {
              int temp1 = smallIndex - DATA_SIZE + DATA_SIZE * index;

                if(mat[temp1] > 0)
                {
                    if(mat[temp1] > distNum[threadIdx.x])
                    {
                        bigDistI[threadIdx.x] = index;
                        distNum[threadIdx.x] = mat[temp1];
                    }
                    mat[temp1] = 0;
                    sum_mat[index] --;
                }
            }
            __syncthreads();
            if(threadIdx.x<16) 
            for(int index = threadIdx.x; index < blockDim.x ; index += 16)
            {
                if(distNum[index] > distNum[threadIdx.x])
                {
                    distNum[threadIdx.x] = distNum[index];
                    bigDistI[threadIdx.x] = bigDistI[index];
                }
            }
            __syncthreads();
            #pragma unroll 
            for(int index = 0; index < 16; index++)
            {
                if(distNum[index] > distNumber)
                {
                    bigDistIndex = bigDistI[index];
                    distNumber = distNum[index];
                }
            }
            __syncthreads();

            sum_mat[bigDistIndex] = 0;
	        result_g[indexId * DATA_SIZE + bigDistIndex] = smallIndex - DATA_SIZE + 1;

            for(int indexIm = threadIdx.x; indexIm < DATA_SIZE; indexIm += blockDim.x)
            {
                int temp2 = indexIm + bigDistIndex * DATA_SIZE;
                if (mat[temp2] > 0)
                {
                   mat[temp2] = 0;
                   sum_mat[indexIm + DATA_SIZE]--;
                }
            }
        }
        __syncthreads();
    }
} 

__global__ static void gpu_functionNew(float * mat_g, int *sum_mat_g,int *result_g)
{    
    int indexId = blockIdx.x;
    if(indexId>99)
    return;
    float *mat = mat_g + indexId * DATA_SIZE * DATA_SIZE;
    int * sum_mat = sum_mat_g + indexId * DOUBLE_DATA_SIZE; 
    __shared__ int smallIndex1[256];
    __shared__ int sum_small1[256]; 

    while(true)
    { 
        __shared__ int sum_small;
        sum_small = 4000; 
         __shared__ int smallIndex;
         smallIndex = -1;
        smallIndex1[threadIdx.x] = -1;
        sum_small1[threadIdx.x] = 4000;
        for(int index_query = threadIdx.x; index_query < DOUBLE_DATA_SIZE; index_query += blockDim.x )
        {
            if(sum_mat[index_query] > 0 && sum_mat[index_query] < sum_small1[threadIdx.x] && index_query < DOUBLE_DATA_SIZE)
            {
              smallIndex1[threadIdx.x] = index_query;
              sum_small1[threadIdx.x] = sum_mat[index_query];//最小个数
            }
            if(sum_mat[index_query] == 1)
            {
                sum_small = 1;
                smallIndex = index_query;
            }
        }
        __syncthreads();
        if(sum_small != 1)
        {
            if(threadIdx.x<16)
            for(int index = threadIdx.x; index < blockDim.x; index += 16)
            {
                if(sum_small1[index]>0 && sum_small1[threadIdx.x] > sum_small1[index]){
                    smallIndex1[threadIdx.x] = smallIndex1[index];
                    sum_small1[threadIdx.x] = sum_small1[index];
                }
            }
            __syncthreads();
            if(threadIdx.x==0)
            for(int index = 0; index < 16; index++)
            {
                if(sum_small1[index]>0 && sum_small > sum_small1[index])
                {
                    smallIndex = smallIndex1[index];
                    sum_small = sum_small1[index];
                }
            }
        }
        
        __syncthreads();
        if(sum_small == 4000)
           return;
        sum_mat[smallIndex] = 0;
        int bigDistIndex = -1;
        float distNumber = 0;
        __shared__ int bigDistI[256];
        __shared__ float distNum[256];
        bigDistI[threadIdx.x] = -1;
        distNum[threadIdx.x] = 0;
        __syncthreads();
        if(smallIndex < DATA_SIZE)
        {
            for(int index = threadIdx.x; index < DATA_SIZE; index += blockDim.x)
            {
                int temp1 = smallIndex * DATA_SIZE + index;

                if(mat[temp1] > 0)
                {
                    if(mat[temp1] > distNum[threadIdx.x])
                    {
                        bigDistI[threadIdx.x] = index;
                        distNum[threadIdx.x] = mat[temp1];
                    }
                    mat[temp1] = 0;
                    sum_mat[DATA_SIZE+index] --;
                }
            }
            __syncthreads();
            if(threadIdx.x<64)
            for(int index = threadIdx.x; index < blockDim.x ; index += 64)
            {
                if(distNum[index] > distNum[threadIdx.x])
                {
                    distNum[threadIdx.x] = distNum[index];
                    bigDistI[threadIdx.x] = bigDistI[index];
                }
            }
            __syncthreads();
            if(threadIdx.x<16)
            for(int index = threadIdx.x; index < 64 ; index += 16)
            {
                if(distNum[index] > distNum[threadIdx.x])
                {
                    distNum[threadIdx.x] = distNum[index];
                    bigDistI[threadIdx.x] = bigDistI[index];
                }
            }
            __syncthreads();
            if(threadIdx.x<4)
            for(int index = threadIdx.x; index < 16 ; index += 4)
            {
                if(distNum[index] > distNum[threadIdx.x])
                {
                    distNum[threadIdx.x] = distNum[index];
                    bigDistI[threadIdx.x] = bigDistI[index];
                }
            }
             __syncthreads();
            for(int index = 0; index < 4; index++)
            {
                if(distNum[index] > distNumber)
                {
                    bigDistIndex = bigDistI[index];
                    distNumber = distNum[index];
                }
            }
            __syncthreads();

            sum_mat[bigDistIndex+DATA_SIZE] = 0;
	        result_g[indexId * DATA_SIZE + smallIndex] = bigDistIndex + 1;

            for(int indexIdx = threadIdx.x; indexIdx < DATA_SIZE; indexIdx += blockDim.x)
            {
               int temp2 = indexIdx * DATA_SIZE + bigDistIndex;
               if (mat[temp2] > 0)
               {
                 mat[temp2] = 0;
                 sum_mat[indexIdx]--;
               }
            }
        }
        else
        {

            for(int index = threadIdx.x; index < DATA_SIZE; index += blockDim.x)
            {
              int temp1 = smallIndex - DATA_SIZE + DATA_SIZE * index;

                if(mat[temp1] > 0)
                {
                    if(mat[temp1] > distNum[threadIdx.x])
                    {
                        bigDistI[threadIdx.x] = index;
                        distNum[threadIdx.x] = mat[temp1];
                    }
                    mat[temp1] = 0;
                    sum_mat[index] --;
                }
            }
            __syncthreads();
            if(threadIdx.x<64) 
            for(int index = threadIdx.x; index < blockDim.x ; index += 64)
            {
                if(distNum[index] > distNum[threadIdx.x])
                {
                    distNum[threadIdx.x] = distNum[index];
                    bigDistI[threadIdx.x] = bigDistI[index];
                }
            }
            __syncthreads();
            if(threadIdx.x<16) 
            for(int index = threadIdx.x; index < 64 ; index += 16)
            {
                if(distNum[index] > distNum[threadIdx.x])
                {
                    distNum[threadIdx.x] = distNum[index];
                    bigDistI[threadIdx.x] = bigDistI[index];
                }
            }
            __syncthreads();
            if(threadIdx.x<4) 
            for(int index = threadIdx.x; index < 16 ; index += 4)
            {
                if(distNum[index] > distNum[threadIdx.x])
                {
                    distNum[threadIdx.x] = distNum[index];
                    bigDistI[threadIdx.x] = bigDistI[index];
                }
            }
            __syncthreads();
             
            for(int index = 0; index < 4; index++)
            {
                if(distNum[index] > distNumber)
                {
                    bigDistIndex = bigDistI[index];
                    distNumber = distNum[index];
                }
            }
            __syncthreads();

            sum_mat[bigDistIndex] = 0;
	        result_g[indexId * DATA_SIZE + bigDistIndex] = smallIndex - DATA_SIZE + 1;

            for(int indexIm = threadIdx.x; indexIm < DATA_SIZE; indexIm += blockDim.x)
            {
                int temp2 = indexIm + bigDistIndex * DATA_SIZE;
                if (mat[temp2] > 0)
                {
                   mat[temp2] = 0;
                   sum_mat[indexIm + DATA_SIZE]--;
                }
            }
        }
        __syncthreads();
    }

} 

void function1(int *result)
{  
    
    // cudaStream_t stream, stream1;
    // cudaStreamCreate(&stream);
    // cudaStreamCreate(&stream1);
    dim3 blockdim(256,1,1);
    dim3 griddim(64,1,1);
    float time_GPU; 
    cudaEvent_t start_GPU, stop_GPU;
    cudaEventCreate(&start_GPU); cudaEventCreate(&stop_GPU);
    cudaEventRecord(start_GPU, 0);   
    cudaMemset(result_gpu, 0, sizeof(int) * DATA_SIZE * 100);
    gpu_function<< <griddim ,blockdim,  0 >> >(mat_gpu, sum_mat_gpu, result_gpu);
    cudaEventRecord(stop_GPU, 0);
 cudaEventSynchronize(start_GPU);
 cudaEventSynchronize(stop_GPU); 
  cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU); 
 printf("The time for function1:\t%f(ms)\n", time_GPU);
 cudaEventDestroy(start_GPU);
 cudaEventDestroy(stop_GPU);  
    //gpu_function<< <griddim ,blockdim,  0 >> >(mat_gpu+DATA_SIZE*DATA_SIZE*50, sum_mat_gpu + DOUBLE_DATA_SIZE*50, result_gpu+ DATA_SIZE*50);

    //  cudaStreamSynchronize(stream);
    //  cudaStreamSynchronize(stream1);
    //  cudaStreamDestroy(stream);
    //  cudaStreamDestroy(stream1);
    
    // static float *mat_cpu = new float[DATA_SIZE*DATA_SIZE * 100];
    // static int *sum_mat_cpu = new int[DATA_SIZE*200];
    // cudaMemcpy(mat_cpu, mat_gpu, sizeof(float) * DATA_SIZE*DATA_SIZE * 100, cudaMemcpyDeviceToHost); 
    // cudaMemcpy(sum_mat_cpu, sum_mat_gpu, sizeof(int) * DATA_SIZE * 200, cudaMemcpyDeviceToHost); 
    // int a = 0;
    // int b = 0;
    // for(int i = 0;i<DATA_SIZE*DATA_SIZE * 100;i++)
    //  if(mat_cpu[i]>0)   a++;
    // for(int i = 0; i<2000*200;i++)
    //     if(sum_mat_cpu[i]!=0)
    //         b++;
    // printf("a:%d\t",a);printf("b:%d\n",b);
    cudaMemcpy(result, result_gpu, sizeof(int) * DATA_SIZE * 100, cudaMemcpyDeviceToHost); 
     
}

__global__ static void gpu_words1(int* wordsId,Entries_gpu0* Entries_gpu,
                                    inverted_gpu0* inverted_gpu,float * mat_g, 
                                    int *sum_mat_g,int * image_index,
                                    float * LUTGpu,float *descriptor,int size)
{
    for(int blockIndex = blockIdx.x;blockIndex<size;blockIndex += gridDim.x)
    {
        int word_id ;
    //= wordsId[blockIndex*blockDim.x+threadIdx.x];
        for(int i = 0; i<5; i++)
        {
        word_id = wordsId[blockIndex*5+i];
    if(word_id <1000000)
    {
        inverted_gpu0   temp = inverted_gpu[word_id]; 

        __shared__ bool des[64];

        for(int i = threadIdx.y; i<64; i += blockDim.y)
        {
            des[i] = (descriptor[blockIndex * 64 + i] > inverted_gpu[word_id].thresholds[i]);
        }

        float idf_weight = temp.idf_weight;
        float squared_idf_weight = idf_weight * idf_weight;
        int match_begin = temp.indexInmap;
        int match_end = inverted_gpu[word_id+1].indexInmap;
        __syncthreads();
//        if(match_end - match_begin > 2000 )
//        {
//            printf("run-%d\t",match_end - match_begin);
//            break;
//            match_end = match_begin +2000;
//        }

        for(int index = match_begin + threadIdx.y; index < match_end; index += blockDim.y)
        {
            Entries_gpu0 temp_entrie = Entries_gpu[index];
            if(image_index[temp_entrie.imageId] < 0)
                continue;
            size_t hamming_dist = 0;
            
            for(int j = 0; j < 64; j++)
            {
                if(des[j] ^ temp_entrie.HMBM[j])
                {
                    hamming_dist ++;//= (des[j] ^ temp_entrie.HMBM[j]);
                }
            }
            
            if (hamming_dist <= 24)
            {
                const float dist = 
                    LUTGpu[hamming_dist] * squared_idf_weight;

                int image_id = temp_entrie.imageId;
                int i = blockIndex;
                int feature_idx = temp_entrie.idxId;
                if (i < DATA_SIZE && feature_idx < DATA_SIZE)
                {
                    int index_match = image_index[image_id] * DATA_SIZE * DATA_SIZE + i * DATA_SIZE + feature_idx;

                    if (mat_g[index_match] == 0)
                    {
                        atomicAdd(&sum_mat_g[image_index[image_id] * DOUBLE_DATA_SIZE + i], 1);
                        atomicAdd(&sum_mat_g[image_index[image_id] * DOUBLE_DATA_SIZE + DATA_SIZE + feature_idx], 1);
                        mat_g[index_match] = float(dist + 1);
                    }
                    else if (mat_g[index_match] < float(dist + 1))
                    {
                        mat_g[index_match] = float(dist + 1);
                    }
                }
            }
        }
    } 
    __syncthreads();
    }
}
}
__global__ static  void gpu_words2(float * mat_g, int *sum_mat_g)
{
    int index_image = blockIdx.x;
    int sum = 0;
    for(int index_idx = threadIdx.x; index_idx<DATA_SIZE; index_idx+=blockDim.x)
    {

        for(int index = 0; index < DATA_SIZE; index++)
        if(mat_g[index_image*DATA_SIZE*DATA_SIZE + index_idx*DATA_SIZE + index])
            sum++;
        sum_mat_g[index_image*DOUBLE_DATA_SIZE + index_idx] = sum;
        sum = 0;
    }
    for(int index_quere = threadIdx.x; index_quere<DATA_SIZE; index_quere+=blockDim.x)
    {
        for(int index = 0; index < DATA_SIZE; index++)
            if(mat_g[index_image*DATA_SIZE*DATA_SIZE + index_quere + DATA_SIZE*index])
                sum++;
        sum_mat_g[index_image*DOUBLE_DATA_SIZE + DATA_SIZE + index_quere] = sum;
        sum = 0;
    }
}
void function_words(int * imageIndex, float *descriptor, int * wordsId, int size)
{
     float time_GPU; 
     cudaEvent_t start_GPU, stop_GPU;
   cudaEventCreate(&start_GPU); cudaEventCreate(&stop_GPU);
   cudaEventRecord(start_GPU, 0);
//    cudaStream_t stream, stream1;
//    cudaStreamCreate(&stream);
//    cudaStreamCreate(&stream1);
    dim3 blockdim(1,64,1);
    int size_mall;
    if(size >DATA_SIZE)
        size_mall = DATA_SIZE;
    else
        size_mall = size;
    dim3 griddim(size_mall,1,1);
    cudaMemcpy(image_index_gpu, imageIndex, sizeof(int)*10000, cudaMemcpyHostToDevice);
    cudaMemcpy(wordsID_gpu, wordsId, sizeof(int)*size_mall*5, cudaMemcpyHostToDevice);
   
    gpu_words1<< <griddim ,blockdim,  0 >> >(wordsID_gpu,Entries_gpu,
                                    inverted_gpu,mat_gpu, sum_mat_gpu,
                                    image_index_gpu,look_up_table_gpu,descriptor_gpu,size_mall);
    //gpu_words2<<<100,size/4+1>>>(mat_gpu,sum_mat_gpu);

    // gpu_words1<< <griddim ,blockdim,  0 ,stream1 >> >(wordsID_gpu + 500,Entries_gpu,
    //                                 inverted_gpu,mat_gpu, sum_mat_gpu,
    //                                 image_index_gpu,look_up_table_gpu,descriptor_gpu,size_mall - 500);                                
    // cudaStreamSynchronize(stream);
    // cudaStreamSynchronize(stream1);
    // cudaStreamDestroy(stream);
	// cudaStreamDestroy(stream1);
                                    cudaEventRecord(stop_GPU, 0);
                                    cudaEventSynchronize(start_GPU);
                                    cudaEventSynchronize(stop_GPU); 
                                     cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU); 
                                    printf("The time for function_words memcpy:\t%f(ms)\n", time_GPU);
                                    cudaEventDestroy(start_GPU);
                                    cudaEventDestroy(stop_GPU);
}

__global__ static void Multiplication(float * d_A, float * d_B, float *result, int B)
{
    float sum = 0;
    
    for(int i = 0;i<B;i++)
    {
        sum += d_A[blockIdx.x*B+i]*d_B[i*blockDim.x+threadIdx.x];
    }
    result[blockIdx.x*blockDim.x+threadIdx.x] = sum;
}

void function_descriptors(const float *h_A, float *h_B, float *result,int A, int B, int C)
{
    float time_GPU; 
     cudaEvent_t start_GPU, stop_GPU;
   cudaEventCreate(&start_GPU); cudaEventCreate(&stop_GPU);
   cudaEventRecord(start_GPU, 0);
    cudaMemcpy(d_A, h_A, sizeof(float) * A * B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeof(float) * B * C, cudaMemcpyHostToDevice);
    Multiplication<<<A,C,0>>>(d_A,d_B,descriptor_gpu,B);
    cudaEventRecord(stop_GPU, 0);
    cudaEventSynchronize(start_GPU);
    cudaEventSynchronize(stop_GPU); 
     cudaEventElapsedTime(&time_GPU, start_GPU, stop_GPU); 
    printf("The time for function_descriptors memcpy:\t%f(ms)\n", time_GPU);
    cudaEventDestroy(start_GPU);
    cudaEventDestroy(stop_GPU);

}
void gpuInit(char* inverted,char* Entries)
{
    if (!InitCUDA()) {
	printf("initcuda error");
        return;
    }


    for(int i = 0; i < 64; i++)
    {
        look_up_table[i] = std::exp(-1.0 * i * i / (24.0*24.0));
    }
    cudaMalloc((void**)&d_A, sizeof(float) * DATA_SIZE * 128);
    cudaMalloc((void**)&d_B, sizeof(float) * 128*64);//邻接矩阵
    cudaMalloc((void**)&d_Result, sizeof(int) * DATA_SIZE * 64);//单对多个数


    cudaMalloc((void**)&result_gpu, sizeof(int) * DATA_SIZE * 100);
    cudaMalloc((void**)&mat_gpu, sizeof(float) * DATA_SIZE*DATA_SIZE * 100);//邻接矩阵
    cudaMalloc((void**)&sum_mat_gpu, sizeof(int) * DOUBLE_DATA_SIZE * 100);//单对多个数
    cudaMalloc((void**)&descriptor_gpu, sizeof(float) * DATA_SIZE*64);
    cudaMalloc((void**)&image_index_gpu, sizeof(int) * 10000);
    cudaMalloc((void**)&look_up_table_gpu, sizeof(float) * 64);
    cudaMalloc((void**)&wordsID_gpu, sizeof(int) * DATA_SIZE * 5 );
    cudaMalloc((void**)&inverted_gpu, sizeof(inverted_gpu0) * 32768);
    cudaMalloc((void**)&Entries_gpu, sizeof(Entries_gpu0) * 2000*2000);//邻接矩阵
    

    cudaMemcpy(look_up_table_gpu, look_up_table, sizeof(float)*64, cudaMemcpyHostToDevice);
    cudaMemcpy(inverted_gpu, inverted, sizeof(inverted_gpu0) * 32768, cudaMemcpyHostToDevice);
    cudaMemcpy(Entries_gpu, Entries, sizeof(Entries_gpu0) * 2000*2000, cudaMemcpyHostToDevice);
    cudaMemset(mat_gpu, 0, sizeof(float) * DATA_SIZE*DATA_SIZE * 100);
    cudaMemset(sum_mat_gpu, 0, sizeof(int) * DOUBLE_DATA_SIZE * 100);
}

void gpuRelese()
{
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_Result);
    cudaFree(wordsID_gpu);
    cudaFree(inverted_gpu);
    cudaFree(Entries_gpu);
    cudaFree(mat_gpu);
    cudaFree(sum_mat_gpu);
    cudaFree(result_gpu);

}


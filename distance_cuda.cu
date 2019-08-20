
#include "feature/distance_cuda.h"
#include "util/logging.h"

#include <stdio.h>
#include <malloc.h>
#include <random>
#include <time.h>
#include <math.h>


__global__ void gpuDeepDistanceMatrixDevice(const float *a, const float *b, float *result, const int feature_len, const int M, const int N)
{
	//int threadId = threadIdx.x + blockIdx.x * blockDim.x;

	// int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x + blockIdx.x * blockDim.x + threadIdx.x;
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (threadId < M * N)
	{
		int row = threadId / N;
		int col = threadId % N;

        result[threadId] = 0;
        for (int i = 0; i < feature_len; i++)
        {
            result[threadId] += pow((a[row * feature_len + i] - b[col * feature_len + i]), 2);
        }
		result[threadId] = sqrt(result[threadId]);
	}
}


__global__ void gpuDeepDistanceMatrixDevice_opt(const float *a, const float *b, float *result, const int feature_len, const int M, const int N)
{
	int threadId = (blockIdx.x * blockDim.x) + threadIdx.x;
	if (threadId < M * N)
	{
		int row = threadId / N;
		int col = threadId % N;

		float result_sum = 0.0f;
		float4 *left = (float4 *)(a + row * feature_len);
		float4 *right = (float4 *)(b + col * feature_len);
		int size = feature_len / 4;
		#pragma unroll
        for (int i = 0; i < size; i++)
        {
			float diff_x = left[i].x - right[i].x;
			float diff_y = left[i].y - right[i].y;
			float diff_z = left[i].z - right[i].z;
			float diff_w = left[i].w - right[i].w;
			result_sum += (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w);
        }
		result[threadId] = sqrtf(result_sum);
	}
}


__global__ void gpuDeepDistanceMatrixDevice_opt_new(const float *a, const float *b, float *result, const int feature_len, const int M, const int N, int pitch)
{
	int threadId = (4 * threadIdx.x);
	float4 temp_result;

	if ((threadId + 4) <= N)
	{
		int row = blockIdx.x;
		int col = threadId;

		float result_sum = 0.0f;
		float4 *left = (float4 *)(a + row * feature_len);
		float4 *right = (float4 *)(b + col * feature_len);
		int size = feature_len / 4;
		#pragma unroll
        for (int i = 0; i < size; i++)
        {
			float diff_x = left[i].x - right[i].x;
			float diff_y = left[i].y - right[i].y;
			float diff_z = left[i].z - right[i].z;
			float diff_w = left[i].w - right[i].w;
			result_sum += (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w);			
		}
		temp_result.x = sqrtf(result_sum);
		
		col = threadId + 1;
		result_sum = 0.0f;
		right = (float4 *)(b + col * feature_len);
		#pragma unroll
        for (int i = 0; i < size; i++)
        {
			float diff_x = left[i].x - right[i].x;
			float diff_y = left[i].y - right[i].y;
			float diff_z = left[i].z - right[i].z;
			float diff_w = left[i].w - right[i].w;
			result_sum += (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w);
		}
		temp_result.y = sqrtf(result_sum);

		col = threadId + 2;
		result_sum = 0.0f;
		right = (float4 *)(b + col * feature_len);
		#pragma unroll
        for (int i = 0; i < size; i++)
        {
			float diff_x = left[i].x - right[i].x;
			float diff_y = left[i].y - right[i].y;
			float diff_z = left[i].z - right[i].z;
			float diff_w = left[i].w - right[i].w;
			result_sum += (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w);
		}
		temp_result.z = sqrtf(result_sum);

		col = threadId + 3;
		result_sum = 0.0f;
		right = (float4 *)(b + col * feature_len);
		#pragma unroll
        for (int i = 0; i < size; i++)
        {
			float diff_x = left[i].x - right[i].x;
			float diff_y = left[i].y - right[i].y;
			float diff_z = left[i].z - right[i].z;
			float diff_w = left[i].w - right[i].w;
			result_sum += (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w);
		}
		temp_result.w = sqrtf(result_sum);

		*(float4*)(&result[row * pitch + threadId]) = temp_result;
	}
	if((threadId + 8) > N)
	{
		for(int i = threadId + 4; i < N; i++)
		{
			int row = blockIdx.x;
			int col = i;
	
			float result_sum = 0.0f;
			float4 *left = (float4 *)(a + row * feature_len);
			float4 *right = (float4 *)(b + col * feature_len);
			int size = feature_len / 4;
			#pragma unroll
			for (int j = 0; j < size; j++)
			{
				float diff_x = left[j].x - right[j].x;
				float diff_y = left[j].y - right[j].y;
				float diff_z = left[j].z - right[j].z;
				float diff_w = left[j].w - right[j].w;
				result_sum += (diff_x * diff_x + diff_y * diff_y + diff_z * diff_z + diff_w * diff_w);
			}
			result[row * pitch + i] = sqrtf(result_sum);
		}
	}

}


cudaError_t gpuDeepDistanceMatrix(const float *a, const float *b, float *result, const int feature_len, const int M, const int N)
{
	float *dev_a = 0;
	float *dev_b = 0;
	float *dev_result = 0;

	cudaError_t cudaStatus;

	cudaStatus = cudaSetDevice(0);
	CHECK_EQ(cudaStatus, cudaSuccess);

	cudaStatus = cudaMalloc((void **)&dev_a, M * feature_len * sizeof(float));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy1 : %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }

	cudaStatus = cudaMalloc((void **)&dev_b, N * feature_len * sizeof(float));
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy2 : %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }

	//cudaStatus = cudaMalloc((void **)&dev_result, M * N * sizeof(float));
	size_t pitch;
	cudaStatus = cudaMallocPitch((void **)&dev_result, &pitch, N * sizeof(float), M);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy3 : %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }

	cudaStatus = cudaMemcpy(dev_a, a, M * feature_len * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy4 : %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }

	cudaStatus = cudaMemcpy(dev_b, b, N * feature_len * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy5 : %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }

	cudaEvent_t gpuStart, gpuFinish;
	float elapsedTime;
	cudaEventCreate(&gpuStart);
	cudaEventCreate(&gpuFinish);
	cudaEventRecord(gpuStart, 0);

	#if 0
	dim3 block(512);
	dim3 grid((M * N + 511) / 512);
	gpuDeepDistanceMatrixDevice_opt <<<grid, block >>>(dev_a, dev_b, dev_result, feature_len, M, N);
	#else
	dim3 block((N + 3) / 4);
	dim3 grid(M);
	gpuDeepDistanceMatrixDevice_opt_new<<<grid, block >>>(dev_a, dev_b, dev_result, feature_len, M, N, pitch/4);
	#endif

	cudaEventRecord(gpuFinish, 0);
	cudaEventSynchronize(gpuFinish);
	cudaEventElapsedTime(&elapsedTime, gpuStart, gpuFinish);
	// printf("\nThe runing time of GPU on Mat Multiply is %f seconds.\n", elapsedTime / 1000.0);


	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy7 : %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }

	//cudaStatus = cudaMemcpy(result, dev_result, M * N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy2D(result, N * sizeof(float), dev_result, pitch, N * sizeof(float), M, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy8 : %s\n", cudaGetErrorString(cudaGetLastError()));
        exit(EXIT_FAILURE);
    }

	cudaFree(dev_result);
    cudaFree(dev_a);
    cudaFree(dev_b);

	return cudaStatus;
}

#ifndef _DISTANCE_GPU_H_
#define _DISTANCE_GPU_H_

#include<cuda_runtime.h>
#include<device_launch_parameters.h>
extern "C" void initDist(int D,int N) ; //D:向量长度;N矩阵行数，结果向量的长度
extern "C" void destoryDist();
extern "C" void computeDist(float* vec_A,float* mat_B, float* dist);
extern "C" void computeDist1(float* vec_A,float* mat_B, float* dist);

#endif

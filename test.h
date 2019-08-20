#ifndef _TEST_H_
#define _TEST_H_
extern void function_descriptors(const float *h_A, float *h_B, float *result,int A, int B, int C);
extern void function_words(int * imageIndex, float *descriptor, int * wordsId, int size);
extern void function1(int *result_g);
extern void gpuInit(char*,char*);
extern void gpuRelese();
#endif

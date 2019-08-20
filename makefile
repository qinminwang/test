objects = cudaTest21.o cudaTest22.o cudaTest23.o
result:$(objects)
	g++ -g -o result $(objects)  -L/usr/local/cuda-9.0/lib64/ -lcudart -lcublas -L/home/wqm/cudaTest -ldistance_gpu
cudaTest21.o:cudaTest21.cpp
	g++ -g -c -I/usr/local/cuda-9.0/include -I/home/wqm/cudaTest cudaTest21.cpp
cudaTest22.o:cudaTest22.cpp
	g++ -g -c -I/usr/local/cuda-9.0/include cudaTest22.cpp	
cudaTest23.o:cudaTest23.cu
	nvcc -g -c cudaTest23.cu
.PHONY:clean

clean:
	-rm result $(objects)

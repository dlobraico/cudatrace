#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>

void render1(int xsz, int ysz, int *host_fb, int samples);
__global__ void render2(int *device_fb, int samples, int xsz, int ysz);

#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }

void cudaAssert(const cudaError err, const char *file, const int line)
{ 
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}

int main(int argc, char **argv) {
    int xres = 256;
    int yres = 256;
    int samples = 1;
    int *array;
    //int array[xres][yres];

    if(!(array = (int *)malloc(xres * yres * sizeof(int)))) {
        perror("pixel buffer allocation failed");
        return EXIT_FAILURE;
    }

    render1(xres, yres, array, samples);

    for(int i = 0; i<yres; i++) {
        for(int j = 0; j<yres; j++) {
            printf("%i, %i: %i\n", i, j, array[i + sizeof(int)*j]);
        }
    }

    free(array);

    return 0;
}

void render1(int xsz, int ysz, int *host_fb, int samples) {
    dim3 threads_per_block(16, 16);

    int whole_blocks_x = xsz/threads_per_block.x;
    int whole_blocks_y = ysz/threads_per_block.y;

    int remainder_threads_x = xsz % threads_per_block.x;
    int remainder_threads_y = ysz % threads_per_block.y;

    int extra_block_x = 0;
    int extra_block_y = 0;

    if (remainder_threads_x > 0) {
        extra_block_x = 1;
    }

    if (remainder_threads_y > 0) {
        extra_block_y = 1;
    }


    int num_blocks_x = whole_blocks_x + extra_block_x;
    int num_blocks_y = whole_blocks_y + extra_block_y;

    printf("num_blocks_x: %i\n", num_blocks_x);
    printf("num_blocks_y: %i\n", num_blocks_y);

    dim3 num_blocks(num_blocks_x, num_blocks_y);
    
    size_t arr_size = xsz * ysz * sizeof(int);
    printf("xsz: %i\n", xsz);
    printf("ysz: %i\n", ysz);
    printf("arr_size: %zi\n", arr_size);

    int *device_fb = 0;
    cudaErrorCheck(cudaMalloc((void **)&device_fb, arr_size));
    cudaErrorCheck(cudaMemcpy(device_fb, host_fb, arr_size, cudaMemcpyHostToDevice));

    render2<<<num_blocks,threads_per_block>>>(device_fb, samples,  xsz, ysz);
    cudaPeekAtLastError(); // Checks for launch error
    cudaErrorCheck( cudaThreadSynchronize() );

    cudaErrorCheck(cudaMemcpy(host_fb, device_fb, arr_size, cudaMemcpyDeviceToHost));
    cudaErrorCheck( cudaFree(device_fb) );
}

__global__ void render2(int *device_fb, int samples, int xsz, int ysz)   {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i >= xsz) || (j >= ysz)) {
        return;
    } else {
        device_fb[i + j*sizeof(int)] = i*j;
        return;
    }
}

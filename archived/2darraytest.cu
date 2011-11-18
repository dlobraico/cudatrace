#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>

/* find the appropriate way to define explicitly sized types */
#if (__STDC_VERSION__ >= 199900) || defined(__GLIBC__)  /* C99 or GNU libc */
#include <stdint.h>
#elif defined(__unix__) || defined(unix) || defined(__MACH__)
#include <sys/types.h>
#elif defined(_MSC_VER) /* the nameless one */
typedef unsigned __int8 uint8_t;
typedef unsigned __int32 uint32_t;
#endif

void render1(int xsz, int ysz, u_int32_t *host_fb, int samples);
__global__ void render2(u_int32_t *device_fb, int samples, int xsz, int ysz);

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
    int xres = 100;
    int yres = 100;
    int samples = 1;
    u_int32_t *array;

    if(!(array = (u_int32_t *)malloc(xres * yres * sizeof(u_int32_t)))) {
        perror("pixel buffer allocation failed");
        return EXIT_FAILURE;
    }

    render1(xres, yres, array, samples);

    for(int i = 0; i<xres; i++) {
        for(int j = 0; j<yres; j++) {
            printf("%i, %i: %i\n", i, j, array[i + sizeof(u_int32_t)*j]);
        }
    }

    free(array);

    return 0;
}

void render1(int xsz, int ysz, u_int32_t *host_fb, int samples) {

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

    dim3 num_blocks(num_blocks_x, num_blocks_y);
    
    size_t arr_size = xsz * ysz * sizeof(u_int32_t);

    u_int32_t *device_fb = 0;
    cudaErrorCheck(cudaMalloc((void **)&device_fb, arr_size));
    cudaErrorCheck(cudaMemcpy(device_fb, host_fb, arr_size, cudaMemcpyHostToDevice));

    render2<<<num_blocks,threads_per_block>>>(device_fb, samples,  xsz, ysz);
    cudaPeekAtLastError(); // Checks for launch error
    cudaErrorCheck( cudaThreadSynchronize() );

    cudaErrorCheck(cudaMemcpy(host_fb, device_fb, arr_size, cudaMemcpyDeviceToHost));
    cudaErrorCheck( cudaFree(device_fb) );
}

__global__ void render2(u_int32_t *device_fb, int samples, int xsz, int ysz)   {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i > xsz) || (j > ysz)) {
        return;
    } else {
        device_fb[i + j*sizeof(u_int32_t)] = i*j;
        return;
    }
}

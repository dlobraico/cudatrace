#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#include <errno.h>

unsigned long get_msec(void);

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
    //int *x;
    unsigned long start_time = get_msec();
    //cudaErrorCheck(cudaMalloc((void **)&x, 128));
    cudaErrorCheck( cudaFree(0) );
    unsigned long total_time = get_msec() - start_time;

    printf("total_time: %lu", total_time);
}

#if defined(__unix__) || defined(unix) || defined(__MACH__)
#include <time.h>
#include <sys/time.h>
unsigned long get_msec(void) {
    static struct timeval timeval, first_timeval;

    gettimeofday(&timeval, 0);
    if(first_timeval.tv_sec == 0) {
        first_timeval = timeval;
        return 0;
    }
    return (timeval.tv_sec - first_timeval.tv_sec) * 1000 + (timeval.tv_usec - first_timeval.tv_usec) / 1000;
}
#elif defined(__WIN32__) || defined(WIN32)
#include <windows.h>
unsigned long get_msec(void) {
    return GetTickCount();
}
#else
#error "I don't know how to measure time on your platform"
#endif



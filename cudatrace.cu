/* c-ray-f - a simple raytracing filter.
 * Copyright (C) 2006 John Tsiombikas <nuclear@siggraph.org>
 * No copyright for the non-functional additions added afterward..
 *
 * You are free to use, modify and redistribute this program under the
 * terms of the GNU General Public License v2 or (at your option) later.
 * see "http://www.gnu.org/licenses/gpl.txt" for details.
 * ---------------------------------------------------------------------
 * Usage:
 *   compile:  cc -o c-ray-f c-ray-f.c -lm
 *   run:      cat scene | ./c-ray-f >foo.ppm
 *   enjoy:    display foo.ppm (with imagemagick)
 *      or:    imgview foo.ppm (on IRIX)
 * ---------------------------------------------------------------------
 * Scene file format:
 *   # sphere (many)
 *   s  x y z  rad   r g b   shininess   reflectivity
 *   # light (many)
 *   l  x y z
 *   # camera (one)
 *   c  x y z  fov   tx ty tz
 * ---------------------------------------------------------------------
 */
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

#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }

void cudaAssert(const cudaError err, const char *file, const int line)
{ 
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}

int objCounter = 0;

struct validsphere {
    double sph[9];
    int null;
};


struct vec3 {
    double x;
    double y; 
    double z;
};

struct ray {
    struct vec3 orig, dir;
};

struct material {
    struct vec3 col;    /* color */
    double spow;        /* specular power */
    double refl;        /* reflection intensity */
};

struct sphere {
    struct vec3 pos;
    double rad;
    struct material mat;
    struct sphere *next;
};

struct reflectdata {  //STRUCT WHICH CONTAINS THE DATA FOR TRACING FURTHER REFLECTION RAYS
    struct ray ray;
    int depth;
    double reflection;
};  

struct spoint {
    struct vec3 pos, normal, vref;  /* position, normal and view reflection */
    double dist;        /* parametric distance of intersection along the ray */
};

struct camera {
    struct vec3 pos, targ;
    double fov;
};

void render1(int xsz, int ysz, u_int32_t **fb, int samples);
__global__ void render2(u_int32_t **device_fb, int samples, double *obj_list_flat, int lnumdev, struct camera camdev, struct vec3 *lightsdev, struct vec3 *uranddev, int *iranddev, int *objCounterdev, int xsz, int ysz); //SPECIFY ARGUMENTS TO RENDER2~!!!!
__device__ struct vec3 trace(struct ray ray, int depth, int *isReflect, struct reflectdata *Rdata, double *obj_list_flat, int lnumdev, struct vec3 *lightsdev, int *objCounterdev); //two arguments added - one to check if a reflection ray must be made, the other to provide the arguments necessary for the reflection ray
__device__ struct vec3 shade(struct validsphere obj, struct spoint *sp, int depth, int *isReflect, struct reflectdata *Rdata, double *obj_list_flat, int lnumdev, struct vec3 *lightsdev);
__device__ struct vec3 reflect(struct vec3 v, struct vec3 n);
__device__ struct vec3 cross_product(struct vec3 v1, struct vec3 v2);
__device__ struct ray get_primary_ray(int x, int y, int sample, struct camera camdev, struct vec3 *uranddev, int *iranddev);
__device__ struct vec3 get_sample_pos(int x, int y, int sample, struct vec3 *uranddev, int *iranddev);
__device__ struct vec3 jitter(int x, int y, int s, struct vec3 *uranddev, int *iranddev);
__device__ int ray_sphere(double *sph, struct ray ray, struct spoint *sp);
__device__ void get_ith_sphere(double *obj_list_flat, int index, double *sphere_flat);
void load_scene(FILE *fp);                //FOR NOW, THIS WILL NOT BE MADE PARALLEL
void flatten_obj_list(struct sphere *obj_list, double *obj_list_flat, int objCounter);
void flatten_sphere(struct sphere *sphere, double *sphere_flat);
unsigned long get_msec(void);             //COUNTING TIME CANNOT BE DONE IN PARALLEL
inline void check_cuda_errors(const char *filename, const int line_number);

#define MAX_LIGHTS      16              /* maximum number of lights */
#define RAY_MAG         1000.0          /* trace rays of this magnitude */
#define MAX_RAY_DEPTH   5               /* raytrace recursion limit */
#define FOV             0.78539816      /* field of view in rads (pi/4) */
#define HALF_FOV        (FOV * 0.5)
#define ERR_MARGIN      1e-6            /* an arbitrary error margin to avoid surface acne */

/* bit-shift ammount for packing each color into a 32bit uint */
#ifdef LITTLE_ENDIAN
#define RSHIFT  16
#define BSHIFT  0
#else   /* big endian */
#define RSHIFT  0
#define BSHIFT  16
#endif  /* endianess */
#define GSHIFT  8   /* this is the same in both byte orders */

/* some helpful macros... */
#define SQ(x)       ((x) * (x))
#define MAX(a, b)   ((a) > (b) ? (a) : (b))
#define MIN(a, b)   ((a) < (b) ? (a) : (b))
#define DOT(a, b)   ((a).x * (b).x + (a).y * (b).y + (a).z * (b).z)
#define NORMALIZE(a)  do {\
    double len = sqrt(DOT(a, a));\
    (a).x /= len; (a).y /= len; (a).z /= len;\
} while(0);

/* global state for host*/
int xres = 800;
int yres = 600;
double aspect = 1.333333;
struct sphere *obj_list;
double *obj_list_flat;
struct vec3 lights[MAX_LIGHTS];
int lnum = 0;
struct camera cam;

#define NRAN    1024
#define MASK    (NRAN - 1)
struct vec3 urand[NRAN];
int irand[NRAN];


/*global state for device*/
__device__ int xresdev = 800;
__device__ int yresdev = 600;
__device__ double aspectdev = 1.333333;
//__device__ struct sphere *obj_listdev = 0;
//__device__ struct vec3 lightsdev[MAX_LIGHTS];
//__device__ int lnumdev = 0;
//__device__ struct camera camdev;

//__device__ struct vec3 uranddev[NRAN];
//__device__ int iranddev[NRAN];


const char *usage = {
    "Usage: c-ray-f [options]\n"
        "  Reads a scene file from stdin, writes the image to stdout, and stats to stderr.\n\n"
        "Options:\n"
        "  -s WxH     where W is the width and H the height of the image\n"
        "  -r <rays>  shoot <rays> rays per pixel (antialiasing)\n"
        "  -i <file>  read from <file> instead of stdin\n"
        "  -o <file>  write to <file> instead of stdout\n"
        "  -h         this help screen\n\n"
};



int main(int argc, char **argv) {
    int i, j;
    unsigned long rend_time, start_time;
    u_int32_t **pixels;
    int rays_per_pixel = 1;
    FILE *infile = stdin, *outfile = stdout;

    for(i=1; i<argc; i++) {
        if(argv[i][0] == '-' && argv[i][2] == 0) {
            char *sep;
            switch(argv[i][1]) {
                case 's':
                    if(!isdigit(argv[++i][0]) || !(sep = strchr(argv[i], 'x')) || !isdigit(*(sep + 1))) {
                        fputs("-s must be followed by something like \"640x480\"\n", stderr);
                        return EXIT_FAILURE;
                    }
                    xres = atoi(argv[i]);
                    yres = atoi(sep + 1);
                    aspect = (double)xres / (double)yres;
                    break;

                case 'i':
                    if(!(infile = fopen(argv[++i], "r"))) {
                        fprintf(stderr, "failed to open input file %s: %s\n", argv[i], strerror(errno));
                        return EXIT_FAILURE;
                    }
                    break;

                case 'o':
                    if(!(outfile = fopen(argv[++i], "w"))) {
                        fprintf(stderr, "failed to open output file %s: %s\n", argv[i], strerror(errno));
                        return EXIT_FAILURE;
                    }
                    break;

                case 'r':
                    if(!isdigit(argv[++i][0])) {
                        fputs("-r must be followed by a number (rays per pixel)\n", stderr);
                        return EXIT_FAILURE;
                    }
                    rays_per_pixel = atoi(argv[i]);
                    break;

                case 'h':
                    fputs(usage, stdout);
                    return 0;

                default:
                    fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
                    fputs(usage, stderr);
                    return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "unrecognized argument: %s\n", argv[i]);
            fputs(usage, stderr);
            return EXIT_FAILURE;
        }
    }

    if(!(pixels = (u_int32_t **)malloc(xres * yres * sizeof(u_int32_t)))) {
        perror("pixel buffer allocation failed");
        return EXIT_FAILURE;
    }
    load_scene(infile);

    /* initialize the random number tables for the jitter */
    for(i=0; i<NRAN; i++) urand[i].x = (double)rand() / RAND_MAX - 0.5;
    for(i=0; i<NRAN; i++) urand[i].y = (double)rand() / RAND_MAX - 0.5;
    for(i=0; i<NRAN; i++) irand[i] = (int)(NRAN * ((double)rand() / RAND_MAX));

    xresdev = xres; 
    yresdev = yres;
    aspectdev = aspect;

    start_time = get_msec();
    render1(xres, yres, pixels, rays_per_pixel);
    rend_time = get_msec() - start_time;

    /* output statistics to stderr */
    fprintf(stderr, "Rendering took: %lu seconds (%lu milliseconds)\n", rend_time / 1000, rend_time);

    /* output the image */
    fprintf(outfile, "P6\n%d %d\n255\n", xres, yres);
    for(i=0; i<yres; i++) {
        for(j=0; j<xres; j++) {
            //fputc((pixels[i][j] >> RSHIFT) & 0xff, outfile);
            //fputc((pixels[i][j] >> GSHIFT) & 0xff, outfile);
            //fputc((pixels[i][j] >> BSHIFT) & 0xff, outfile);
        }
    }
    fflush(outfile);

    if(infile != stdin) fclose(infile);
    if(outfile != stdout) fclose(outfile);

    free(pixels);

    return 0;
}

/* render a frame of xsz/ysz dimensions into the provided framebuffer */
void render1(int xsz, int ysz, u_int32_t **host_fb, int samples)
{
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
    
    u_int32_t **device_fb = 0;
    //u_int32_t **host_fb = 0;
    size_t arr_size = xsz * ysz * sizeof(u_int32_t*);

    cudaErrorCheck(cudaMalloc((void **)&device_fb, arr_size));
    host_fb = (u_int32_t **)malloc(arr_size);

    cudaErrorCheck(cudaMemcpy(device_fb, host_fb, (xsz*ysz*sizeof(u_int32_t*)), cudaMemcpyHostToDevice));

    double *obj_list_flat_dev = 0;
    obj_list_flat = (double *)malloc(sizeof(double)*objCounter*9);
    flatten_obj_list(obj_list,obj_list_flat,objCounter);

    //create obj_list_flat_dev array size of objCounter
    cudaErrorCheck(cudaMalloc((void **)&obj_list_flat_dev, (sizeof(double)*objCounter*9)) );
    cudaErrorCheck(cudaMemcpy(obj_list_flat_dev, obj_list_flat, (sizeof(double)*objCounter*9), cudaMemcpyHostToDevice)); //copying over flat sphere array to obj_listdevflat

    int *objCounterdev = 0;
    cudaErrorCheck(cudaMalloc((void**)&objCounterdev, sizeof(int)));
    cudaErrorCheck( cudaMemcpy(objCounterdev, &objCounter, sizeof(int), cudaMemcpyHostToDevice) );

    //lights and camera and whatnot
    int lnumdev = 0;
    struct camera camdev;
    struct vec3 *lightsdev = 0;

    cudaErrorCheck(cudaMalloc((void **)&lightsdev, MAX_LIGHTS*sizeof(struct vec3)) );
    cudaErrorCheck(cudaMemcpy(lightsdev, lights, MAX_LIGHTS*sizeof(struct vec3), cudaMemcpyHostToDevice));

    lnumdev = lnum; //remember to pass lnumdev into render2!
    camdev = cam;   //remember to pass camdev into render2!

    //urand and whatnot
    struct vec3 *uranddev = 0;
    cudaErrorCheck(cudaMalloc((void **)&uranddev, NRAN*sizeof(struct vec3)) );
    cudaErrorCheck(cudaMemcpy(uranddev, urand, sizeof(struct vec3) * NRAN, cudaMemcpyHostToDevice)); //remember to pass all of these into render2!!

    //irand and whatnot
    int *iranddev = 0;
    cudaErrorCheck(cudaMalloc((void **)&iranddev, NRAN*sizeof(int)) );
    cudaErrorCheck(cudaMemcpy(iranddev, irand, sizeof(int) * NRAN, cudaMemcpyHostToDevice)); //remember to pass all of these into render2!!

    // KERNEL CALL!
    render2<<<num_blocks,threads_per_block>>>(device_fb, samples, obj_list_flat_dev, lnumdev, camdev, lightsdev, uranddev, iranddev, objCounterdev, xsz, ysz);
    cudaPeekAtLastError(); // Checks for launch error
    cudaErrorCheck( cudaThreadSynchronize() );

    //In all seriousness, all of the cores should now be operating on the ray tracing, if things are working correctly 
    //once done, copy contents of device array to host array  

    cudaErrorCheck(cudaMemcpy(host_fb, device_fb, xsz*ysz*sizeof(u_int32_t*), cudaMemcpyDeviceToHost));
    cudaErrorCheck(cudaMemcpy(lights, lightsdev, sizeof(struct vec3) * MAX_LIGHTS, cudaMemcpyDeviceToHost));

    printf("host_fb[0][0] = %u\n", host_fb[0][0]);

    free(obj_list_flat);
    cudaErrorCheck( cudaFree(lightsdev) );
    cudaErrorCheck( cudaFree(uranddev) );
    cudaErrorCheck( cudaFree(iranddev) );
    cudaErrorCheck( cudaFree(device_fb) );
    cudaErrorCheck( cudaFree(obj_list_flat_dev) );
}   


__global__ void render2(u_int32_t **device_fb, int samples, double *obj_list_flat_dev, int lnumdev, struct camera camdev, struct vec3 *lightsdev, struct vec3 *uranddev, int *iranddev, int *objCounterdev, int xsz, int ysz)            //SPECIFY ARGUMENTS!!!
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if ((i >= xsz) || (j >= ysz)) {
        return;
    } else {

    int isReflect[1];                        //WHETHER OR NOT RAY TRACED WILL NEED A REFLECTION RAY AS WELL
    isReflect[0] = 0;
    struct reflectdata RData[1];           //ARRAY WHICH CONTAINS REFLECT DATA STRUCT TO BE PASSED ON TO TRACE FUNCTION

    double rcp_samples = 1.0 / (double)samples;

    // for each subpixel, trace a ray through the scene, accumulate the
    // colors of the subpixels of each pixel, then pack the color and
    // put it into the framebuffer.
    // XXX: assumes contiguous scanlines with NO padding, and 32bit pixels. 

    double r, g, b;
    r = g = b = 0.0;

    for(int s = 0; s<samples; s++) {
        struct vec3 col = trace(get_primary_ray(i, j, s, camdev, uranddev, iranddev), 0, isReflect, RData, obj_list_flat_dev, lnumdev, lightsdev, objCounterdev);
        //printf("trace success!\n");	
        while (*isReflect)        //while there are still reflection rays to trace
        {
            struct vec3 rcol;    //holds the output of the reflection ray calculcation
            rcol = trace(RData->ray, RData->depth, isReflect, RData, obj_list_flat_dev, lnumdev, lightsdev, objCounterdev);    //trace a reflection ray
            col.x += rcol.x * RData->reflection;       //I really am unsure about the usage of pointers here..
            col.y += rcol.y * RData->reflection;
            col.z += rcol.z * RData->reflection;
        }   
        r += col.x;
        g += col.y;
        b += col.z;
    }

    r = r * rcp_samples;
    g = g * rcp_samples;
    b = b * rcp_samples;


    device_fb[i][j] = ((u_int32_t)(MIN(r, 1.0) * 255.0) & 0xff) << RSHIFT |   
                      ((u_int32_t)(MIN(g, 1.0) * 255.0) & 0xff) << GSHIFT |
                      ((u_int32_t)(MIN(b, 1.0) * 255.0) & 0xff) << BSHIFT;

    return;
    }
}


/* trace a ray throught the scene recursively (the recursion happens through                
 * shade() to calculate reflection rays if necessary).
 */
__device__ struct vec3 trace(struct ray ray, int depth, int *isReflect, struct reflectdata *Rdata, double *obj_list_flat_dev, int lnumdev, struct vec3 *lightsdev, int *objCounterdev) {
    //printf("beginning of trace success!\n");
    struct vec3 col;
    struct spoint sp, nearest_sp;

    struct validsphere nearest_obj;

    nearest_obj.null = 0;

    int iterCount = 0; 

    double flat_sphere[9]; 

    get_ith_sphere(obj_list_flat_dev, iterCount, flat_sphere); // populates flat_sphere

    /* if we reached the recursion limit, bail out */
    if(depth >= MAX_RAY_DEPTH) {
        col.x = col.y = col.z = 0.0;
        return col;
    }

    int i;	
    /* find the nearest intersection ... */

    while(iterCount <= *objCounterdev) {
        if(ray_sphere(flat_sphere, ray, &sp)) {
            if(!nearest_obj.null || sp.dist < nearest_sp.dist) {
                for (i=0; i<9; i++)
                {
                    nearest_obj.sph[i] = flat_sphere[i];
                }
                nearest_obj.null = 1;
                nearest_sp = sp;
            }
        }
        iterCount++;
        get_ith_sphere(obj_list_flat_dev, iterCount, flat_sphere);

    }

    //and perform shading calculations as needed by calling shade() 
    if(nearest_obj.null) {
        col = shade(nearest_obj, &nearest_sp, depth, isReflect, Rdata, obj_list_flat_dev, lnumdev, lightsdev);
    } else {
        col.x = col.y = col.z = 0.0;
    }
    return col;
}

/* Calculates direct illumination with the phong reflectance model.
 * Also handles reflections by calling trace again, if necessary.
 */
__device__ struct vec3 shade(struct validsphere obj, struct spoint *sp, int depth, int *isReflect, struct reflectdata *Rdata, double *obj_list_flat_dev, int lnumdev, struct vec3 *lightsdev) {
    int i;
    struct vec3 col = {0, 0, 0};
    int iterCount = 0;

    // for all lights ... 
    for(i=0; i<lnumdev; i++) {
        double ispec, idiff;
        struct vec3 ldir;
        struct ray shadow_ray;

        double flat_sphere[9];
        get_ith_sphere(obj_list_flat_dev, iterCount, flat_sphere); // populates flat_sphere

        int in_shadow = 0;

        ldir.x = lightsdev[i].x - sp->pos.x;
        ldir.y = lightsdev[i].y - sp->pos.y;
        ldir.z = lightsdev[i].z - sp->pos.z;

        shadow_ray.orig = sp->pos;
        shadow_ray.dir = ldir;

        //shoot shadow rays to determine if we have a line of sight with the light 
        while(flat_sphere) {
            if(ray_sphere(flat_sphere, shadow_ray, 0)) {
                in_shadow = 1;
                break;
            }
            iterCount++;
            get_ith_sphere(obj_list_flat_dev, iterCount, flat_sphere);
        }

        //and if we're not in shadow, calculate direct illumination with the phong model. 
        if(!in_shadow) {
            NORMALIZE(ldir);

            idiff = MAX(DOT(sp->normal, ldir), 0.0);
            ispec = obj.sph[7] > 0.0 ? pow(MAX(DOT(sp->vref, ldir), 0.0), obj.sph[7]) : 0.0;  // ASSUMING OBJ[7] = obj->mat.spow

            col.x += idiff * obj.sph[4] + ispec;      // assuming obj[4] = obj->mat.col.x
            col.y += idiff * obj.sph[5] + ispec;      // assuming obj[5] = obj->mat.col.y
            col.z += idiff * obj.sph[6] + ispec;   //assuming obj[6] = obj->may.col.z

        }


    }

    // Also, if the object is reflective, spawn a reflection ray, and call trace()
    //to calculate the light arriving from the mirror direction.
    //
    //FOR EVERY REFLECTION RAY THAT IS SUPPOSED TO BE TRACED, THERE MUST BE A STRUCTURE SAVED, CONTAINING THE SPECIFIC PIXEL, THE RAY, AND DEPTH + 1. ALL OF THESE MUST BE STORED IN A "NEW" ARRAY, WHICH IS THEN ACCESSED IN RENDER2 FOLLOWING THE MAIN COMPUTATIONS.!!!!!!!!!!!*******************8
    if(obj.sph[8] > 0.0) {           //assuming obj[8] = obj->mat.refl
        isReflect[0] = 1;    //set isReflect to affirmative 


        Rdata->ray.orig = sp->pos;     //SET VALUES OF REFLECTIONDATA STRUCT
        Rdata->ray.dir = sp->vref;
        Rdata->ray.dir.x *= RAY_MAG;
        Rdata->ray.dir.y *= RAY_MAG;
        Rdata->ray.dir.z *= RAY_MAG;
        Rdata->depth = depth + 1;
        Rdata->reflection = obj.sph[8];


    }
    else
        isReflect[0] = 0;      //IF THERE IS NO REFLECTION, SET ISREFLECT TO ZERO        

    return col;
}

/* calculate reflection vector */
__device__ struct vec3 reflect(struct vec3 v, struct vec3 n) {
    struct vec3 res;
    double dot = v.x * n.x + v.y * n.y + v.z * n.z;
    res.x = -(2.0 * dot * n.x - v.x);
    res.y = -(2.0 * dot * n.y - v.y);
    res.z = -(2.0 * dot * n.z - v.z);
    return res;
}

__device__ struct vec3 cross_product(struct vec3 v1, struct vec3 v2) {
    struct vec3 res;
    res.x = v1.y * v2.z - v1.z * v2.y;
    res.y = v1.z * v2.x - v1.x * v2.z;
    res.z = v1.x * v2.y - v1.y * v2.x;
    return res;
}

/* determine the primary ray corresponding to the specified pixel (x, y) */
__device__ struct ray get_primary_ray(int x, int y, int sample, struct camera camdev, struct vec3 *uranddev, int *iranddev) {
    struct ray ray;
    float m[3][3];
    struct vec3 i, j = {0, 1, 0}, k, dir, orig, foo;

    k.x = camdev.targ.x - camdev.pos.x;
    k.y = camdev.targ.y - camdev.pos.y;
    k.z = camdev.targ.z - camdev.pos.z;
    NORMALIZE(k);

    i = cross_product(j, k);
    j = cross_product(k, i);
    m[0][0] = i.x; m[0][1] = j.x; m[0][2] = k.x;
    m[1][0] = i.y; m[1][1] = j.y; m[1][2] = k.y;
    m[2][0] = i.z; m[2][1] = j.z; m[2][2] = k.z;

    ray.orig.x = ray.orig.y = ray.orig.z = 0.0;
    ray.dir = get_sample_pos(x, y, sample, uranddev, iranddev);
    ray.dir.z = 1.0 / HALF_FOV;
    ray.dir.x *= RAY_MAG;
    ray.dir.y *= RAY_MAG;
    ray.dir.z *= RAY_MAG;

    dir.x = ray.dir.x + ray.orig.x;
    dir.y = ray.dir.y + ray.orig.y;
    dir.z = ray.dir.z + ray.orig.z;
    foo.x = dir.x * m[0][0] + dir.y * m[0][1] + dir.z * m[0][2];
    foo.y = dir.x * m[1][0] + dir.y * m[1][1] + dir.z * m[1][2];
    foo.z = dir.x * m[2][0] + dir.y * m[2][1] + dir.z * m[2][2];

    orig.x = ray.orig.x * m[0][0] + ray.orig.y * m[0][1] + ray.orig.z * m[0][2] + camdev.pos.x;
    orig.y = ray.orig.x * m[1][0] + ray.orig.y * m[1][1] + ray.orig.z * m[1][2] + camdev.pos.y;
    orig.z = ray.orig.x * m[2][0] + ray.orig.y * m[2][1] + ray.orig.z * m[2][2] + camdev.pos.z;

    ray.orig = orig;
    ray.dir.x = foo.x + orig.x;
    ray.dir.y = foo.y + orig.y;
    ray.dir.z = foo.z + orig.z;

    return ray;
}


__device__ struct vec3 get_sample_pos(int x, int y, int sample, struct vec3 *uranddev, int *iranddev) {
    struct vec3 pt;
    //   double xsz = 2.0, ysz = xresdev / aspectdev;   not being used by program?
    /*static*/ double sf = 0.0;

    if(sf == 0.0) {
        sf = 2.0 / (double)xresdev;
    }

    pt.x = ((double)x / (double)xresdev) - 0.5;
    pt.y = -(((double)y / (double)yresdev) - 0.65) / aspectdev;

    if(sample) {
        struct vec3 jt = jitter(x, y, sample, uranddev, iranddev);
        pt.x += jt.x * sf;
        pt.y += jt.y * sf / aspectdev;
    }
    return pt;
}

/* jitter function taken from Graphics Gems I. */
__device__ struct vec3 jitter(int x, int y, int s, struct vec3 *uranddev, int *iranddev) {
    struct vec3 pt;
    pt.x = uranddev[(x + (y << 2) + iranddev[(x + s) & MASK]) & MASK].x;
    pt.y = uranddev[(y + (x << 2) + iranddev[(y + s) & MASK]) & MASK].y;
    return pt;
}

/* Calculate ray-sphere intersection, and return {1, 0} to signify hit or no hit.
 * Also the surface point parameters like position, normal, etc are returned through
 * the sp pointer if it is not NULL.
 */
__device__ int ray_sphere(double *sph, struct ray ray, struct spoint *sp) {
    double a, b, c, d, sqrt_d, t1, t2;

    a = SQ(ray.dir.x) + SQ(ray.dir.y) + SQ(ray.dir.z);
    b = 2.0 * ray.dir.x * (ray.orig.x - sph[0]) +
        2.0 * ray.dir.y * (ray.orig.y - sph[1]) +
        2.0 * ray.dir.z * (ray.orig.z - sph[2]);
    c = SQ(sph[0]) + SQ(sph[1]) + SQ(sph[2]) +
        SQ(ray.orig.x) + SQ(ray.orig.y) + SQ(ray.orig.z) +
        2.0 * (-sph[0] * ray.orig.x - sph[1] * ray.orig.y - sph[2] * ray.orig.z) - SQ(sph[3]);

    if((d = SQ(b) - 4.0 * a * c) < 0.0) return 0;

    sqrt_d = sqrt(d);
    t1 = (-b + sqrt_d) / (2.0 * a);
    t2 = (-b - sqrt_d) / (2.0 * a);

    if((t1 < ERR_MARGIN && t2 < ERR_MARGIN) || (t1 > 1.0 && t2 > 1.0)) return 0;

    if(sp) {
        if(t1 < ERR_MARGIN) t1 = t2;
        if(t2 < ERR_MARGIN) t2 = t1;
        sp->dist = t1 < t2 ? t1 : t2;

        sp->pos.x = ray.orig.x + ray.dir.x * sp->dist;
        sp->pos.y = ray.orig.y + ray.dir.y * sp->dist;
        sp->pos.z = ray.orig.z + ray.dir.z * sp->dist;

        sp->normal.x = (sp->pos.x - sph[0]) / sph[3];
        sp->normal.y = (sp->pos.y - sph[1]) / sph[3];
        sp->normal.z = (sp->pos.z - sph[2]) / sph[3];

        sp->vref = reflect(ray.dir, sp->normal);
        NORMALIZE(sp->vref);
    }
    return 1;
}

/* Load the scene from an extremely simple scene description file */
#define DELIM   " \t\n"
void load_scene(FILE *fp) {
    char line[256], *ptr, type;

    obj_list = (sphere *)malloc(sizeof(struct sphere));
    obj_list->next = 0;
    objCounter = 0;

    while((ptr = fgets(line, 256, fp))) {
        int i;
        struct vec3 pos, col;
        double rad, spow, refl;

        while(*ptr == ' ' || *ptr == '\t') ptr++;
        if(*ptr == '#' || *ptr == '\n') continue;

        if(!(ptr = strtok(line, DELIM))) continue;
        type = *ptr;

        for(i=0; i<3; i++) {
            if(!(ptr = strtok(0, DELIM))) break;
            *((double*)&pos.x + i) = atof(ptr);
        }

        if(type == 'l') {
            lights[lnum++] = pos;
            continue;
        }

        if(!(ptr = strtok(0, DELIM))) continue;
        rad = atof(ptr);

        for(i=0; i<3; i++) {
            if(!(ptr = strtok(0, DELIM))) break;
            *((double*)&col.x + i) = atof(ptr);
        }

        if(type == 'c') {
            cam.pos = pos;
            cam.targ = col;
            cam.fov = rad;
            continue;
        }

        if(!(ptr = strtok(0, DELIM))) continue;
        spow = atof(ptr);

        if(!(ptr = strtok(0, DELIM))) continue;
        refl = atof(ptr);

        if(type == 's') { 
            objCounter++;
            struct sphere *sph = (sphere *)malloc(sizeof(*sph));
            sph->next = obj_list->next;
            obj_list->next = sph;

            sph->pos = pos;
            sph->rad = rad;
            sph->mat.col = col;
            sph->mat.spow = spow;
            sph->mat.refl = refl;

        } else {
            fprintf(stderr, "unknown type: %c\n", type);
        }
    }

}

void flatten_sphere(struct sphere *sphere, double *sphere_flat) {
    struct vec3 pos = sphere->pos;
    double rad = sphere->rad;
    struct material mat = sphere->mat;

    sphere_flat[0] = pos.x;
    sphere_flat[1] = pos.y;
    sphere_flat[2] = pos.z;
    sphere_flat[3] = rad;
    sphere_flat[4] = mat.col.x;
    sphere_flat[5] = mat.col.y;
    sphere_flat[6] = mat.col.z;
    sphere_flat[7] = mat.spow;
    sphere_flat[8] = mat.refl;
}

void flatten_obj_list(struct sphere *obj_list, double *obj_list_flat, int objCounter) {

    for (int i = 0; i < objCounter; i++) {
        struct sphere *sphere = obj_list;
        double sphere_flat[9];
        flatten_sphere(sphere, sphere_flat);

        for (int j = 0; j < objCounter; j++) {
            for (int k = 0; k < 9; k++) {
                obj_list_flat[9*j+k] = sphere_flat[k];
                //printf("sphere_flat[%i] = %f", k, sphere_flat[k]);
            }
        }

        obj_list = obj_list->next;
        i++;
    }
}

__device__ void get_ith_sphere(double *obj_list_flat, int index, double *flat_sphere) {
    int base_index = index * 9;

    for (int i = 0; i <= 9; i++) {
        flat_sphere[i] = obj_list_flat[base_index + i]; 
    }


}


inline void check_cuda_errors(const char *filename, const int line_number)
{
#ifdef DEBUG
    cudaThreadSynchronize();
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        printf("CUDA error at %s:%i: %s\n", filename, line_number, cudaGetErrorString(error));
        exit(-1);
    }
#endif
}   


/* provide a millisecond-resolution timer for each system */
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



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

int objCounter = 0;

struct parallelPixels { //STRUCT WHICH DIVIDES PIXELS FOR RENDER2 GLOBAL FUNCTION TO USE
    signed int start[1];
    signed int end[1];
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

void render1(int xsz, int ysz, u_int32_t *fb, int samples);
__global__ void render2(u_int32_t **fbDevice, struct parallelPixels *pixelsPerCore, int samples); //SPECIFY ARGUMENTS TO RENDER2~!!!!
__device__ struct vec3 trace(struct ray ray, int depth, int *isReflect, struct reflectdata *Rdata); //two arguments added - one to check if a reflection ray must be made, the other to provide the arguments necessary for the reflection ray
__device__ struct vec3 shade(struct sphere *obj, struct spoint *sp, int depth, int *isReflect, struct reflectdata *Rdata);
__device__ struct vec3 reflect(struct vec3 v, struct vec3 n);
__device__ struct vec3 cross_product(struct vec3 v1, struct vec3 v2);
__device__ struct ray get_primary_ray(int x, int y, int sample);
__device__ struct vec3 get_sample_pos(int x, int y, int sample);
__device__ struct vec3 jitter(int x, int y, int s);
__device__ int ray_sphere(const struct sphere *sph, struct ray ray, struct spoint *sp);
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
__device__ struct sphere *obj_listdev = 0;
__device__ struct vec3 lightsdev[MAX_LIGHTS];
__device__ int lnumdev = 0;
__device__ struct camera camdev;

__device__ struct vec3 uranddev[NRAN];
__device__ int iranddev[NRAN];


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
    int i;
    unsigned long rend_time, start_time;
    u_int32_t *pixels;
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

    if(!(pixels = (u_int32_t *)malloc(xres * yres * sizeof *pixels))) {
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

    //I'M NOT GOING TO COPY OVER OBJ_LIST BECAUSE I THINK OBJ_LISTDEV CAN BE DIRECTLY SENT TO THE DEVICE MEMORY! I HAVEN'T IMPLEMENTED THIS YET, THOUGH..

    //cuda memcopy could also work here
    for(i=0; i<MAX_LIGHTS; i++) {
        lightsdev[i] = lights[i];
        lnumdev = lnum;
        camdev = cam;
    }
    //cuda memcopy could also work here
    for(i=0; i<NRAN; i++) {
       uranddev[i]= urand[i];
    }
    //cuda memcopy could also work here
    for(i=0; i<NRAN; i++) {
       iranddev[i] = irand[i]; 
    }

    
    start_time = get_msec();
    render1(xres, yres, pixels, rays_per_pixel);
    rend_time = get_msec() - start_time;
    
    /* output statistics to stderr */
    fprintf(stderr, "Rendering took: %lu seconds (%lu milliseconds)\n", rend_time / 1000, rend_time);

    /* output the image */
    fprintf(outfile, "P6\n%d %d\n255\n", xres, yres);
    for(i=0; i<xres * yres; i++) {
        fputc((pixels[i] >> RSHIFT) & 0xff, outfile);
        fputc((pixels[i] >> GSHIFT) & 0xff, outfile);
        fputc((pixels[i] >> BSHIFT) & 0xff, outfile);
    }
    fflush(outfile);

    if(infile != stdin) fclose(infile);
    if(outfile != stdout) fclose(outfile);
    return 0;
}

/* render a frame of xsz/ysz dimensions into the provided framebuffer */
void render1(int xsz, int ysz, u_int32_t *fb, int samples)
{

    int num_elementsParallelPixel = 3; //there will be an array of three structs which each determine the pixel one core will begin tracing at and the pixel it will end tracing at(x and y values)
 
    int block_size = 3;              //3 * 1 = total number of cores
    int grid_size = 1;
 
    int totalOps = xsz * ysz;       //total number of pixels to have rays traced to (x coord. multiplied by y coord.)
    int numOpsPerCore = totalOps/(block_size*grid_size);   // amount of rays to be traced by each core

    int num_bytes_ParallelPixel = num_elementsParallelPixel * sizeof(struct parallelPixels);
    struct parallelPixels *device_pixelspercore = 0;        //device array of pixels per core(sent to all of the cores)
    struct parallelPixels *host_pixelspercore = 0;          //host array of pixels per core(only contained by host)
 
    // malloc a host array
    host_pixelspercore = (struct parallelPixels*)malloc(num_bytes_ParallelPixel); 

    // cudaMalloc a device array
    if (cudaSuccess != cudaMalloc((void**)&device_pixelspercore, num_bytes_ParallelPixel)) {
        printf("cudaMalloc failed: device_pixelspercore\n");
        exit(1);
    }

    
//Ah, the frame buffer.
//Originally, the frame buffer contained all of the pixels of the frame, sorted in order(of increasing x per y value?)
//Anyway, all of the devices(cores) need some place to put the output of their computations.
//They could choose to put the output into the frame buffer array!
//But there's a problem. 
//Each device is writing multiple outputs to the array, based on how many rays it will be rendering.
//Additionally, each device has only -one- index to this array, based on what number core it is!(grid number * block number * something else which uniquely identifies each core)
//This means that each core can only save one element into the frame buffer array which won't get overwritten by another core!
//So, what do we do?
//Well, we could choose to send a special 2D array to the devices, which allows no outdata to get overwritten!
//Then, we could transfer this 2D array to a host 2D array.
//Finally, we could loop through all of the output in the 2D array and sequentially put it into a regular 1D array(the REAL frame buffer). ***Ah, did someone say "sequentially"? Is this an area which could possibly be made parallel as well?**
//Yup, that's it. You've just read through the most inefficient idea ever.  

    
//and now to determine the details of the 2D fb array...
// code heavily taken from 
//http://forums.nvidia.com/index.php?s=5712c99a6838532e8e43081108fce9f8&showtopic=69403&st=20
// AND http://pleasemakeanote.blogspot.com/2008/06/2d-arrays-in-c-using-malloc.html

    u_int32_t **device_fb = 0;
    u_int32_t **host_fb = 0;
    u_int32_t *h_temp[block_size*grid_size];

    host_fb = (u_int32_t**) malloc((block_size*grid_size)*sizeof(u_int32_t*));
    //Special host 2D array is created(malloced)
    for (int i = 0; i < (block_size*grid_size); i++)
        host_fb[i] = (u_int32_t*) malloc(numOpsPerCore*sizeof(u_int32_t));
    
    // Now to create the special device 2D Array
    if (cudaSuccess != cudaMalloc((void **)&device_fb, (block_size*grid_size)*sizeof(u_int32_t))) {
        printf("cudaMalloc failed: device_fb\n");
        exit(1);
    }

    for(int i=0; i<(block_size*grid_size); i++)
    {
        if (cudaSuccess != cudaMalloc( (void **)&h_temp[i], numOpsPerCore*sizeof(u_int32_t))) {
            printf("cudaMalloc failed: h_temp\n");
            exit(1);
        }
    }
    cudaMemcpy(device_fb, h_temp, (block_size*grid_size)*sizeof(u_int32_t*), cudaMemcpyHostToDevice);

    // Do not destroy the contents of h_temp
    // So, we dont need to copy the pointers from the device again and again. We will hold a copy of the row pointers in h_temp


    for(int i=0; i<block_size*grid_size; i++)
    {
        cudaMemcpy(h_temp[i], host_fb[i], numOpsPerCore*sizeof(u_int32_t), cudaMemcpyHostToDevice);
    }
    //PUT DATA INTO PARALLEL PIXEL STRUCT PIXELSPERCORE!!

    int completeCore = 0;   //representing how many cores have the max data stored
    int dataPerCore = 0;   //representing how much data is stored per core

    int y, x;              //representing the x and y values of the pixels which are stored in the pixelspercore array

    for (y=0; y<ysz; y++)
    {//for each y value, store every possible (x,y) coord into pixelspercore structs
        for (x=0;x<xsz; x++)
        {//if there is no data yet in struct, then we must record the first (x,y) pixel which a given core will trace a ray to
            if (dataPerCore == 0)
            {
                host_pixelspercore[completeCore].start[0] = x;
                host_pixelspercore[completeCore].start[1] = y;
            }
        
            if (dataPerCore==(numOpsPerCore-1))
            {//If the maximum amount of pixels which can have rays traced per core are about to be stored/counted, record the last (x,y) pixel which a given core will trace a ray to in the struct
                host_pixelspercore[completeCore].end[0] = x;
                host_pixelspercore[completeCore].end[1] = y;
                completeCore++;    //one entire core data completed/stored
            }

            dataPerCore++; //one part of core data stored
            
            if (dataPerCore == numOpsPerCore)
                dataPerCore = 0; //reset counter if entire core data has been stored
        }
    }
    //if there are any cores which will not be doing any operations, start and end values will be set to -1
    //each core/thread will have to check if any of the given values are -1. if this is the case, they should immediately return.
    if ((totalOps/numOpsPerCore) < (block_size*grid_size))
    {
        int emptyCores = (block_size*grid_size) - (totalOps/numOpsPerCore); //total number of cores - cores used
        for (int k = 0; k < emptyCores; k++) // for every empty core left, continue to add -1s to the pixelspercore struct
        {
            host_pixelspercore[completeCore].start[0] = -1;
            host_pixelspercore[completeCore].start[1] = -1;
            host_pixelspercore[completeCore].end[0] = -1;
            host_pixelspercore[completeCore].end[1] = -1;
            completeCore++; //one entire core data completed/stored
        }
    }
    //copy over host array which determines which pixels should have rays traced per core to device array
    cudaMemcpy(device_pixelspercore, host_pixelspercore, num_bytes_ParallelPixel, cudaMemcpyHostToDevice);
    
    //FUNCTION TIEM
    render2<<<block_size,grid_size>>>(device_fb, device_pixelspercore, samples);
    //In all seriousness, all of the cores should now be operating on the ray tracing, if things are working correctly 

    //check_cuda_errors(debug_errors, 732)//GIVEN line number of  FUNCTION WE WISH TO TEST!);    //debugging support (see notes)
    //once done, copy contents of device array to host array    
    for(int i=0; i<(block_size*grid_size); i++)
    {
        cudaMemcpy(host_fb[i], h_temp[i], numOpsPerCore*sizeof(u_int32_t), cudaMemcpyDeviceToHost);
    }//WHAT EXACTLY IS THE PURPOSE OF THE H_TEMP ARRAY? IT SHOULD BE FREED! CAN IT BE GOTTEN RID OF WITHOUT A PROBLEM???       ******IMPLEEEEEEEEEEEMEEEEEEEEEEENNNNNNNNNNNNTTTTTTT!!!!!!!******

    //then, copy host_fb contents to THE REAL frame buffer array so that everything is in order...
    
    int fbCounter = 0;
    for(int c=0; c < (block_size*grid_size);c++)
    {//for each core
        if (fbCounter == totalOps)  //if total amount of pixels have been stored, stop computing through loop 
            break;
        for (int c2=0; c2<numOpsPerCore; c2++)
        {//and for each pixel which had a ray traced per a given core...
            fb[fbCounter] = host_fb[c][c2]; //one piece of pixel data is stored into frame buffer
            fbCounter++;                    //frame buffer array increases
        }   
    }   
    //free host and device pixelspercore 

    free(host_pixelspercore);  
    cudaFree(device_pixelspercore);
    
    //free host and device fb 2D arrays

    //this might have to be looked at in terms of repeated incrementer values and the like (LIKE USING THE VALUE i MULTIPLE TIMES? **********************************************)
    for(int i=0; i<(block_size*grid_size); i++)
    {
        cudaFree(host_fb[i]);
    }
    cudaFree(device_fb);

    for (int i = 0; i < (block_size*grid_size); i++)
    {  
       free(host_fb[i]);  
    }  
    free(host_fb);  
    
}   


__global__ void render2(u_int32_t **fbDevice, struct parallelPixels *pixelsPerCore, int samples)            //SPECIFY ARGUMENTS!!!
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; //DETERMINING INDEX BASED ON WHICH THREAD IS CURRENTLY RUNNING

    int s;
    int i = pixelsPerCore[index].start[0];   //x value of first pixel 
    if (i==-1) return;                       //if a -1 value is placed in the start value, then there is no pertinent data here. return early. Note that this will not decrease speed of parallel operations, since the speed is determined by the slowest parallel operation..
    int j = pixelsPerCore[index].start[1];   //y value of first pixel
    int xsz = pixelsPerCore[index].end[0];   //x value of last pixel
    int ysz = pixelsPerCore[index].end[1];   //y value of last pixel
    int raysTraced = 0;                     // number of rays traced 
    int isReflect[1];                        //WHETHER OR NOT RAY TRACED WILL NEED A REFLECTION RAY AS WELL
    isReflect[0] = 0;
    struct reflectdata RData[1];           //ARRAY WHICH CONTAINS REFLECT DATA STRUCT TO BE PASSED ON TO TRACE FUNCTION
    
    double rcp_samples = 1.0 / (double)samples;

    /* for each subpixel, trace a ray through the scene, accumulate the
     * colors of the subpixels of each pixel, then pack the color and
     * put it into the framebuffer.
     * XXX: assumes contiguous scanlines with NO padding, and 32bit pixels.
     */
    for(j=0; j<=ysz; j++) {
        for(i=0; i<=xsz; i++) {
            double r, g, b;
            r = g = b = 0.0;
            
            for(s=0; s<samples; s++) {
                struct vec3 col = trace(get_primary_ray(i, j, s), 0, isReflect, RData);
                while (*isReflect)        //while there are still reflection rays to trace
                {
                    struct vec3 rcol;    //holds the output of the reflection ray calculcation
                    rcol = trace(RData->ray, RData->depth, isReflect, RData);    //trace a reflection ray
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
                
            fbDevice[index][raysTraced] = ((u_int32_t)(MIN(r, 1.0) * 255.0) & 0xff) << RSHIFT |   
                    ((u_int32_t)(MIN(g, 1.0) * 255.0) & 0xff) << GSHIFT |
                    ((u_int32_t)(MIN(b, 1.0) * 255.0) & 0xff) << BSHIFT;
                    
            raysTraced++;       //one pixel-post-ray-trace data has been stored!
        }
    }
}



/* trace a ray throught the scene recursively (the recursion happens through                
 * shade() to calculate reflection rays if necessary).
 */
__device__ struct vec3 trace(struct ray ray, int depth, int *isReflect, struct reflectdata *Rdata) {
    struct vec3 col;
    struct spoint sp, nearest_sp;
    struct sphere *nearest_obj = 0;
    struct sphere *iter = obj_listdev->next;

    /* if we reached the recursion limit, bail out */
    if(depth >= MAX_RAY_DEPTH) {
        col.x = col.y = col.z = 0.0;
        return col;
    }
    
    /* find the nearest intersection ... */
    while(iter) {
        if(ray_sphere(iter, ray, &sp)) {
            if(!nearest_obj || sp.dist < nearest_sp.dist) {
                nearest_obj = iter;
                nearest_sp = sp;
            }
        }
        iter = iter->next;
    }

    /* and perform shading calculations as needed by calling shade() */
    if(nearest_obj) {
        col = shade(nearest_obj, &nearest_sp, depth, isReflect, Rdata);
    } else {
        col.x = col.y = col.z = 0.0;
    }

    return col;
}

/* Calculates direct illumination with the phong reflectance model.
 * Also handles reflections by calling trace again, if necessary.
 */
__device__ struct vec3 shade(struct sphere *obj, struct spoint *sp, int depth, int *isReflect, struct reflectdata *Rdata) {
    int i;
    struct vec3 col = {0, 0, 0};

    /* for all lights ... */
    for(i=0; i<lnumdev; i++) {
        double ispec, idiff;
        struct vec3 ldir;
        struct ray shadow_ray;
        struct sphere *iter = obj_listdev->next;
        int in_shadow = 0;

        ldir.x = lightsdev[i].x - sp->pos.x;
        ldir.y = lightsdev[i].y - sp->pos.y;
        ldir.z = lightsdev[i].z - sp->pos.z;

        shadow_ray.orig = sp->pos;
        shadow_ray.dir = ldir;

        /* shoot shadow rays to determine if we have a line of sight with the light */
        while(iter) {
            if(ray_sphere(iter, shadow_ray, 0)) {
                in_shadow = 1;
                break;
            }
            iter = iter->next;
        }

        /* and if we're not in shadow, calculate direct illumination with the phong model. */
        if(!in_shadow) {
            NORMALIZE(ldir);

            idiff = MAX(DOT(sp->normal, ldir), 0.0);
            ispec = obj->mat.spow > 0.0 ? pow(MAX(DOT(sp->vref, ldir), 0.0), obj->mat.spow) : 0.0;

            col.x += idiff * obj->mat.col.x + ispec;
            col.y += idiff * obj->mat.col.y + ispec;
            col.z += idiff * obj->mat.col.z + ispec;
        }
    }

    /* Also, if the object is reflective, spawn a reflection ray, and call trace()
     * to calculate the light arriving from the mirror direction.
     */
     //FOR EVERY REFLECTION RAY THAT IS SUPPOSED TO BE TRACED, THERE MUST BE A STRUCTURE SAVED, CONTAINING THE SPECIFIC PIXEL, THE RAY, AND DEPTH + 1. ALL OF THESE MUST BE STORED IN A "NEW" ARRAY, WHICH IS THEN ACCESSED IN RENDER2 FOLLOWING THE MAIN COMPUTATIONS.!!!!!!!!!!!*******************8
    if(obj->mat.refl > 0.0) {
        isReflect[0] = 1;    //set isReflect to affirmative 


        Rdata->ray.orig = sp->pos;     //SET VALUES OF REFLECTIONDATA STRUCT
        Rdata->ray.dir = sp->vref;
        Rdata->ray.dir.x *= RAY_MAG;
        Rdata->ray.dir.y *= RAY_MAG;
        Rdata->ray.dir.z *= RAY_MAG;
        Rdata->depth = depth + 1;
        Rdata->reflection = obj->mat.refl;
        

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
__device__ struct ray get_primary_ray(int x, int y, int sample) {
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
    ray.dir = get_sample_pos(x, y, sample);
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


__device__ struct vec3 get_sample_pos(int x, int y, int sample) {
    struct vec3 pt;
 //   double xsz = 2.0, ysz = xresdev / aspectdev;   not being used by program?
    /*static*/ double sf = 0.0;

    if(sf == 0.0) {
        sf = 2.0 / (double)xresdev;
    }

    pt.x = ((double)x / (double)xresdev) - 0.5;
    pt.y = -(((double)y / (double)yresdev) - 0.65) / aspectdev;

    if(sample) {
        struct vec3 jt = jitter(x, y, sample);
        pt.x += jt.x * sf;
        pt.y += jt.y * sf / aspectdev;
    }
    return pt;
}

/* jitter function taken from Graphics Gems I. */
__device__ struct vec3 jitter(int x, int y, int s) {
    struct vec3 pt;
    pt.x = uranddev[(x + (y << 2) + iranddev[(x + s) & MASK]) & MASK].x;
    pt.y = uranddev[(y + (x << 2) + iranddev[(y + s) & MASK]) & MASK].y;
    return pt;
}

/* Calculate ray-sphere intersection, and return {1, 0} to signify hit or no hit.
 * Also the surface point parameters like position, normal, etc are returned through
 * the sp pointer if it is not NULL.
 */
__device__ int ray_sphere(const struct sphere *sph, struct ray ray, struct spoint *sp) {
    double a, b, c, d, sqrt_d, t1, t2;
    
    a = SQ(ray.dir.x) + SQ(ray.dir.y) + SQ(ray.dir.z);
    b = 2.0 * ray.dir.x * (ray.orig.x - sph->pos.x) +
                2.0 * ray.dir.y * (ray.orig.y - sph->pos.y) +
                2.0 * ray.dir.z * (ray.orig.z - sph->pos.z);
    c = SQ(sph->pos.x) + SQ(sph->pos.y) + SQ(sph->pos.z) +
                SQ(ray.orig.x) + SQ(ray.orig.y) + SQ(ray.orig.z) +
                2.0 * (-sph->pos.x * ray.orig.x - sph->pos.y * ray.orig.y - sph->pos.z * ray.orig.z) - SQ(sph->rad);
    
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
        
        sp->normal.x = (sp->pos.x - sph->pos.x) / sph->rad;
        sp->normal.y = (sp->pos.y - sph->pos.y) / sph->rad;
        sp->normal.z = (sp->pos.z - sph->pos.z) / sph->rad;

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
    struct vec3 pos = obj_list->pos;
    double rad = obj_list->rad;
    struct material mat = obj_list->mat;

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
    obj_list_flat = (double *)malloc(9*objCounter*sizeof(double));

    double doubleCounter = objCounter*9;

    for (int i = 0; i <= objCounter; i++) {
        struct sphere *sphere = obj_list;
        double sphere_flat[9];
        flatten_sphere(sphere, sphere_flat);

        for (int j = 0; j <= doubleCounter; j++) {
            for (int k = 0; k <= 9; k++) {
                obj_list_flat[j] = sphere_flat[k];
            }
        }

        obj_list = obj_list->next;
        i++;
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

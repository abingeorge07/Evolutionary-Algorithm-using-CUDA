#include <cstdio>
#include <cstdlib>
#include <math.h>
#include "cuPrintf.cuh"
#include "cuPrintf.cu"
#include <stdlib.h>
#include<float.h>
#include <stdio.h>
#include <curand_kernel.h>

// Run with 
// nvcc EA_cuda.cu
// ./a.out

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

#define numberCities 100
#define PRINT_TIME   1
#define xLimit 10
#define yLimit 0
#define blocks 1
#define threads 10
#define population blocks*threads
#define iters 100000



__device__ float global_minDis = FLT_MAX;



#define IMUL(a, b) __mul24(a, b)

double interval(struct timespec start, struct timespec end)
{
  struct timespec temp;
  temp.tv_sec = end.tv_sec - start.tv_sec;
  temp.tv_nsec = end.tv_nsec - start.tv_nsec;
  if (temp.tv_nsec < 0) {
    temp.tv_sec = temp.tv_sec - 1;
    temp.tv_nsec = temp.tv_nsec + 1000000000;
  }
  return (((double)temp.tv_sec) + ((double)temp.tv_nsec)*1.0e-9);
}

void setup(float* x, float* y, int*in, float* max, float*min)
{
    int i;
    for(i=0; i<numberCities; i++)
    {
        x[i]= ((float)rand()/(float)RAND_MAX)*xLimit;
        if(x[i]>*max){
            *max=x[i];
        }
        if(x[i]<*min){
            *min=x[i];
        }
        y[i]= ((float)rand()/(float)RAND_MAX)*yLimit;
        in[i]=i;
    }

}

void swap(int* arr, int one, int two){
    int temp = arr[one];
    arr[one]= arr[two];
    arr[two]= temp;

}

void shuffle(int* random){
    int i, randN;


    for(i =0;i<numberCities;i++){
        randN=round(((float)rand()/(float)RAND_MAX)*(numberCities-1-i)+i);
        swap(random, i, randN);
    }

}

void assignGene(int* genes, float* x, float*y, int* ind){
    int i,j;
    int random[numberCities];

    for(i=0; i<numberCities; i++){
        random[i]= ind[i];
    }

    for(i =0; i<population; i++){
        shuffle(random);
        for(j=0; j<numberCities; j++){
            genes[i*numberCities+j]= random[j];
        }
    }
}

void printCityLoc (float *x, float* y){
    int i;

    for(i=0; i<numberCities; i++){
        printf("Index %d: x = %f y=%f \n", i, x[i], y[i]);
    }
}

__device__ float global_min = FLT_MAX;

__device__ float dist(float x1, float x2, float y1, float y2){
    float temp= ((x1-x2)*(x1-x2)) + ((y1-y2)*(y1-y2));
    temp = sqrt(temp);
    return temp;
}

__device__ void copyBest(int ind, int* genes, int* d_bestSoFar){

  int i;
  for(i=0; i<numberCities; i++){
    d_bestSoFar[i]= genes[ind*numberCities+i];
  }

}

__device__ void lock(int* mutex) {
    /* compare mutex to 0.
       when it equals 0, set it to 1
       we will break out of the loop after mutex gets set to 1 */
    while (atomicCAS(mutex, 0, 1) != 0) {
        /* do nothing */
    }
}

__device__ void unlock(int* mutex) {
    atomicExch(mutex, 0);
}

__device__ void distanceCal(float* x, float* y, int* genes, float* dis, int* local_min_ind, float* local_min_dis, int* d_bestSoFar){

  int id= blockIdx.x*threads+ threadIdx.x;
  int j;
  int ind;
  float temp;
  float sum;
  float minDis;

  int temp2;
__shared__ int dis_temp[threads];

dis_temp[id]= id;

  sum=0;
  for(j=0; j<numberCities-1; j++){
    temp = dist(x[genes[id*numberCities+j]], x[genes[id*numberCities+j+1]],y[genes[id*numberCities+j]], y[genes[id*numberCities+j+1]]);
    sum =sum + temp;
  }

__syncthreads();

  dis[id]=sum;



//Following is the parallel reduction used to find the individual with the 
//minimum distance
__syncthreads();
  for(j=threads/2; j>0; j= j/2){
    if(id<j)
    {
      if(dis[dis_temp[id]]<dis[dis_temp[id+j]]){
        dis_temp[id]= dis_temp[id];
      }
      else{
        dis_temp[id]= dis_temp[id+j];
      }
    }
    if(id==0)
    {
      if(j%2!=0){
        temp2 =j;
      }
      else 
      {
        temp2=0;
      }

      if(temp2!=0){
        if(dis[dis_temp[0]]<dis[dis_temp[temp2]]){
        dis_temp[0]= dis_temp[0];
      }
      else{
        dis_temp[0]= dis_temp[temp2];
      }
      }
    }
    __syncthreads();
  }



  if(id==0){
    ind= dis_temp[id];
    minDis= dis[ind];
    *local_min_dis = minDis;
    *local_min_ind = ind;
    if(*local_min_dis<global_min){
    global_min = *local_min_dis;
    copyBest(*local_min_ind, genes, d_bestSoFar);
  }
  }


//could potentially improve
// if(id==0){
//   minDis= dis[0];
//   ind=0;
//   for(j=1; j<population; j++)
//   {
//     if(dis[j]<minDis){
//       minDis = dis[j];
//       ind=j;
//     }
//   }

//   *local_min_dis = minDis;
//   *local_min_ind = ind;

//   if(*local_min_dis<global_min){
//     global_min = *local_min_dis;
//     copyBest(*local_min_ind, genes, d_bestSoFar);
//   }
// }
__syncthreads();
}

//used for testing, not required for the functionality
// __device__ void printDist(float* dis){
//   int id= blockIdx.x* threads+ threadIdx.x;
// }

__device__ void fitnessFun(float* fitVal, float* dis){
  int id= blockIdx.x* threads+ threadIdx.x;
  float total;
  int i;
  if(id==0){
    total=0;
    for(i=0; i<population; i++){
      total = total+ dis[i];
    }
    for(i=0; i<population; i++){
      fitVal[i] = dis[i]/total ;
    }
    // printf("%f \n", total);
  }
  __syncthreads();

}

__device__ void mutate(int* genes, int* local_min_ind, float rate, curandState *state, int* d_bestSoFar){

  int num_chosen =int(rate*numberCities);
  int id= blockIdx.x*threads + threadIdx.x;
  int temporary[numberCities];
  int i,j,hold,ind1,ind2;
  int flag;
  int ind = *local_min_ind;
  int randN=floor(curand_uniform(state+id)*(numberCities-1)); //should be changed to be random
  float dec= curand_uniform(state+id);

 __syncthreads();

  for(i=0; i<num_chosen; i++){
   temporary[i]= genes[ind*numberCities+ randN];
   randN++;
   randN = randN % numberCities;
  }

 int next =num_chosen;

 for(i=0; i<numberCities; i++){
   hold =genes[id*numberCities+i];
   flag =0;

   for(j=0; j<num_chosen; j++){
      if(hold == temporary[j]){
        flag =1;
        break;
      }
    }

    if (flag==0){
      temporary[next]=hold;
      next++;
    }

    if(next== numberCities){
      break;
    }
 }


 for(i=0; i<numberCities;i++){
   genes[id*numberCities+i] = temporary[i];
 }

 float mRate= curand_uniform(state+id);

 if(mRate<0.8){
    ind1 = round(curand_uniform(state+id)*(numberCities-1));
    ind2 = (ind1+1)%numberCities;
    hold = genes[id*numberCities+ ind1];
    genes[id*numberCities+ ind1]= genes[id*numberCities+ ind2];
    genes[id*numberCities+ ind2]= hold;

 }
  __syncthreads();
}
//
__device__ void EA( int* genes, float* fitVal,int* local_min_ind, int iter, curandState *state, int* d_bestSoFar){
  int id = blockIdx.x*threads+ threadIdx.x;
  // int hold;
  // int flag;
  float dec= curand_uniform(state+id);
  // int temporary[numberCities];
  // int next;
  // int num_chosen; //can change later
  // int temp;
  float frac = iter/iters;
  float rate =curand_uniform(state+id) *(1-frac);

  mutate(genes, local_min_ind, rate, state, d_bestSoFar);
  __syncthreads();
}

__device__ void printRelevantInfo(float* local_min_dis){
    printf("Minimum Local Dis in this Iteration is %f. \nMinimum Global Dis is %f.\n\n\n", *local_min_dis, global_min);

}

__device__ void printGenes(int* genes){

  int i;
  int j;
  int id = blockIdx.x*threads+ threadIdx.x;
  if(id==0){
  for(i=0; i<population; i++){
    for(j=0; j<numberCities; j++){
      printf("%d  ", genes[i*numberCities+j]);
    }
    printf("\n");
  }
  }
}

__global__ void setup_kernel(curandState *state){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, 0, &state[idx]);
}

__global__ void main_Kernel(float* x, float* y,  float* dis, int* genes, int* local_min_ind, float* local_min_dis, int* d_bestSoFar, float* fitVal, curandState* state){
//  int j,i;
  int id=  blockIdx.x*threads +threadIdx.x;
  int iter =2;



  // printGenes(genes);
  distanceCal( x,  y,  genes,  dis, local_min_ind, local_min_dis, d_bestSoFar);

  __syncthreads();
  if(id==0){
   printf("Iteration 1\n\n");
   printRelevantInfo(local_min_dis);
  }

  while(iter<iters){
    fitnessFun(fitVal, dis);
    EA(genes, fitVal, local_min_ind, iter, state, d_bestSoFar);
    // printGenes(genes);
    __syncthreads();
    distanceCal( x,  y,  genes,  dis, local_min_ind, local_min_dis, d_bestSoFar);
    if(id==0){
      printf("Iteration %d\n\n", iter);
      printRelevantInfo(local_min_dis);
    }
    __syncthreads();
    iter++;
  }
}


int main(){
  srand(time(NULL));
  int i;

  curandState *d_state;
  cudaMalloc((void**)&d_state, threads*blocks*sizeof(curandState));

  // GPU Timing variables
    cudaEvent_t start, stop;
    float elapsed_gpu;

  // Select GPU
  CUDA_SAFE_CALL(cudaSetDevice(0));

 // Randomly chooses points that represents cities
  float h_xCoord[numberCities];
  float h_yCoord[numberCities];
  int indices[numberCities];
  float max;
  float min;
  max = 0;
  min = FLT_MAX;
  setup(h_xCoord, h_yCoord, indices, &max,&min);

// Host Variables
  int* h_bestSoFar;
  //genes where each row represent one gene
  int* h_genes;


// Device Variables
  float* d_xCoord;
  float* d_yCoord;
  float* d_dis;
  float* d_fitVal;
  int* d_local_popMinIndex;
  float* d_local_min_dist;
  int* d_bestSoFar;
  //genes where each row represent one gene
  int* d_genes;


  // Size allocation for the genes array
  size_t allocSize2d = numberCities* sizeof(int)*population;
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_genes, allocSize2d));
  h_genes = (int*) malloc(allocSize2d);


  assignGene(h_genes, h_xCoord, h_yCoord, indices);


  // size allocation for minimum local index and distance
  size_t allocSize_fl =  sizeof(float);
  size_t allocSize_int =  sizeof(int);
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_local_min_dist, allocSize_fl));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_local_popMinIndex, allocSize_int));

  // Size allocation for the x and y coordinates
  size_t allocSize_cities_fl = numberCities* sizeof(float);
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_xCoord, allocSize_cities_fl));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_yCoord, allocSize_cities_fl));

  // Size allocation for the array of best sequence of indices
  size_t allocSize_cities_int = numberCities* sizeof(int);
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_bestSoFar, allocSize_cities_int));
  h_bestSoFar = (int*) malloc(allocSize_cities_int);


  size_t allocSize_pop_fl = population* sizeof(float);
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_dis, allocSize_pop_fl));
  CUDA_SAFE_CALL(cudaMalloc((void**)&d_fitVal, allocSize_pop_fl));

  #if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Record event on the default stream
    cudaEventRecord(start, 0);
   #endif

  CUDA_SAFE_CALL(cudaMemcpy(d_genes, h_genes,allocSize2d, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_xCoord, h_xCoord,allocSize_cities_fl, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_yCoord, h_yCoord,allocSize_cities_fl, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(d_bestSoFar, h_bestSoFar,allocSize_cities_int, cudaMemcpyHostToDevice));

  setup_kernel<<<blocks, threads>>>(d_state);
  main_Kernel <<<blocks, threads>>> (d_xCoord, d_yCoord,d_dis,d_genes, d_local_popMinIndex, d_local_min_dist, d_bestSoFar, d_fitVal, d_state);

  CUDA_SAFE_CALL(cudaMemcpy(h_bestSoFar, d_bestSoFar,allocSize_cities_int, cudaMemcpyDeviceToHost));

  #if PRINT_TIME
    // Stop and destroy the timer
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_gpu, start, stop);
    printf("\nTotal time on GPU is: %f(msec)\n", elapsed_gpu);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  #endif

  for(i=0; i<numberCities; i++){
        printf("%d ", h_bestSoFar[i]);
  }

  printf("\n");

  printf("Number of Cities:%d, Population: %d, Iterations: %d, Best Value: %f \n",numberCities, population, iters, max-min);


  // printCityLoc(h_xCoord, h_yCoord);


  CUDA_SAFE_CALL(cudaFree(d_xCoord));
  CUDA_SAFE_CALL(cudaFree(d_yCoord));
  CUDA_SAFE_CALL(cudaFree(d_genes));


 free(h_genes);




  return 0;
}
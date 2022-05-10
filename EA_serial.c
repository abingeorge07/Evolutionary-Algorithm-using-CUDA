#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include<float.h>
/*
gcc EA_serial.c -lm -o EA
./EA
*/
#define totalIters 10000
int numberCities=100;
int population = 1000;

float max =0;
float min=FLT_MAX;

int xLimit= 10;
int yLimit= 0;

float global_minDis = FLT_MAX;

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

void setup(float* x, float* y, int*in)
{
    int i;
    for(i=0; i<numberCities; i++)
    {
        x[i]= ((float)rand()/(float)RAND_MAX)*xLimit;
        if(x[i]>max){
            max=x[i];
        }
        if(x[i]<min){
            min=x[i];
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

void assignGene(int genes[][numberCities], float* x, float*y, int* ind){
    int i,j;
    int random[numberCities];

    for(i=0; i<numberCities; i++){
        random[i]= ind[i];
    }

    for(i =0; i<population; i++){
        shuffle(random);
        for(j=0; j<numberCities; j++){
            genes[i][j]= random[j];
        }
    }
}

float dist(float x1, float x2, float y1, float y2){
    float temp= pow((x1-x2),2) + pow((y1-y2),2);
    temp = pow(temp,0.5);
    return temp;
}

void copyBest(int index,int genes[][numberCities], int* bestSoFar){
    int i;
    for(i=0; i<numberCities; i++){
        bestSoFar[i]= genes[index][i];
    }

}

void distanceCalc(float* x, float* y,int genes[][numberCities], float* dis, int* local_ind, float* local_dist, int* bestSoFar){
    int index=0;
    int i,j;
    float sum;
    float minDis= FLT_MAX;
    for(i=0; i<population; i++){
        sum=0;
        for(j=0; j<numberCities-1; j++){
            sum=sum+dist(x[genes[i][j]], x[genes[i][j+1]], y[genes[i][j]], y[genes[i][j+1]]);
        }
        dis[i]= sum;
        if(sum<minDis){
            minDis=sum;
            index=i;
        }
    }

    if(minDis<global_minDis){
        global_minDis = minDis;
        copyBest(index, genes, bestSoFar);
    }

    *local_ind =index;
    *local_dist = minDis;    
}


void printDis(float* dis, int popMinIndex){
    int i;
    for(i=0;i<population; i++){
        printf(" Gene %d: distance: %f\n", i, dis[i]);
    }
    printf(" Index of min distance is %d \n", popMinIndex);
}

void fitnessFun(float* fitVal , float* dis){
    float sum=0;
    int i;
    for (i=0;i<population; i++){
        sum= sum+(1/dis[i]);
    }

    for(i=0; i<population; i++){
        fitVal[i]= (1/dis[i])/sum;
    }
}

void mutate( int indexi, int genes[][numberCities], int indexj, float* fitVal, float rate, int* bestSoFar){

        // includes crossover
        int num_chosen = floor(rate*numberCities);
        int ind1 = round(((float)rand()/(float)RAND_MAX)*(numberCities-1));
        int temporary[numberCities];
        int i,j,hold, ind2;
        int flag; // 1->true, 0-> false

        for(i=0; i<num_chosen; i++){
            temporary[i]= genes[indexj][ind1];
            // temporary[i] = bestSoFar[ind1];
            if(ind1== numberCities-1){
                ind1=0;
            }
            else{
                ind1++;
            }
        }

        int next = num_chosen;
        for (i=0; i<numberCities; i++){
            hold = genes[indexi][i];
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

        for(i=0; i<numberCities; i++){
            genes[indexi][i]= temporary[i];
        }




   // do nothing, but evolve
   
    ind1 = round(((float)rand()/(float)RAND_MAX)*(numberCities-1));
    // ind2 = round(((float)rand()/(float)RAND_MAX)*(numberCities-1));
    ind2 = (ind1+1)%numberCities;
    hold = genes[indexi][ind1];
    genes[indexi][ind1]= genes[indexi][ind2];
    genes[indexi][ind2]= hold;
   
    


}

void EA(int genes[][numberCities], float* fitVal, int local_popMinIndex, int iters, int* bestSoFar){
    int i,j;
    float choose, start,rate,frac;
    int end;
    for (i=0; i<population; i++){
        // choose = ((float)rand()/(float)RAND_MAX);
        // start =0;
        // end =0;
        // for (j=0; j<population; j++){            
        //     if( (choose>start) && (choose< (start+fitVal[end]))){
        //         mutate(i,genes,local_popMinIndex, fitVal, 0.8);
        //         break;
        //     }
        //     else{
        //         start = start+fitVal[end];
        //         end=end+1;
        //     }
        // }
        frac = iters/totalIters;
        rate = ((float) rand()/ (float)RAND_MAX)*(1-frac);
        mutate(i,genes,local_popMinIndex, fitVal, rate, bestSoFar);
    }
}



void printRelevantInfo(float local_dist){
    printf("Minimum Local Dis in this Iteration is %f. \nMinimum Global Dis is %f.\n\n\n", local_dist, global_minDis);
}

void printCityLoc (float *x, float* y){
    int i;

    for(i=0; i<numberCities; i++){
        printf("Index %d: x = %f y=%f \n", i, x[i], y[i]);
    }
}

int main()
{
    srand(time(NULL));
    int i;
    float xCoord[numberCities];
    float yCoord[numberCities]; 
    int indices[numberCities];
    float dis[population];
    float fitVal[population];
    int local_popMinIndex;
    float local_min_dist;
    int bestSoFar[numberCities];
    //genes where each row represent one gene 
    int genes[population][numberCities];
    struct timespec time_start, time_stop;
    double time_stamp;


    //responsible for setting up an array of 10 coordinate points for each city
    setup(xCoord, yCoord, indices);
    assignGene(genes, xCoord, yCoord, indices);
    clock_gettime(CLOCK_REALTIME, &time_start);
    distanceCalc(xCoord, yCoord, genes, dis, &local_popMinIndex, &local_min_dist, bestSoFar);
    printf("Iteration 1\n\n");
    printRelevantInfo(local_min_dist);
    int iter=2;
    
    float bestdisever= max-min;

    while(iter<=totalIters ){//&& local_min_dist!=bestdisever){
        // printDis(dis, popMinIndex);
        fitnessFun(fitVal, dis);
        EA(genes, fitVal,local_popMinIndex,iter, bestSoFar);
        distanceCalc(xCoord, yCoord, genes, dis, &local_popMinIndex, &local_min_dist, bestSoFar);
        printf("Iteration %d\n\n", iter);
        printRelevantInfo(local_min_dist);
        iter++;
    }

    clock_gettime(CLOCK_REALTIME, &time_stop);
    time_stamp = interval(time_start, time_stop);
    
    printf("Max =%f, Min =%f\n Best Dist = %f \n\n", max, min,max-min);
    printf("Following is the best sequence: \n");

    for(i=0; i<numberCities; i++){
        printf("%d ", bestSoFar[i]);
    }

    printf("\n");

    // printCityLoc(xCoord, yCoord);
    printf("\nCPU time: %f (sec)\n\n", time_stamp);
    return 0;

}

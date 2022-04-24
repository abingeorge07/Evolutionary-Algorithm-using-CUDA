#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include<float.h>
/*
gcc EA.c -lm -o EA
./EA
*/

int numberCities=10000;
int population = 100;

int xLimit= 100;
int yLimit= 100;

float global_minDis = FLT_MAX;

void setup(float* x, float* y, int*in)
{
    int i;
    for(i=0; i<numberCities; i++)
    {
        x[i]= ((float)rand()/(float)RAND_MAX)*xLimit;
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

void mutate( int indexi, int genes[][numberCities], int indexj, float* fitVal, float rate){

    if(fitVal[indexi]>= fitVal[indexj])
    {
        // do nothing, but evolve
        int num_switches = floor(rate*numberCities);
        int i, ind1, ind2, hold;
        for(i=0; i<num_switches; i++){
            ind1 = round(((float)rand()/(float)RAND_MAX)*(numberCities-1));
            ind2 = round(((float)rand()/(float)RAND_MAX)*(numberCities-1));
            hold = genes[indexi][ind1];
            genes[indexi][ind1]= genes[indexi][ind2];
            genes[indexi][ind2]= hold;
        }
    }else{
        // includes crossover
        int num_chosen = floor(rate*numberCities);
        int ind1 = round(((float)rand()/(float)RAND_MAX)*(numberCities-1));
        int temporary[numberCities];
        int i,j,hold;
        int flag; // 1->true, 0-> false
        for(i=0; i<num_chosen; i++){
            temporary[i]= genes[indexj][ind1];

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
                genes[indexi][next];
                next++;
            }

            if(next== numberCities){
                break;
            }
        }

    }

}

void EA(int genes[][numberCities], float* fitVal){
    int i,j;
    float choose, start;
    int end;
    for (i=0; i<population; i++){
        choose = ((float)rand()/(float)RAND_MAX);
        start =0;
        end =0;
        for (j=0; j<population; j++){            
            if( (choose>start) && (choose< (start+fitVal[end]))){
                mutate(i,genes,j, fitVal, 0.1);
                break;
            }
            else{
                start = start+fitVal[end];
                end=end+1;
            }
        }
    }
}



void printRelevantInfo(float local_dist){
    printf("Minimum Local Dis in this Iteration is %f. \nMinimum Global Dis is %f.\n\n\n", local_dist, global_minDis);
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


    //responsible for setting up an array of 10 coordinate points for each city
    setup(xCoord, yCoord, indices);
    assignGene(genes, xCoord, yCoord, indices);
    distanceCalc(xCoord, yCoord, genes, dis, &local_popMinIndex, &local_min_dist, bestSoFar);
    printf("Iteration 1\n\n");
    printRelevantInfo(local_min_dist);
    int iter=2;
    
    while(iter<=100){
        // printDis(dis, popMinIndex);
        fitnessFun(fitVal, dis);
        EA(genes, fitVal);
        distanceCalc(xCoord, yCoord, genes, dis, &local_popMinIndex, &local_min_dist, bestSoFar);
        printf("Iteration %d\n\n", iter);
        printRelevantInfo(local_min_dist);
        iter++;
    }

    printf("Following is the best sequence: \n");

    for(i=0; i<numberCities; i++){
        printf("%d ", bestSoFar[i]);
    }

    printf("\n");

    return 0;

}

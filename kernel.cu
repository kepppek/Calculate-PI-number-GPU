#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>
#include "device_launch_parameters.h" //blockIdx
#include <conio.h>
#include <string>

//#define N 100000
#define P 24576


//__constant__ int S[1];

__device__ double atomicAddd(double* address, double val)
{
    unsigned long long int* address_as_ull =
        (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
            __double_as_longlong(val +
                __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}

//Функция сложения
__global__ void add(double* pi)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;

    long long  N = 1000000;
    long long S = N / 24576 + 1;
    long long start = id * S;
    long long finish = (id + 1) * S;
    double result=0;
    double s=0;
    double a = 0;
    double b = 1;
    double w = (b - a) / N;

    if (finish > N)
        finish = N;

    for (long long i = start;i < finish;i++)
    {
        s = (w * i + a) + w / 2;
        result += 4 / (1 + s * s) * w;
    }

    //pi[0] += result;
    atomicAddd(&pi[0], result);
}

int main(int argc, char* argv[])
{
    double* dev_pi;


    double pi[1];


    // int s[1] = { N / P + 1 };
     //0 - кол-во элементов выполняемое одним потоком
     //1 - шаг (b-a)/n


    clock_t t;
    t = clock();

    //Выделение памяти на устройстве
    cudaMalloc((void**)&dev_pi, sizeof(double));


    //  cudaMemcpyToSymbol(S, s, sizeof(int));

      //Копируем массивы на устройство
      //cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);


    add << <96, 256 >> > (dev_pi);

    //Хост ожидает завершения работы девайса
    //cudaDeviceSynchronize();

    //Получаем результат
    cudaMemcpy(&pi, dev_pi, sizeof(double), cudaMemcpyHostToHost);

    //Очищаем память на устройстве
    cudaFree(dev_pi);




    t = clock() - t;
    printf("\ntime %.3f\n", ((double)t) / CLOCKS_PER_SEC);
    printf("\nresult = %.15f", pi[0]);//*(1 / 1000000));


    getch();
    return 0;
}

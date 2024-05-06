#include <stdio.h>
#include <time.h>
#include <stdlib.h>

#define MAX(a, b) ((a) > (b) ? (a) : (b))

int main(void)
{
    int N = 1 << 30;
    float *x, *y;
    x = (float *)malloc(N * sizeof(float));
    y = (float *)malloc(N * sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    clock_t start;
    start = clock();
    for (int i = 0; i < N; i++) {
        y[i] = 2.0 * x[i] * y[i];
    }
    clock_t end;
    end = clock();
    double cpu_time_used = ((double)(end - start)) / CLOCKS_PER_SEC;

    

    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = MAX(maxError, abs(y[i] - 4.0f));
    }
    printf("[CLang] Calculate Time Comsuption -> count [%d]: %f seconds, maxError: %f\n", N, cpu_time_used, maxError);
    free(x);
    free(y);

    return 0;
}

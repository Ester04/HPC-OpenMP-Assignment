#include <math.h>
#include <stdio.h>

#define main benchmark_main
#include "lu_v2_steps.c"
#undef main

int main(void)
{
  DATA_TYPE actual[N][N];
  DATA_TYPE expected[N][N];
  const int sizes[] = {1, N};

  for (unsigned int sample = 0; sample < sizeof(sizes) / sizeof(sizes[0]); sample++)
  {
    const int n = sizes[sample];
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        expected[i][j] = actual[i][j] =
          (i == j ? n + 1.0 : (double)((i + 2 * j) % 7) / 10.0);

    for (int k = 0; k < n; k++)
    {
      for (int j = k + 1; j < n; j++)
        expected[k][j] /= expected[k][k];
      for (int i = k + 1; i < n; i++)
        for (int j = k + 1; j < n; j++)
          expected[i][j] -= expected[i][k] * expected[k][j];
    }

    kernel_lu(n, actual);
    for (int i = 0; i < n; i++)
      for (int j = 0; j < n; j++)
        if (!isfinite(actual[i][j]) ||
            fabs(actual[i][j] - expected[i][j]) > 1e-10 * (1.0 + fabs(expected[i][j])))
        {
          fprintf(stderr, "LU_STEP=%d n=%d element=(%d,%d): %.17g != %.17g\n",
                  LU_STEP, n, i, j, actual[i][j], expected[i][j]);
          return 1;
        }
  }
  return 0;
}

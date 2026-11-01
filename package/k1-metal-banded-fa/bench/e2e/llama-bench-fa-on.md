| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |           pp512 |       217.86 ± 13.60 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |           tg128 |         19.52 ± 0.31 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |   pp512 @ d4096 |       320.49 ± 12.64 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |   tg128 @ d4096 |         18.34 ± 0.14 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |  pp512 @ d16384 |       266.60 ± 17.64 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |  tg128 @ d16384 |         16.26 ± 2.17 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |  pp512 @ d32768 |        339.97 ± 5.57 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |  tg128 @ d32768 |         12.84 ± 0.63 |

build: 4ddc6ed6d (11985)

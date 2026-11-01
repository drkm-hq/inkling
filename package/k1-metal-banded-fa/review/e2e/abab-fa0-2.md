| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   0 |           pp512 |        273.09 ± 4.17 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   0 |           tg128 |         21.26 ± 1.13 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   0 |   pp512 @ d4096 |       335.31 ± 12.80 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   0 |   tg128 @ d4096 |         21.69 ± 0.10 |

build: 90290efad (11987)

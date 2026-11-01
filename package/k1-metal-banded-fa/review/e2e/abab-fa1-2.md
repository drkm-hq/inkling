| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |           pp512 |       300.90 ± 16.46 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |           tg128 |         19.86 ± 2.91 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |   pp512 @ d4096 |        358.21 ± 2.03 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |   tg128 @ d4096 |         15.23 ± 1.06 |

build: 90290efad (11987)

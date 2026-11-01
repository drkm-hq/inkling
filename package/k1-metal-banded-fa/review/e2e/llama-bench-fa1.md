| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |           pp512 |       244.13 ± 22.00 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |           tg128 |         21.16 ± 2.42 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |   pp512 @ d4096 |       247.27 ± 15.51 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |   tg128 @ d4096 |         17.00 ± 1.73 |

build: 90290efad (11987)

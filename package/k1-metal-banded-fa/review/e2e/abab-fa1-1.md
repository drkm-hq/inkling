| model                          |       size |     params | backend    | threads |  fa |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --: | --------------: | -------------------: |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |           pp512 |       247.80 ± 80.11 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |           tg128 |         13.50 ± 0.25 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |   pp512 @ d4096 |        314.80 ± 2.36 |
| inkling ?B IQ3_XXS - 3.0625 bpw |  91.19 GiB |   263.66 B | MTL,BLAS   |       6 |   1 |   tg128 @ d4096 |         12.18 ± 1.10 |

build: 90290efad (11987)

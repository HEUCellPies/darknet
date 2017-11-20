gcc conv_test.c  -I ../include/ -I ../src/   ../libdarknet.a   -L/usr/local/cuda/lib64 -lcuda -lcudart -lcublas -lcurand -lcudnn -lpthread  -lm  -fopenmp -g

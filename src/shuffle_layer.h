#ifndef SHUFFILE_LAYER_H
#define SHUFFILE_LAYER_H
#include "network.h"
#include "layer.h"

typedef layer shuffle_layer;


shuffle_layer make_shuffle_layer(int batch, int h, int w, int c, int groups);
void forward_shuffle_layer(const shuffle_layer l, network net);
void backward_shuffle_layer(const shuffle_layer l, network net);
void resize_shuffle_layer(shuffle_layer *l, int w, int h);

#ifdef GPU
void forward_shuffle_layer_gpu(const shuffle_layer l, network net);
void backward_shuffle_layer_gpu(const shuffle_layer l, network net);
#endif

#endif

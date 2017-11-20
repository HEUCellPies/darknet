#include "shuffle_layer.h"
#include "cuda.h"
#include "blas.h"

#include <stdio.h>

shuffle_layer make_shuffle_layer(int batch, int h, int w, int c, int groups) {
    fprintf(stderr, "shuffle ");
    shuffle_layer l = {0};
    l.type = SHUFFLE;
    l.batch = batch;
    l.h = h;
    l.w = w;
    l.c = c;
    l.groups = groups;

    l.out_w = h;
    l.out_h = w;
    l.n = c;

    int outputs = l.n * l.out_w * l.out_h;
    fprintf(stderr, " %d\n", outputs);
    l.outputs = outputs;
    l.inputs = outputs;
    l.output = calloc(outputs * batch, sizeof (float));
    l.delta = calloc(outputs * batch, sizeof (float));

    l.forward = forward_shuffle_layer;
    l.backward = backward_shuffle_layer;
#ifdef GPU
    l.forward_gpu = forward_shuffle_layer_gpu;
    l.backward_gpu = backward_shuffle_layer_gpu;

    l.delta_gpu = cuda_make_array(l.delta, outputs * batch);
    l.output_gpu = cuda_make_array(l.output, outputs * batch);
#endif
    return l;
}

void resize_shuffle_layer(shuffle_layer *l, int w, int h) {
    l->h = h;
    l->w = w;
    l->out_w = h;
    l->out_h = w;

    int outputs = l->n * l->out_w * l->out_h;
    l->outputs = outputs;
    l->inputs = l->outputs;
    l->output = realloc(l->output, l->outputs * l->batch * sizeof (float));
    l->delta = realloc(l->delta, l->outputs * l->batch * sizeof (float));

#ifdef GPU
    cuda_free(l->output_gpu);
    cuda_free(l->delta_gpu);
    l->output_gpu = cuda_make_array(l->output, l->outputs * l->batch);
    l->delta_gpu = cuda_make_array(l->delta, l->outputs * l->batch);
#endif
}

void shuffle_resize_cpu(float *output, const float *input, int group_row, int group_column, int len) {
    int i, j;
    for (i = 0; i < group_row; ++i) {
        for (j = 0; j < group_column; ++j) {
            const float *p_i = input + (i * group_column + j) * len;
            float *p_o = output + (j * group_row + i) * len;
            copy_cpu(len, p_i, 1, p_o, 1);
        }
    }
}

void forward_shuffle_layer(const shuffle_layer l, network net) {
    int feature_map_size = l.c * l.w * l.h;
    int sp_sz = l.w * l.h;

    int group_row = l.groups;
    int group_column = l.c / group_row;

    int n;
    for (n = 0; n < l.batch; n++) {
        shuffle_resize_cpu(l.output + n * feature_map_size,
                net.input + n * feature_map_size, group_row, group_column, sp_sz);
    }
}

void backward_shuffle_layer(const shuffle_layer l, network net) {
    int feature_map_size = l.c * l.w * l.h;
    int sp_sz = l.w * l.h;

    int group_column = l.groups;
    int group_row = l.c / group_column;

    int n;
    for (n = 0; n < l.batch; n++) {
        shuffle_resize_cpu(l.delta + n * feature_map_size,
                net.input + n * feature_map_size, group_row, group_column, sp_sz);
    }
}

#ifdef GPU

void shuffle_resize_gpu(float *output, const float *input, int group_row, int group_column, int len) {
    int i, j;
    for (i = 0; i < group_row; ++i) {
        for (j = 0; j < group_column; ++j) {
            const float *p_i = input + (i * group_column + j) * len;
            float *p_o = output + (j * group_row + i) * len;
            copy_gpu(len, p_i, 1, p_o, 1);
        }
    }
}

void forward_shuffle_layer_gpu(const shuffle_layer l, network net) {
    int feature_map_size = l.c * l.w * l.h;
    int sp_sz = l.w * l.h;

    int group_row = l.groups;
    int group_column = l.c / group_row;

    int n;
    for (n = 0; n < l.batch; n++) {
        shuffle_resize_gpu(l.output_gpu + n * feature_map_size,
                net.input_gpu + n * feature_map_size,
                group_row, group_column, sp_sz);
    }
}

void backward_shuffle_layer_gpu(const shuffle_layer l, network net) {
    int feature_map_size = l.c * l.w * l.h;
    int sp_sz = l.w * l.h;

    int group_column = l.groups;
    int group_row = l.c / group_column;

    int n;
    for (n = 0; n < l.batch; n++) {
        shuffle_resize_gpu(l.delta_gpu + n * feature_map_size,
                net.input_gpu + n * feature_map_size, group_row, group_column, sp_sz);
    }
}
#endif

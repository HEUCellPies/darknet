#include "convolutional_layer.h"
#include "utils.h"
#include "batchnorm_layer.h"
#include "im2col.h"
#include "col2im.h"
#include "blas.h"
#include "gemm.h"
#include <stdio.h>
#include <time.h>

#ifdef AI2
#include "xnor_layer.h"
#endif

#define checkCUDNN(status)                                                     \
  {                                                                            \
    if (status != CUDNN_STATUS_SUCCESS) {                                      \
      printf("CUDNN failure\nError: %s", cudnnGetErrorString(status));         \
    }                                                                          \
  }

void swap_binary(convolutional_layer *l) {
  float *swap = l->weights;
  l->weights = l->binary_weights;
  l->binary_weights = swap;

#ifdef GPU
  swap = l->weights_gpu;
  l->weights_gpu = l->binary_weights_gpu;
  l->binary_weights_gpu = swap;
#endif
}

void binarize_weights(float *weights, int n, int size, float *binary) {
  int i, f;
  for (f = 0; f < n; ++f) {
    float mean = 0;
    for (i = 0; i < size; ++i) {
      mean += fabs(weights[f * size + i]);
    }
    mean = mean / size;
    for (i = 0; i < size; ++i) {
      binary[f * size + i] = (weights[f * size + i] > 0) ? mean : -mean;
    }
  }
}

void binarize_cpu(float *input, int n, float *binary) {
  int i;
  for (i = 0; i < n; ++i) {
    binary[i] = (input[i] > 0) ? 1 : -1;
  }
}

void binarize_input(float *input, int n, int size, float *binary) {
  int i, s;
  for (s = 0; s < size; ++s) {
    float mean = 0;
    for (i = 0; i < n; ++i) {
      mean += fabs(input[i * size + s]);
    }
    mean = mean / n;
    for (i = 0; i < n; ++i) {
      binary[i * size + s] = (input[i * size + s] > 0) ? mean : -mean;
    }
  }
}

int convolutional_out_height(convolutional_layer l) {
  return (l.h + 2 * l.pad - l.size) / l.stride + 1;
}

int convolutional_out_width(convolutional_layer l) {
  return (l.w + 2 * l.pad - l.size) / l.stride + 1;
}

image get_convolutional_image(convolutional_layer l) {
  return float_to_image(l.out_w, l.out_h, l.out_c, l.output);
}

image get_convolutional_delta(convolutional_layer l) {
  return float_to_image(l.out_w, l.out_h, l.out_c, l.delta);
}

static size_t get_workspace_size(layer l) {
#ifdef CUDNN
  if (gpu_index >= 0) {
    size_t most = 0;
    size_t s = 0;

    cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle(), l.srcTensorDesc,
                                            l.weightDesc, l.convDesc,
                                            l.dstTensorDesc, l.fw_algo, &s);
    if (s > most)
      most = s;
    cudnnGetConvolutionBackwardFilterWorkspaceSize(
        cudnn_handle(), l.srcTensorDesc, l.ddstTensorDesc, l.convDesc,
        l.dweightDesc, l.bf_algo, &s);
    if (s > most)
      most = s;
    cudnnGetConvolutionBackwardDataWorkspaceSize(
        cudnn_handle(), l.weightDesc, l.ddstTensorDesc, l.convDesc,
        l.dsrcTensorDesc, l.bd_algo, &s);
    if (s > most)
      most = s;
    return most;
  }
#endif
  return (size_t)l.out_h * l.out_w * l.size * l.size * l.c * sizeof(float) /
         l.groups;
}

#ifdef GPU
#ifdef CUDNN

void cudnn_convolutional_setup(layer *l) {
  cudnnStatus_t status;

  status =
      cudnnSetTensor4dDescriptor(l->dsrcTensorDesc, CUDNN_TENSOR_NCHW,
                                 CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
  checkCUDNN(status);
  status = cudnnSetTensor4dDescriptor(l->ddstTensorDesc, CUDNN_TENSOR_NCHW,
                                      CUDNN_DATA_FLOAT, l->batch, l->out_c,
                                      l->out_h, l->out_w);
  checkCUDNN(status);
  status = cudnnSetFilter4dDescriptor(l->dweightDesc, CUDNN_DATA_FLOAT,
                                      CUDNN_TENSOR_NCHW, l->n, l->c / l->groups,
                                      l->size, l->size);
  checkCUDNN(status);

  status =
      cudnnSetTensor4dDescriptor(l->srcTensorDesc, CUDNN_TENSOR_NCHW,
                                 CUDNN_DATA_FLOAT, l->batch, l->c, l->h, l->w);
  checkCUDNN(status);
  status = cudnnSetTensor4dDescriptor(l->dstTensorDesc, CUDNN_TENSOR_NCHW,
                                      CUDNN_DATA_FLOAT, l->batch, l->out_c,
                                      l->out_h, l->out_w);
  checkCUDNN(status);
  status = cudnnSetTensor4dDescriptor(l->normTensorDesc, CUDNN_TENSOR_NCHW,
                                      CUDNN_DATA_FLOAT, 1, l->out_c, 1, 1);
  checkCUDNN(status);
  status = cudnnSetFilter4dDescriptor(l->weightDesc, CUDNN_DATA_FLOAT,
                                      CUDNN_TENSOR_NCHW, l->n, l->c / l->groups,
                                      l->size, l->size);
  checkCUDNN(status);
  status = cudnnSetConvolutionGroupCount(l->convDesc, l->groups);
  checkCUDNN(status);
  status = cudnnSetConvolution2dDescriptor(
      l->convDesc, l->pad, l->pad, l->stride, l->stride, 1, 1,
      CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
  checkCUDNN(status);
  status = cudnnGetConvolutionForwardAlgorithm(
      cudnn_handle(), l->srcTensorDesc, l->weightDesc, l->convDesc,
      l->dstTensorDesc, CUDNN_CONVOLUTION_FWD_PREFER_FASTEST, 0, &l->fw_algo);
  checkCUDNN(status);

  status = cudnnGetConvolutionBackwardDataAlgorithm(
      cudnn_handle(), l->weightDesc, l->ddstTensorDesc, l->convDesc,
      l->dsrcTensorDesc, CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST, 0,
      &l->bd_algo);
  checkCUDNN(status);
  status = cudnnGetConvolutionBackwardFilterAlgorithm(
      cudnn_handle(), l->srcTensorDesc, l->ddstTensorDesc, l->convDesc,
      l->dweightDesc, CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST, 0,
      &l->bf_algo);
  checkCUDNN(status);
}
#endif
#endif

convolutional_layer make_convolutional_layer(int batch, int h, int w, int c,
                                             int n, int size, int stride,
                                             int padding, int groups,
                                             ACTIVATION activation,
                                             int batch_normalize, int binary,
                                             int xnor, int adam) {
  int i;
  convolutional_layer l = {0};
  l.type = CONVOLUTIONAL;

  l.h = h;
  l.w = w;
  l.c = c;
  l.n = n;
  l.binary = binary;
  l.xnor = xnor;
  l.batch = batch;
  l.stride = stride;
  l.size = size;
  l.pad = padding;
  l.groups = groups;
  l.batch_normalize = batch_normalize;

  l.nweights = c * n * size * size / l.groups;
  l.nbiases = n;

  l.weights = calloc(l.nweights, sizeof(float));
  l.weight_updates = calloc(l.nweights, sizeof(float));

  l.biases = calloc(n, sizeof(float));
  l.bias_updates = calloc(n, sizeof(float));

  float scale = sqrt(2. / (size * size * c));
  for (i = 0; i < l.nweights; ++i)
    l.weights[i] = scale * rand_normal();
  int out_w = convolutional_out_width(l);
  int out_h = convolutional_out_height(l);
  l.out_h = out_h;
  l.out_w = out_w;
  l.out_c = n;
  l.outputs = l.out_h * l.out_w * l.out_c;
  l.inputs = l.w * l.h * l.c;

  l.output = calloc(l.batch * l.outputs, sizeof(float));
  l.delta = calloc(l.batch * l.outputs, sizeof(float));

  l.forward = forward_convolutional_layer;
  l.backward = backward_convolutional_layer;
  l.update = update_convolutional_layer;

  if (binary) {
    l.binary_weights = calloc(l.nweights, sizeof(float));
    l.cweights = calloc(l.nweights, sizeof(char));
    l.scales = calloc(n, sizeof(float));
  }

  if (xnor) {
    l.binary_weights = calloc(l.nweights, sizeof(float));
    l.binary_input = calloc(l.inputs * l.batch, sizeof(float));
  }

  if (batch_normalize) {
    l.scales = calloc(n, sizeof(float));
    l.scale_updates = calloc(n, sizeof(float));
    for (i = 0; i < n; ++i) {
      l.scales[i] = 1;
    }

    l.mean = calloc(n, sizeof(float));
    l.variance = calloc(n, sizeof(float));

    l.mean_delta = calloc(n, sizeof(float));
    l.variance_delta = calloc(n, sizeof(float));

    l.rolling_mean = calloc(n, sizeof(float));
    l.rolling_variance = calloc(n, sizeof(float));
    l.x = calloc(l.batch * l.outputs, sizeof(float));
    l.x_norm = calloc(l.batch * l.outputs, sizeof(float));
  }

  if (adam) {
    l.m = calloc(l.nweights, sizeof(float));
    l.v = calloc(l.nweights, sizeof(float));
    l.bias_m = calloc(n, sizeof(float));
    l.scale_m = calloc(n, sizeof(float));
    l.bias_v = calloc(n, sizeof(float));
    l.scale_v = calloc(n, sizeof(float));
  }

#ifdef GPU
  l.forward_gpu = forward_convolutional_layer_gpu;
  l.backward_gpu = backward_convolutional_layer_gpu;
  l.update_gpu = update_convolutional_layer_gpu;

  if (gpu_index >= 0) {
    if (adam) {
      l.m_gpu = cuda_make_array(l.m, l.nweights);
      l.v_gpu = cuda_make_array(l.v, l.nweights);
      l.bias_m_gpu = cuda_make_array(l.bias_m, n);
      l.bias_v_gpu = cuda_make_array(l.bias_v, n);
      l.scale_m_gpu = cuda_make_array(l.scale_m, n);
      l.scale_v_gpu = cuda_make_array(l.scale_v, n);
    }

    l.weights_gpu = cuda_make_array(l.weights, l.nweights);
    l.weight_updates_gpu = cuda_make_array(l.weight_updates, l.nweights);

    l.biases_gpu = cuda_make_array(l.biases, n);
    l.bias_updates_gpu = cuda_make_array(l.bias_updates, n);

    l.delta_gpu = cuda_make_array(l.delta, l.batch * out_h * out_w * n);
    l.output_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);

    if (binary) {
      l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
    }
    if (xnor) {
      l.binary_weights_gpu = cuda_make_array(l.weights, l.nweights);
      l.binary_input_gpu = cuda_make_array(0, l.inputs * l.batch);
    }

    if (batch_normalize) {
      l.mean_gpu = cuda_make_array(l.mean, n);
      l.variance_gpu = cuda_make_array(l.variance, n);

      l.rolling_mean_gpu = cuda_make_array(l.mean, n);
      l.rolling_variance_gpu = cuda_make_array(l.variance, n);

      l.mean_delta_gpu = cuda_make_array(l.mean, n);
      l.variance_delta_gpu = cuda_make_array(l.variance, n);

      l.scales_gpu = cuda_make_array(l.scales, n);
      l.scale_updates_gpu = cuda_make_array(l.scale_updates, n);

      l.x_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);
      l.x_norm_gpu = cuda_make_array(l.output, l.batch * out_h * out_w * n);
    }

#ifdef CUDNN
    cudnnStatus_t status;
    status = cudnnCreateTensorDescriptor(&l.normTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateTensorDescriptor(&l.srcTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateTensorDescriptor(&l.dstTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateFilterDescriptor(&l.weightDesc);
    checkCUDNN(status);
    status = cudnnCreateTensorDescriptor(&l.dsrcTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateTensorDescriptor(&l.ddstTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateFilterDescriptor(&l.dweightDesc);
    checkCUDNN(status);
    status = cudnnCreateConvolutionDescriptor(&l.convDesc);
    checkCUDNN(status);
    cudnn_convolutional_setup(&l);
#endif
  }
#endif
  l.workspace_size = get_workspace_size(l);
  l.activation = activation;

  fprintf(stderr,
          "conv  %5d %2d x%2d /%2d  %4d x%4d x%4d   ->  %4d x%4d x%4d\n", n,
          size, size, stride, w, h, c, l.out_w, l.out_h, l.out_c);

  return l;
}

void denormalize_convolutional_layer(convolutional_layer l) {
  int i, j;
  for (i = 0; i < l.n; ++i) {
    float scale = l.scales[i] / sqrt(l.rolling_variance[i] + .00001);
    for (j = 0; j < l.c * l.size * l.size / l.groups; ++j) {
      l.weights[i * l.c * l.size * l.size / l.groups + j] *= scale;
    }
    l.biases[i] -= l.rolling_mean[i] * scale;
    l.scales[i] = 1;
    l.rolling_mean[i] = 0;
    l.rolling_variance[i] = 1;
  }
}

void resize_convolutional_layer(convolutional_layer *l, int w, int h) {
  l->w = w;
  l->h = h;
  int out_w = convolutional_out_width(*l);
  int out_h = convolutional_out_height(*l);

  l->out_w = out_w;
  l->out_h = out_h;

  l->outputs = l->out_h * l->out_w * l->out_c;
  l->inputs = l->w * l->h * l->c;

  l->output = realloc(l->output, l->batch * l->outputs * sizeof(float));
  l->delta = realloc(l->delta, l->batch * l->outputs * sizeof(float));
  if (l->batch_normalize) {
    l->x = realloc(l->x, l->batch * l->outputs * sizeof(float));
    l->x_norm = realloc(l->x_norm, l->batch * l->outputs * sizeof(float));
  }

#ifdef GPU
  cuda_free(l->delta_gpu);
  cuda_free(l->output_gpu);

  l->delta_gpu = cuda_make_array(l->delta, l->batch * l->outputs);
  l->output_gpu = cuda_make_array(l->output, l->batch * l->outputs);

  if (l->batch_normalize) {
    cuda_free(l->x_gpu);
    cuda_free(l->x_norm_gpu);

    l->x_gpu = cuda_make_array(l->output, l->batch * l->outputs);
    l->x_norm_gpu = cuda_make_array(l->output, l->batch * l->outputs);
  }
#ifdef CUDNN
  cudnn_convolutional_setup(l);
#endif
#endif
  l->workspace_size = get_workspace_size(*l);
}

void add_bias(float *output, float *biases, int batch, int n, int size) {
  int i, j, b;
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < size; ++j) {
        output[(b * n + i) * size + j] += biases[i];
      }
    }
  }
}

void scale_bias(float *output, float *scales, int batch, int n, int size) {
  int i, j, b;
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      for (j = 0; j < size; ++j) {
        output[(b * n + i) * size + j] *= scales[i];
      }
    }
  }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n,
                   int size) {
  int i, b;
  for (b = 0; b < batch; ++b) {
    for (i = 0; i < n; ++i) {
      //每一个输出通道的delta之和的累积
      bias_updates[i] += sum_array(delta + size * (i + b * n), size);
    }
  }
}

void forward_convolutional_layer(convolutional_layer l, network net) {
  fill_cpu(l.outputs * l.batch, 0, l.output, 1);

  if (l.xnor) {
    binarize_weights(l.weights, l.n, l.c / l.groups * l.size * l.size, l.binary_weights);
    swap_binary(&l);
    binarize_cpu(net.input, l.c * l.h * l.w * l.batch, l.binary_input);
    net.input = l.binary_input;
  }

  // image im = float_to_image(l.w, l.h, l.c, net.input);
  // printf("\nfilter_before:\n");
  // print_image(im);

  int m = l.n;                   // output channel
  int k = l.size * l.size * l.c; // kernel size, input channel
  int n = l.out_h * l.out_w;     // output size

  float *a = l.weights;
  float *c = l.output;

  int group_size = l.c / l.groups;
  int group_step = l.h * l.w * group_size;
  k = k / l.groups;
  m = m / l.groups;
  int i, j;
  for (i = 0; i < l.batch; ++i) {
    for (j = 0; j < l.groups; j++) {
      float *aoffset = a + j * k;
      float *boffset = net.workspace;
      float *coffset = c + j * n * group_size;
      float *inputoffset = net.input + group_step * j;
      im2col_cpu(inputoffset, group_size, l.h, l.w, l.size, l.stride, l.pad,
                 boffset);
      gemm(0, 0, m, n, k, 1, aoffset, k, boffset, n, 1, coffset, n);
    }

    c += l.out_h * l.out_w * l.n;
    net.input += l.c * l.h * l.w;
  }

  // im = float_to_image(l.out_w, l.out_h, l.out_c, l.output);
  // printf("\nfilter:\n");
  // print_image(im);

  // im = float_to_image(l.size, l.size, l.n, l.weights);
  // printf("\nweights:\n");
  // print_image(im);

  if (l.batch_normalize) {
    forward_batchnorm_layer(l, net);
  } else {
    add_bias(l.output, l.biases, l.batch, l.n, l.out_w * l.out_h);
  }

  activate_array(l.output, l.outputs * l.batch, l.activation);
  if (l.binary || l.xnor)
    swap_binary(&l);
}

void backward_convolutional_layer(convolutional_layer l, network net) {
  int i, j;
  int m = l.n;
  int n = l.size * l.size * l.c;
  int k = l.out_w * l.out_h;

  gradient_array(l.output, m * k * l.batch, l.activation, l.delta);

  if (l.batch_normalize) {
    backward_batchnorm_layer(l, net);
  } else {
    backward_bias(l.bias_updates, l.delta, l.batch, l.n, k);
  }

  int group_size = l.c / l.groups;
  int group_step = l.h * l.w * group_size;
  n = n / l.groups;
  m = m / l.groups;
  for (i = 0; i < l.batch; ++i) {
    float *input_data = net.input + i * l.c * l.h * l.w;
    float *deltas = l.delta + i * l.n * l.out_w * l.out_h;
    float *outdeltas = net.delta + i * l.c * l.w * l.h;
    for (j = 0; j < l.groups; j++) {
      float *im = input_data + j * group_step;
      float *aoffset = deltas + j * group_size * k;
      float *boffset = net.workspace;
      float *coffset = l.weight_updates + j * n;

      //得到权重的更新
      im2col_cpu(im, group_size, l.h, l.w, l.size, l.stride, l.pad, boffset);
      gemm(0, 1, m, n, k, 1, aoffset, k, boffset, k, 1, coffset, n);

      if (net.delta) {
        aoffset = l.weights + j * n;
        boffset = deltas + j * group_size * k;
        coffset = net.workspace;

        gemm(1, 0, n, k, m, 1, aoffset, n, boffset, k, 0, coffset, k);
        col2im_cpu(net.workspace, group_size, l.h, l.w, l.size, l.stride, l.pad,
                   outdeltas + j * group_step);
      }
    }
  }
}

void update_convolutional_layer(convolutional_layer l, update_args a) {
  float learning_rate = a.learning_rate * l.learning_rate_scale;
  float momentum = a.momentum;
  float decay = a.decay;
  int batch = a.batch;

  axpy_cpu(l.n, learning_rate / batch, l.bias_updates, 1, l.biases, 1);
  scal_cpu(l.n, momentum, l.bias_updates, 1);

  if (l.scales) {
    axpy_cpu(l.n, learning_rate / batch, l.scale_updates, 1, l.scales, 1);
    scal_cpu(l.n, momentum, l.scale_updates, 1);
  }

  axpy_cpu(l.nweights, -decay * batch, l.weights, 1, l.weight_updates, 1);
  axpy_cpu(l.nweights, learning_rate / batch, l.weight_updates, 1, l.weights,
           1);
  scal_cpu(l.nweights, momentum, l.weight_updates, 1);
}

image get_convolutional_weight(convolutional_layer l, int i) {
  int h = l.size;
  int w = l.size;
  int c = l.c / l.groups;
  return float_to_image(w, h, c, l.weights + i * h * w * c);
}

void rgbgr_weights(convolutional_layer l) {
  int i;
  for (i = 0; i < l.n; ++i) {
    image im = get_convolutional_weight(l, i);
    if (im.c == 3) {
      rgbgr_image(im);
    }
  }
}

void rescale_weights(convolutional_layer l, float scale, float trans) {
  int i;
  for (i = 0; i < l.n; ++i) {
    image im = get_convolutional_weight(l, i);
    if (im.c == 3) {
      scale_image(im, scale);
      float sum = sum_array(im.data, im.w * im.h * im.c);
      l.biases[i] += sum * trans;
    }
  }
}

image *get_weights(convolutional_layer l) {
  image *weights = calloc(l.n, sizeof(image));
  int i;
  for (i = 0; i < l.n; ++i) {
    weights[i] = copy_image(get_convolutional_weight(l, i));
    normalize_image(weights[i]);
    /*
       char buff[256];
       sprintf(buff, "filter%d", i);
       save_image(weights[i], buff);
     */
  }
  // error("hey");
  return weights;
}

image *visualize_convolutional_layer(convolutional_layer l, char *window,
                                     image *prev_weights) {
  image *single_weights = get_weights(l);
  show_images(single_weights, l.n, window);

  image delta = get_convolutional_image(l);
  image dc = collapse_image_layers(delta, 1);
  char buff[256];
  sprintf(buff, "%s: Output", window);
  // show_image(dc, buff);
  // save_image(dc, buff);
  free_image(dc);
  return single_weights;
}

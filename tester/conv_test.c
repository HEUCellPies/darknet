
#include <nnpack.h>
#include "../src/convolutional_layer.h"
#include "conv_test.h"
#include "cuda.h"

void test_1x1_convolutional_layer() {
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 1, 1, 1, 0, 1, LEAKY, 0, 0, 0, 0);

    float data[] = {
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
    };

    network net = make_network(1);
    net.h = 5;
    net.w = 5;
    net.c = 3;
    net.workspace = calloc(1, l.workspace_size);
    net.batch = 1;
    net.input = data;

    {
        memset(l.output, 0, sizeof (float) * l.out_h * l.out_w * l.out_c);
        //forward_convolutional_layer_gpu(l, net);
        int h = l.out_h;
        int w = l.out_w;
        int c = l.n;

        image iw = float_to_image(l.size, l.size, l.n, l.weights);
        printf("weihts:\n");
        print_image(iw);

        image im = float_to_image(w, h, c, l.output);
        printf("dw_3x3 filter:\n");
        print_image(im);
        fflush(stderr);
    }

}

void test_depthwise_convolutional_layer() {
    convolutional_layer l = make_convolutional_layer(1, 5, 5, 3, 3, 3, 1, 1, 3, LEAKY, 0, 0, 0, 0);

    float data[] = {1, 1, 1, 1, 1,
        1, 2, 1, 1, 1,
        1, 1, 3, 1, 1,
        1, 1, 1, 4, 1,
        1, 1, 1, 1, 5,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 2,
        4, 8, 2, 2, 2,
        2, 2, 2, 2, 2,
        2, 2, 2, 2, 9,
        3, 3, 3, 3, 3,
        5, 3, 3, 7, 3,
        3, 3, 3, 3, 3,
        3, 3, 3, 3, 3,
        4, 3, 3, 3, 3};

    network net = make_network(1);
    net.h = 5;
    net.w = 5;
    net.c = 3;
    net.workspace = calloc(1, l.workspace_size);
    net.batch = 1;

    net.input = data;


    {

        memset(l.output, 0, sizeof (float) * l.out_h * l.out_w * l.out_c);
        //forward_convolutional_layer_gpu(l, net);
        int h = l.out_h;
        int w = l.out_w;
        int c = l.n;

        image iw = float_to_image(l.size, l.size, l.n, l.weights);
        printf("weihts:\n");
        print_image(iw);

        image im = float_to_image(w, h, c, l.output);
        printf(" filter:\n");
        print_image(im);
        fflush(stderr);
    }
}

#define checkCUDNN(status) { \
    if (status != CUDNN_STATUS_SUCCESS) { \
      printf("CUDNN failure\nError: %s", cudnnGetErrorString(status)); \
    }                                                                  \
}

void test_cudnn() {
    cudnnHandle_t cudnnHandle;
    checkCUDNN( cudnnCreate(&cudnnHandle) );
    cudnnTensorDescriptor_t srcTensorDesc, dstTensorDesc;
    cudnnTensorDescriptor_t dsrcTensorDesc, ddstTensorDesc;
    cudnnTensorDescriptor_t normTensorDesc;
    cudnnFilterDescriptor_t weightDesc;
    cudnnFilterDescriptor_t dweightDesc;
    cudnnConvolutionDescriptor_t convDesc;
    cudnnConvolutionFwdAlgo_t fw_algo;
    cudnnConvolutionBwdDataAlgo_t bd_algo;
    cudnnConvolutionBwdFilterAlgo_t bf_algo;

    cudnnStatus_t status;
    status = cudnnCreateTensorDescriptor(&normTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateTensorDescriptor(&srcTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateTensorDescriptor(&dstTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateFilterDescriptor(&weightDesc);
    checkCUDNN(status);
    status = cudnnCreateTensorDescriptor(&dsrcTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateTensorDescriptor(&ddstTensorDesc);
    checkCUDNN(status);
    status = cudnnCreateFilterDescriptor(&dweightDesc);
    checkCUDNN(status);
    status = cudnnCreateConvolutionDescriptor(&convDesc);
    checkCUDNN(status);

    int batch = 1;
    int c = 3, h = 416, w = 416;
    int out_c = 16, out_h = 416, out_w = 416;
    int n = 16;
    int size = 3;
    int pad = 1;
    int stride = 1;
    int groups = 1;

    status = cudnnSetTensor4dDescriptor(dsrcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, c, h, w);
    checkCUDNN(status);
    status = cudnnSetTensor4dDescriptor(ddstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, out_c, out_h, out_w);
    checkCUDNN(status);
    status = cudnnSetFilter4dDescriptor(dweightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, size, size);
    checkCUDNN(status);

    status = cudnnSetTensor4dDescriptor(srcTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, c, h, w);
    checkCUDNN(status);
    status = cudnnSetTensor4dDescriptor(dstTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, batch, out_c, out_h, out_w);
    checkCUDNN(status);
    status = cudnnSetTensor4dDescriptor(normTensorDesc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, 1, out_c, 1, 1);
    checkCUDNN(status);
    status = cudnnSetFilter4dDescriptor(weightDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, n, c, size, size); 
    checkCUDNN(status);
    status = cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    checkCUDNN(status);

    status = cudnnSetConvolution2dDescriptor(convDesc, pad, pad, stride, stride, 1, 1, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT);
    checkCUDNN(status);
    status = cudnnSetConvolutionGroupCount(convDesc, groups);
    checkCUDNN(status);
    printf("\ncudnn handle:%d\n", cudnn_handle());

    status = cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
            srcTensorDesc,
            weightDesc,
            convDesc,
            dstTensorDesc,
            CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
            CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
            &fw_algo);
    checkCUDNN(status);

    status = cudnnGetConvolutionBackwardDataAlgorithm(cudnnHandle,
            weightDesc,
            ddstTensorDesc,
            convDesc,
            dsrcTensorDesc,
            CUDNN_CONVOLUTION_BWD_DATA_PREFER_FASTEST,
            0,
            &bd_algo);
    checkCUDNN(status);
    status = cudnnGetConvolutionBackwardFilterAlgorithm(cudnnHandle,
            srcTensorDesc,
            ddstTensorDesc,
            convDesc,
            dweightDesc,
            CUDNN_CONVOLUTION_BWD_FILTER_PREFER_FASTEST,
            0,
            &bf_algo);
    checkCUDNN(status);
}

int main() {
    checkCUDNN(cudaSetDevice(0));
    test_cudnn();
    checkCUDNN(cudaDeviceReset());
}

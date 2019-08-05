//
// Created by Bryan Wong on 2019-08-03.
//

#ifndef ARRAYFIRE_TPCDI_OPENCL_PARSERS_H
#define ARRAYFIRE_TPCDI_OPENCL_PARSERS_H
#include <arrayfire.h>
#include <include/AFTypes.h>
#include "opencl_helper.h"
#include "include/TPCDI_Utils.h"
#ifdef IS_APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#ifndef ULL
    #define ULL
typedef unsigned long long ull;
#endif

template<typename T>
static void launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const loop_len, ull const row_num) {
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu -D PARSE_TYPE=%s", loop_len, GetAFType<T>().str);
    // Build the OpenCL program and get the kernel
    cl_program program = build_parse_program(context, options);
    cl_kernel kernel = create_kernel(program, "parser");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &row_num);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    auto num = row_num;
    // Set launch configuration parameters and launch kernel
    size_t local  = 256;
    size_t global = local * (num / local + ((num % local) ? 1 : 0));
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to enqueue kernel\n", err);
        throw (err);
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Kernel failed to finish\n", err);
        throw (err);
    }
}

template<typename T>
void inline numericParse(af::array &output, af::array const &input, af::array const &indexer) {
    using namespace af;

    auto const loop_length = sum<ull>(max(indexer.row(1), 1));
    auto const row_nums = indexer.elements() / 2;
    output = array(1, row_nums + 1, GetAFType<T>().af_type);
    auto out_ptr = output.device<T>();
    auto idx_ptr = indexer.device<ull>();
    auto in_ptr = input.device<unsigned char>();
    af::sync();

    launchNumericParse(out_ptr, idx_ptr, in_ptr, loop_length, row_nums);

    output.unlock();
    input.unlock();
    indexer.unlock();
    output = output.cols(0, end - 1);
    output.eval();
}

static void launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const loop_len, ull const output_size, ull const row_num) {

    static auto KERNELS = get_kernel_string();
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu", loop_len);
    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context, KERNELS, options);
    cl_kernel kernel = create_kernel(program, "string_gather");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, arg++, sizeof(ull), &output_size);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &row_num);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    auto num = row_num;
    // Set launch configuration parameters and launch kernel
    size_t local  = 256;
    size_t global = local * (num / local + ((num % local) ? 1 : 0));
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to enqueue kernel\n", err);
        throw (err);
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Kernel failed to finish\n", err);
        throw (err);
    }
}

void inline stringGather(af::array &output, af::array const &input, af::array const &indexer) {
    using namespace af;
    auto const loop_length = sum<ull>(max(indexer.row(1), 1));
    auto const out_length = sum<ull>(indexer.row(1));
    auto const row_nums = indexer.elements() / 3;
    output = array(out_length, u8);

    auto out_ptr = output.device<unsigned char>();
    auto in_ptr = input.device<unsigned char>();
    auto idx_ptr = indexer.device<ull>();
    af::sync();

    launchStringGather(out_ptr, idx_ptr, in_ptr, loop_length, out_length, row_nums);

    output.unlock();
    input.unlock();
    indexer.unlock();
}

#endif //ARRAYFIRE_TPCDI_OPENCL_PARSERS_H

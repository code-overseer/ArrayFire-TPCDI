#ifndef ARRAYFIRE_TPCDI_OPENCL_PARSERS_H
#define ARRAYFIRE_TPCDI_OPENCL_PARSERS_H
#ifdef IS_APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif
#include <arrayfire.h>
#include "opencl_helper.h"
#include <include/AFTypes.h>
#include "include/TPCDI_Utils.h"
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
af::array inline numericParse(af::array const &input, af::array const &indexer) {
    using namespace af;

    auto const loop_length = sum<ull>(max(indexer.row(1), 1));
    auto const row_nums = indexer.elements() / 2;
    auto output = array(1, row_nums + 1, GetAFType<T>().af_type);
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
    return output;
}

static void launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const loop_len,
        ull const output_size, ull const row_num) {
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu", loop_len);
    // Build the OpenCL program and get the kernel
    cl_program program = build_parse_program(context, options);
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

af::array inline stringGather(af::array const &input, af::array &indexer) {
    using namespace af;
    indexer = join(0, indexer, scan(indexer.row(1), 1, AF_BINARY_ADD, false));
    auto const loop_length = sum<ull>(max(indexer.row(1), 1));
    auto const out_length = sum<ull>(indexer.row(1));
    auto output = array(out_length, u8);

    #ifdef USING_AF
    for (ull i = 0; i < loop_length; ++i) {
        auto b = indexer.row(1) > i;
        af::array o_idx = indexer(2, b) + i;
        af::array i_idx = indexer(0, b) + i;
        output(o_idx) = (array)input(i_idx);
    }
    output.eval();
    #else
    auto const row_nums = indexer.elements() / 3;
    auto out_ptr = output.device<unsigned char>();
    auto in_ptr = input.device<unsigned char>();
    auto idx_ptr = indexer.device<ull>();
    af::sync();

    launchStringGather(out_ptr, idx_ptr, in_ptr, loop_length, out_length, row_nums);

    output.unlock();
    input.unlock();
    indexer.unlock();
    #endif
    indexer.row(0) = (array)indexer.row(2);
    indexer = indexer.rows(0, 1);
    indexer.eval();
    return output;
}



af::array inline stringComp(Column const &lhs, Column const &rhs) {
    using namespace af;
    auto out = lhs.index().row(1) == rhs.index().row(1);
    auto loop_length = sum<ull>(max(lhs.index()(1, out)));

    for (ull i = 0; i < loop_length; ++i) {
        out = flat(out) && (flat(lhs.index().row(1) <= i) || flat(lhs(lhs.index().row(0) + i) == rhs(
                rhs.index().row(0) + i)) );
    }
    out.eval();

    return out;
}

#endif //ARRAYFIRE_TPCDI_OPENCL_PARSERS_H

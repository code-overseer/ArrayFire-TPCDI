#ifndef ARRAYFIRE_TPCDI_OPENCL_KERNELS_H
#define ARRAYFIRE_TPCDI_OPENCL_KERNELS_H

#include "opencl_helper.h"
#include <arrayfire.h>
#ifndef ULL
    #define ULL
    typedef unsigned long long ull;
#endif


void inline launchIntersect(ull *result, ull const *input, ull const *comparison, ull const bag_size, ull const set_size) {
    static auto KERNELS = get_kernel_string();
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)result);
    cl_command_queue queue = create_queue(context);
    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context, KERNELS);

    cl_kernel kernel = create_kernel(program, "is_exist_kernel");
    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &result);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &comparison);
    err |= clSetKernelArg(kernel, arg++, sizeof(ull), &bag_size);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &set_size);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    auto num = bag_size * set_size;
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

void inline lauchJoinScatter(ull const *il, ull const *ir, ull const *cl, ull const *cr, ull const *outpos,
                             ull *l, ull *r, ull const equals, ull const left_max, ull const right_max, ull const out_size) {
    static auto KERNELS = get_kernel_string();
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)il);
    cl_command_queue queue = create_queue(context);

    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context, KERNELS);
    cl_kernel kernel = create_kernel(program, "join_scatter");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &il);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &ir);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &cl);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &cr);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &outpos);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &r);
    err |= clSetKernelArg(kernel, arg++, sizeof(ull), &equals);
    err |= clSetKernelArg(kernel, arg++, sizeof(ull), &left_max);
    err |= clSetKernelArg(kernel, arg++, sizeof(ull), &right_max);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &out_size);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    auto num = left_max * right_max * equals;
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

void inline launchStringGather(unsigned char *output, ull const *idx, unsigned char *input, ull const loop_len, ull const output_size, ull const row_nums) {

    static auto KERNELS = get_kernel_string();
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu -D ROW_NUMS=%llu", loop_len, row_nums);
    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context, KERNELS, options);
    cl_kernel kernel = create_kernel(program, "string_gather");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &output_size);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    auto num = row_nums;
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

void inline bagSetIntersect(af::array &bag, af::array const &set) {
    using namespace af;
    auto const bag_size = bag.row(0).elements();
    auto const set_size = set.elements();
    auto result = constant(0, dim4(1, bag_size + 1), u64);
#ifdef USING_AF
    auto id = range(dim4(1, bag_size * set_size), 1, u64);
    auto i = id / set_size;
    auto j = id % set_size;
    printf("SetSize: %llu\n", set_size);
    af_print(set(0))
    af_print(bag(0))
    auto b = moddims(set(j), i.dims()) == moddims(bag(0, i), i.dims());
    auto k = b * i + !b * bag_size;
    result(k) = 1;
#else
    auto result_ptr = result.device<ull>();
    auto set_ptr = set.device<ull>();
    auto bag_ptr = bag.device<ull>();
    af::sync();

    launchIntersect(result_ptr, bag_ptr, set_ptr, bag_size, set_size);

    bag.unlock();
    set.unlock();
    result.unlock();
#endif
    result = result.cols(0, end - 1);
    bag = bag(span, where(result));
    bag.eval();
}

void inline joinScatter(af::array &lhs, af::array &rhs, ull const equals) {
    using namespace af;
    using namespace TPCDI_Utils;
    auto left_count = accum(join(1, constant(1, 1, u64), (diff1(lhs.row(0), 1) > 0).as(u64)), 1) - 1;
    left_count = flipdims(histogram(left_count, left_count.elements())).as(u64);
    left_count = left_count(left_count > 0);
    auto left_max = sum<unsigned int>(max(left_count, 1));
    auto left_idx = scan(left_count, 1, AF_BINARY_ADD, false);

    auto right_count = accum(join(1, constant(1, 1, u64), (diff1(rhs.row(0), 1) > 0).as(u64)), 1) - 1;
    right_count = flipdims(histogram(right_count, right_count.elements())).as(u64);
    right_count = right_count(right_count > 0);
    auto right_max = sum<unsigned int>(max(right_count, 1));
    auto right_idx = scan(right_count, 1, AF_BINARY_ADD, false);

    auto output_pos = right_count * left_count;
    auto output_size = sum<ull>(output_pos);
    output_pos = scan(output_pos, 1, AF_BINARY_ADD, false);
    array left_out(1, output_size + 1, u64);
    array right_out(1, output_size + 1, u64);
#ifdef USING_AF
    auto i = range(dim4(1, equals * left_max * right_max), 1, u64);
    auto j = i / right_max % left_max;
    auto k = i % right_max;
    i = i / left_max / right_max;
    auto b = !(j / left_count(i)) && !(k / right_count(i));
    left_out(b * (output_pos(i) + left_count(i) * k + j) + !b * output_size) = left_idx(i) + j;
    right_out(b * (output_pos(i) + right_count(i) * j + k) + !b * output_size) = right_idx(i) + k;
#else
    auto idx_l = left_idx.device<ull>();
    auto idx_r = right_idx.device<ull>();
    auto count_l = left_count.device<ull>();
    auto count_r = right_count.device<ull>();
    auto pos = output_pos.device<ull>();
    auto left = left_out.device<ull>();
    auto right = right_out.device<ull>();
    af::sync();

    lauchJoinScatter(idx_l, idx_r, count_l, count_r, pos, left, right, equals, left_max, right_max, output_size);

    left_idx.unlock();
    right_idx.unlock();
    left_count.unlock();
    right_count.unlock();
    output_pos.unlock();
    left_out.unlock();
    right_out.unlock();
#endif
    left_out = left_out.cols(0, end - 1);
    right_out = right_out.cols(0, end - 1);
    lhs = lhs(1, left_out);
    rhs = rhs(1, right_out);
    lhs.eval();
    rhs.eval();
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

#endif

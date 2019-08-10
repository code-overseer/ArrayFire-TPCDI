#ifndef ARRAYFIRE_TPCDI_OPENCL_KERNELS_H
#define ARRAYFIRE_TPCDI_OPENCL_KERNELS_H
#include "opencl_helper.h"
#include "include/AFTypes.h"
#include <arrayfire.h>
#ifndef ULL
#define ULL
typedef unsigned long long ull;
#endif

#define LOCAL_GROUP_SIZE 256

void inline launchBagSet(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
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
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bag);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &set);
    err |= clSetKernelArg(kernel, arg++, sizeof(ull), &bag_size);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &set_size);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    auto num = bag_size * set_size;
    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
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

void inline lauchJoinScatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
                             ull *l, ull *r, ull const equals, ull const left_max, ull const right_max, ull const out_size) {
    static auto KERNELS = get_kernel_string();
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)l_idx);
    cl_command_queue queue = create_queue(context);

    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context, KERNELS);
    cl_kernel kernel = create_kernel(program, "join_scatter");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l_idx);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &r_idx);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l_cnt);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &r_cnt);
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
    size_t local = LOCAL_GROUP_SIZE;
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
void inline launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows, ull const loops) {
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu -D PARSE_TYPE=%s", loops, GetAFType<T>().str);
    // Build the OpenCL program and get the kernel
    cl_program program = build_parse_program(context, options);
    cl_kernel kernel = create_kernel(program, "parser");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &rows);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }
    
    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (rows / local + ((rows % local) ? 1 : 0));
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

void inline launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const output_size,
        ull const rows, ull const loops) {
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu", loops);
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
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &rows);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (rows / local + ((rows % local) ? 1 : 0));
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

void inline launchStringComp(bool *output, unsigned char const *left, unsigned char const *right,
                             ull const *l_idx, ull const *r_idx, ull const rows, ull const loops) {
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu", loops);
    // Build the OpenCL program and get the kernel
    cl_program program = build_parse_program(context, options);
    cl_kernel kernel = create_kernel(program, "str_cmp");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &left);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &right);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l_idx);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &r_idx);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &rows);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    auto num = rows;
    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
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

void inline launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
                             ull const rows, ull const loops) {
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu", loops);
    // Build the OpenCL program and get the kernel
    cl_program program = build_parse_program(context, options);
    cl_kernel kernel = create_kernel(program, "str_cmp_single");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &left);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &right);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l_idx);
    err |= clSetKernelArg(kernel, arg, sizeof(ull), &rows);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (rows / local + ((rows % local) ? 1 : 0));
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

#endif

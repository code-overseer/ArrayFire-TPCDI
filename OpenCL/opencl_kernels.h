#ifndef ARRAYFIRE_TPCDI_OPENCL_KERNELS_H
#define ARRAYFIRE_TPCDI_OPENCL_KERNELS_H

#include "opencl_helper.h"

void inline launch_IsExist(uint64_t *result, uint64_t const *input, uint64_t const *comparison, uint64_t const i_size, uint64_t const comp_size) {
    std::string is_exist_kernel = get_kernel_string(OCL_KERNEL_DIR"/opencl_kernels.cl");
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)result);
    cl_command_queue queue = create_queue(context);

    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context, is_exist_kernel);
    cl_kernel kernel = create_kernel(program, "is_exist_kernel");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &result);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel, arg++, sizeof(cl_mem), &comparison);
    err |= clSetKernelArg(kernel, arg++, sizeof(uint64_t), &i_size);
    err |= clSetKernelArg(kernel, arg, sizeof(uint64_t), &comp_size);

    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw (err);
    }

    auto num = i_size * comp_size;
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

#endif

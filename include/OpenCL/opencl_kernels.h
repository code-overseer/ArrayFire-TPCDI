#ifndef ARRAYFIRE_TPCDI_OPENCL_KERNELS_H
#define ARRAYFIRE_TPCDI_OPENCL_KERNELS_H
#include "opencl_helper.h"
#include "include/AFTypes.h"
#include <arrayfire.h>
#include <exception>
#ifndef ULL
#define ULL
typedef unsigned long long ull;
#endif

#define LOCAL_GROUP_SIZE 256

void inline launchBagSet(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)result);
    cl_command_queue queue = create_queue(context);
    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context);
    cl_kernel kernel = create_kernel(program, "intersect_kernel");
    ull work_items = bag_size * set_size;
    int32_t groups = (work_items / (ull)INT32_MAX) + ((work_items % (ull)INT32_MAX) ? 1Ull : 0Ull);
    auto remainder = bag_size % (INT32_MAX / set_size);
    cl_int err = CL_SUCCESS;
    for (int i = 0; i < groups; ++i) {
        ull b_size = ((i == groups - 1) && remainder) ? remainder: (INT32_MAX / set_size);
        ull offset = i * (INT32_MAX / set_size) * set_size;
        int arg = 0;
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &result);
        if (err != CL_SUCCESS) goto ARG_FAIL;
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bag);
        if (err != CL_SUCCESS) goto ARG_FAIL;
        err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &set);
        if (err != CL_SUCCESS) goto ARG_FAIL;
        err = clSetKernelArg(kernel, arg++, sizeof(ull), &b_size);
        if (err != CL_SUCCESS) goto ARG_FAIL;
        err = clSetKernelArg(kernel, arg++, sizeof(ull), &set_size);
        if (err != CL_SUCCESS) goto ARG_FAIL;
        err = clSetKernelArg(kernel, arg, sizeof(ull), &offset);
        if (err != CL_SUCCESS) {
            ARG_FAIL:
            sprintf(msg, "OpenCL Error(%d): Failed to set kernel arguments\n", err);
            throw std::runtime_error(msg);
        }
        // Set launch configuration parameters and launch kernel
        auto num = b_size * set_size;
        size_t local = LOCAL_GROUP_SIZE;
        size_t global = local * (num / local + ((num % local) ? 1 : 0));
        err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            sprintf(msg, "OpenCL Error(%d): Failed to enqueue kernel\n", err);
            throw std::runtime_error(msg);
        }
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Kernel failed to finish\n", err);
        throw std::runtime_error(msg);
    }

}

void inline launchHashIntersect(char *result, ull const *bag, ull const *ht_val, ull const *ht_ptr, ull const *ht_occ,
                                unsigned int const buckets, ull const bag_size) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)result);
    cl_command_queue queue = create_queue(context);

    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context);
    cl_kernel kernel = create_kernel(program, "hash_intersect");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &result);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &bag);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &ht_val);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &ht_ptr);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &ht_occ);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(unsigned int), &buckets);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg, sizeof(ull), &bag_size);
    if (err != CL_SUCCESS) {
        ARG_FAIL:
        sprintf(msg, "OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw std::runtime_error(msg);
    }
    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (bag_size / local + ((bag_size % local) ? 1 : 0));
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to enqueue kernel\n", err);
        throw std::runtime_error(msg);
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Kernel failed to finish\n", err);
        throw std::runtime_error(msg);
    }
}

void inline lauchJoinScatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
                             ull *l, ull *r, ull const equals, ull const left_max, ull const right_max, ull const out_size) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)l_idx);
    cl_command_queue queue = create_queue(context);

    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context);
    cl_kernel kernel = create_kernel(program, "join_scatter");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l_idx);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &r_idx);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l_cnt);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &r_cnt);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &outpos);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &r);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(ull), &equals);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(ull), &left_max);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(ull), &right_max);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg, sizeof(ull), &out_size);
    if (err != CL_SUCCESS) {
        ARG_FAIL:
        sprintf(msg, "OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw std::runtime_error(msg);
    }

    auto num = left_max * right_max * equals;
    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (num / local + ((num % local) ? 1 : 0));
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to enqueue kernel\n", err);
        throw std::runtime_error(msg);
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Kernel failed to finish\n", err);
        throw std::runtime_error(msg);
    }

}

void inline launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const output_size,
        ull const rows, ull const loops) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    cl_program program = build_program(context);
    cl_kernel kernel = create_kernel(program, "string_gather");

    cl_int err = CL_SUCCESS;
    auto work_items = rows * loops;
    int arg = 0;
    // Set input parameters for the kernel
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(ull), &output_size);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(ull), &rows);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg, sizeof(ull), &loops);
    if (err != CL_SUCCESS) {
        ARG_FAIL:
        sprintf(msg, "OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw std::runtime_error(msg);
    }
    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (work_items / local + ((work_items % local) ? 1 : 0));
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to enqueue kernel\n", err);
        throw std::runtime_error(msg);
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Kernel failed to finish\n", err);
        throw std::runtime_error(msg);
    }
}

void inline launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
                             ull const *r_idx, ull const rows) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context);
    cl_kernel kernel = create_kernel(program, "str_cmp");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &left);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &right);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l_idx);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &r_idx);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg, sizeof(ull), &rows);
    if (err != CL_SUCCESS) {
        ARG_FAIL:
        sprintf(msg, "OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw std::runtime_error(msg);
    }

    auto num = rows;
    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (num / local + ((num % local) ? 1 : 0));
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to enqueue kernel\n", err);
        throw std::runtime_error(msg);
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Kernel failed to finish\n", err);
        throw std::runtime_error(msg);
    }
}

template<typename T>
void inline launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows, ull const loops) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context);

    char kernel_name[128];
    sprintf(kernel_name, "parser_%s", GetAFType<T>().str);
    cl_kernel kernel = create_kernel(program, kernel_name);

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &idx);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &input);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg, sizeof(ull), &rows);
    if (err != CL_SUCCESS) {
        ARG_FAIL:
        sprintf(msg, "OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw std::runtime_error(msg);
    }

    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (rows / local + ((rows % local) ? 1 : 0));
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to enqueue kernel\n", err);
        throw std::runtime_error(msg);
    }

    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Kernel failed to finish\n", err);
        throw std::runtime_error(msg);
    }
}

void inline launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
                             ull const rows, ull const loops) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);

    char options[128];
    sprintf(options, "-D LOOP_LENGTH=%llu", loops);
    // Build the OpenCL program and get the kernel
    cl_program program = build_single_use_program(context, options);
    cl_kernel kernel = create_kernel(program, "str_cmp_single");

    cl_int err = CL_SUCCESS;
    int arg = 0;
    // Set input parameters for the kernel
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &output);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &left);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &right);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg++, sizeof(cl_mem), &l_idx);
    if (err != CL_SUCCESS) goto ARG_FAIL;
    err = clSetKernelArg(kernel, arg, sizeof(ull), &rows);
    if (err != CL_SUCCESS) {
        ARG_FAIL:
        sprintf(msg, "OpenCL Error(%d): Failed to set kernel arguments\n", err);
        throw std::runtime_error(msg);
    }

    // Set launch configuration parameters and launch kernel
    size_t local = LOCAL_GROUP_SIZE;
    size_t global = local * (rows / local + ((rows % local) ? 1 : 0));
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to enqueue kernel\n", err);
        throw std::runtime_error(msg);
    }
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Kernel failed to finish\n", err);
        throw std::runtime_error(msg);
    }

}

#endif

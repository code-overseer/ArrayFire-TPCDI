#ifndef ARRAYFIRE_TPCDI_OPENCL_KERNELS_CPP
#define ARRAYFIRE_TPCDI_OPENCL_KERNELS_CPP

#include "Kernels.h"
#include "AFTypes.h"
#include "Utils.h"
#include <arrayfire.h>
#include <exception>
#include <string>
#include <cstdio>
#include <cstdlib>
#ifdef IS_APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static cl_context get_context(cl_mem x) {
    char msg[128];
    cl_context context;
    cl_int err = clGetMemObjectInfo(x, CL_MEM_CONTEXT, sizeof(cl_context), &context, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to get context from Memory object\n", err);
        throw std::runtime_error(msg);
    }
    return context;
}

static void printProgramBuildError(cl_context context, cl_program program) {
    size_t size;
    auto err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    auto devices = (cl_device_id*) malloc(size);
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
    err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
    char* log = (char*) malloc(size);
    err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, size, log, NULL);
    puts(log);

    free(log);
    free(devices);
}

static cl_program build_program(cl_context context) {
    char msg[128];
    static std::string kernel = Utils::loadFile(OCL_KERNEL_DIR);
    static cl_program program = nullptr;
    static cl_context previous = context;
    if (program) {
        if (context == previous) return program;
        else clReleaseProgram(program);
    }

    cl_int err;
    const char *source = kernel.c_str();
    size_t length = kernel.size();

    program = clCreateProgramWithSource(context, 1, &source, &length, &err);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to create program\n", err);
        throw std::runtime_error(msg);
    }

    err = clBuildProgram(program, 0, NULL, nullptr, NULL, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to build program\n", err);
        printProgramBuildError(context, program);
        throw std::runtime_error(msg);
    }

    return program;
}

static void printKernelBuildError(cl_context context, cl_program program) {
    size_t size;
    auto err = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    auto devices = (cl_device_id*) malloc(size);
    err = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, devices, NULL);
    err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, 0, NULL, &size);
    char* log = (char*) malloc(size);
    err = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, size, log, NULL);
    puts(log);

    free(log);
    free(devices);
}

static cl_kernel create_kernel(cl_program program, const char *kernel_name) {
    char msg[128];
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to create kernel %s\n", err, kernel_name);
        throw std::runtime_error(msg);
    }
    return kernel;
}

static cl_command_queue create_queue(cl_context context) {
    static cl_context prev = context;
    static cl_command_queue queue = nullptr;
    char msg[128];
    if (queue) {
        if (prev != context) prev = context;
        else return queue;
    }

    cl_device_id device;
    cl_int err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to get device from context\n", err);
        throw std::runtime_error(msg);
    }

    #ifdef IS_APPLE
    queue = clCreateCommandQueue(context, device, 0, &err);
    #else
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    #endif
    if (err != CL_SUCCESS) {
        sprintf(msg, "OpenCL Error(%d): Failed to command queue\n", err);
        throw std::runtime_error(msg);
    }
    return queue;
}


#ifdef REDUCE_THREADS
#define LOCAL_GROUP_SIZE 64
#else
#define LOCAL_GROUP_SIZE 256
#endif
typedef unsigned long long ull;

void launchCrossIntersect(char *result, ull const *bag, ull const *set, ull const bag_size, ull const set_size) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)result);
    cl_command_queue queue = create_queue(context);
    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context);
    cl_kernel kernel = create_kernel(program, "cross_intersect");
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

void launchHashIntersect(char *result, ull const *bag, ull const *ht_val, ull const *ht_ptr, ull const *ht_occ,
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

void lauchJoinScatter(ull const *l_idx, ull const *r_idx, ull const *l_cnt, ull const *r_cnt, ull const *outpos,
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

void launchStringGather(unsigned char *output, ull const *idx, unsigned char const *input, ull const output_size,
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

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, unsigned long long const *l_idx,
                      unsigned long long const *r_idx, unsigned long long const rows) {
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

void launchStringComp(bool *output, unsigned char const *left, unsigned char const *right, ull const *l_idx,
                      ull const rows, ull const loops) {
    char msg[128];
    // Get OpenCL context from memory buffer and create a Queue
    cl_context context = get_context((cl_mem)output);
    cl_command_queue queue = create_queue(context);
    // Build the OpenCL program and get the kernel
    cl_program program = build_program(context);
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

template<typename T>
void launchNumericParse(T *output, ull const * idx, unsigned char const *input, ull const rows) {
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

#define PARSER(TYPE) \
template void launchNumericParse<TYPE>(TYPE *output, ull const * idx, unsigned char const *input, ull const rows);

PARSER(unsigned char)
PARSER(float)
PARSER(double)
PARSER(unsigned short)
PARSER(short)
PARSER(unsigned int)
PARSER(int)
PARSER(ull)
PARSER(long long)

#undef PARSER

#endif
#ifndef ARRAYFIRE_TPCDI_OPENCL_HELPER_H
#define ARRAYFIRE_TPCDI_OPENCL_HELPER_H

#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include "TPCDI_Utils.h"
#ifdef IS_APPLE
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

static std::string& get_kernel_string() {
    static std::string kernel = TPCDI_Utils::loadFile(OCL_KERNEL_DIR"/opencl_kernels.cl");
    return kernel;
}

static cl_context get_context(cl_mem x) {
    cl_context context;
    cl_int err = clGetMemObjectInfo(x, CL_MEM_CONTEXT, sizeof(cl_context), &context, NULL);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to get context from Memory object\n", err);
        throw (err);
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

static cl_program build_program(cl_context context, const std::string& kernel_string, char const *options = nullptr) {
    static cl_program program = nullptr;
    if (program) return program;

    cl_int err;
    const char *source = kernel_string.c_str();
    size_t length = kernel_string.size();
    program = clCreateProgramWithSource(context, 1, &source, &length, &err);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to create program\n", err);
        throw std::runtime_error("Terminated");
    }

    err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to build program\n", err);
        printProgramBuildError(context, program);
        throw std::runtime_error("Terminated");
    }
    return program;
}

static cl_program build_parse_program(cl_context context, char const *options = nullptr) {
    cl_program program;
    cl_int err;
    static std::string kernel_str = TPCDI_Utils::loadFile(OCL_KERNEL_DIR"/opencl_parsers.cl");
    static const char *source = kernel_str.c_str();
    static size_t length = kernel_str.size();

    program = clCreateProgramWithSource(context, 1, &source, &length, &err);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to create program\n", err);
        throw std::runtime_error("Terminated");
    }

    err = clBuildProgram(program, 0, NULL, options, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to build program\n", err);
        printProgramBuildError(context, program);
        throw std::runtime_error("Terminated");
    }
    return program;
}

static cl_kernel create_kernel(cl_program program, const char *kernel_name) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name, &err);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to create kernel %s\n", err, kernel_name);
        throw std::runtime_error("Terminated");
    }
    return kernel;
}

static cl_command_queue create_queue(cl_context context) {
    static cl_context prev = nullptr;
    static cl_command_queue queue = nullptr;
    if (prev != context) prev = context;
    else if (queue) return queue;

    cl_device_id device;
    cl_int err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(cl_device_id), &device, NULL);
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to get device from context\n", err);
        throw std::runtime_error("Terminated");
    }

    #ifdef IS_APPLE
    queue = clCreateCommandQueue(context, device, 0, &err);
    #else
    queue = clCreateCommandQueueWithProperties(context, device, 0, &err);
    #endif
    if (err != CL_SUCCESS) {
        printf("OpenCL Error(%d): Failed to command queue\n", err);
        throw std::runtime_error("Terminated");
    }
    return queue;
}

#endif
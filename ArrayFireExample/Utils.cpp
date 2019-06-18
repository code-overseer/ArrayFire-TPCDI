//
//  Utils.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "Utils.hpp"

using namespace af;
char const* HOME = getenv("HOME");
char const* HR = "/Downloads/HRs/HR";
char const* CPU = "/Downloads/Results/CPU_trim.csv";
char const* OCL = "/Downloads/Results/OCL_trim.csv";

std::string textToString(char const *filename) {
    std::ifstream file(filename);
    std::string data;
    file.seekg(0, std::ios::end);
    data.reserve(file.tellg());
    file.seekg(0, std::ios::beg);
    data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
    return data;
}

void experiment() {
    std::stringstream opencl;
    std::stringstream cpu;
    auto path = std::string(HOME);
    int s[] = {3,5,10,20,40,80,160,320};
    char sf [4];
    char out[32];
    for (int i = 0; i <= 5; i++) {
        setBackend(AF_BACKEND_CPU);
        setDevice(0);
        for (int j = 0; j < 8; j++)
        {
            auto scale = s[j];
            sprintf(sf, "%d", scale);
            {
                auto cpu_o = AFCSVParser::parse((path + HR + sf + ".csv").c_str(), false);
                sync();
                timer::start();
                cpu_o.select(5, "314");
                sync();
                sprintf(out, "%d, %g\n", scale, timer::stop());

                if (i) cpu << out;
            }
        }
        setBackend(AF_BACKEND_OPENCL);
        setDevice(0);
        for (int j = 0; j < 8; j++) {
            auto scale = s[j];
            sprintf(sf, "%d", scale);
            {
                auto opencl_o = AFCSVParser::parse((path + HR + sf + ".csv").c_str(), false);
                sync();
                timer::start();
                opencl_o.select(5, "314");
                sync();
                sprintf(out, "%d, %g\n", scale, timer::stop());

                if (i) opencl << out;
            }
        }

    }
    std::ofstream fileout(path + CPU);
    fileout << cpu.str();
    fileout.close();
    fileout = std::ofstream(path + OCL);
    fileout << opencl.str();
    fileout.close();
    std::cout << cpu.str()<<std::endl;
    std::cout << opencl.str()<<std::endl;
}

void single_run(Backend const backend) {
    auto path = std::string(HOME) + HR;
    setBackend(backend);
    setDevice(0);
    char sf [4];
    auto scale = 5;
    sprintf(sf, "%d", scale);
    path = (path + sf + ".csv");
    auto o = AFCSVParser::parse(path.c_str(), false);
    sync();
    timer::start();
    o.select(5, "314");
    sync();
    printf("scale: %d, elapsed time: %g\n", scale, timer::stop());
    o.printColumn(std::cout, 5);
}

//
//  Utils.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "Utils.hpp"
#include <cstring>

using namespace af;
char const* HOME = getenv("HOME");
char const* HR = "/Downloads/TPCData/HR";
char const* DATE = "/Downloads/TPCData/DateFormat";

int isExists(unsigned int const scale, int scales[], int length) {
    for (int i = 0; i < length; i++) {
        if (scales[i] == scale) return i;
    }
    return -1;
}

std::string ParseAndTrim(Backend backend, int const device, unsigned int const runs, unsigned int const scale) {
    std::stringstream ss;
    auto path = std::string(HOME);
    int s[] = {3,5,10,20,40,80,160,320};
    int n = isExists(scale, s, 8);
    if (n < 0) throw std::invalid_argument("scale does not exist");

    char sf [4];
    char out[32];
    setBackend(backend);
    setDevice(device);
    ss << infoString();

    bool b = scale == 0;
    for (int i = b ? 0 : scale; (b ? i < 8 : i == scale); i++) {
        for (int j = 0; j < runs; j++)
        {
            auto scale = s[j];
            sprintf(sf, "%d", scale);
            {
                sync();
                timer::start();
                auto parser = AFParser::parse((path + HR + sf + ".csv").c_str(), ',');
                parser.select(5, "314");
                sync();
                sprintf(out, "%d, %g\n", scale, timer::stop());

                if (i) ss << out;
            }
        }

    }
    return ss.str();
}

void load_DimDate(AFParser &dimDate) {
    char path[128] = {'\0'};
    setBackend(AF_BACKEND_CPU);
    strcat(path, HOME);
    strcat(path, DATE);
    strcat(path, ".txt");
    std::cout<<path<<std::endl;
    dimDate = AFParser::parse(path, '|');
    array tmp;
    dimDate.asDate(1, tmp, DateFormat::YYYYMMDD, true);


}

void load_DimBroker() {
    char path[128] = {'\0'};
    strcat(path, HOME);
    strcat(path, HR);
    strcat(path, "3.csv");

    auto parser = AFParser::parse(path, ',');

}
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
char const* DATE = "/Downloads/TPCData/Date";

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
                AFParser parser((path + HR + sf + ".csv").c_str(), ',');
                parser.stringMatch(5, "314");
                sync();
                sprintf(out, "%d, %g\n", scale, timer::stop());

                if (i) ss << out;
            }
        }

    }
    return ss.str();
}

AFParser load_DimDate() {
    char path[128] = {'\0'};
    strcat(path, HOME);
    strcat(path, DATE);
    strcat(path, ".txt");
    setBackend(AF_BACKEND_CPU);
    auto dimDate = AFParser(path, '|');
    dimDate.asDate(1,YYYYMMDD,true);
    return dimDate;
}

AFParser load_DimBroker() {
    char path[128] = {'\0'};
    setBackend(AF_BACKEND_CPU);
    strcat(path, HOME);
    strcat(path, HR);
    strcat(path, "3.csv");
    auto dimBroker = AFParser(path, ',');

    {
        auto rows = dimBroker.stringMatch(5, "314");
        dimBroker.keepRows(rows);
    }
    dimBroker.removeColumn(5);


    {
        auto SK_BrokerID = range(dim4(dimBroker.length(),1),0,u32);
        SK_BrokerID = AFParser::serializeUnsignedInt(SK_BrokerID);
        dimBroker.insertAsFirst(SK_BrokerID);
    }
    {
        array IsCurrent(1,5, "True\n");
        IsCurrent = tile(IsCurrent, dimBroker.length()).as(u8);
        dimBroker.insertAsLast(IsCurrent);
    }
    {
        auto BatchID = constant(1, dimBroker.length(), u8);
        BatchID = AFParser::serializeUnsignedInt(BatchID);
        dimBroker.insertAsLast(BatchID);
    }
    {
        array Date;
        {
            auto dates = load_DimDate();
            Date = dates.asDate(1, YYYYMMDD, true);
            Date = sort(Date, 0);
        }
        Date = AFParser::serializeDate(Date(0));
        Date = tile(Date, dimBroker.length());
        dimBroker.insertAsLast(Date);
        Date = AFParser::serializeDate(AFParser::endDate());
        Date = tile(Date, dimBroker.length());
        dimBroker.insertAsLast(Date);
    }
    sync();
    return dimBroker;
}

AFParser test_SignedInt() {
    char path[128] = {'\0'};
    strcat(path, HOME);
    strcat(path, "/Downloads/TPCData/TestInt.csv");
    auto testInt = AFParser(path, ',');
    testInt.asSigned32(0);
    sync();
    return testInt;
}

AFParser test_Float() {
    char path[128] = {'\0'};
    strcat(path, HOME);
    strcat(path, "/Downloads/TPCData/TestFloat.csv");
    auto testInt = AFParser(path, ',');
    testInt.asFloat(0);
    sync();
    return testInt;
}
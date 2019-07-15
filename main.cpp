#include <cstdio>
#include <cstdlib>
#include <rapidxml.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "BatchFunctions.h"
#include "Tests.h"
#include "TPC_DI.h"
#include "XMLFlattener.h"

using namespace af;
using namespace rapidxml;

namespace DIR {
    char const* HR = "/Users/bryanwong/Downloads/TPCData/HR3.csv";
    char const* DATE = "/Users/bryanwong/Downloads/TPCData/TestDate.csv";
    char const* UINT = "/Users/bryanwong/Downloads/TPCData/TestUint.csv";
    char const* UCHAR = "/Users/bryanwong/Downloads/TPCData/TestUchar.csv";
    char const* INT = "/Users/bryanwong/Downloads/TPCData/TestInt.csv";
    char const* FLOAT = "/Users/bryanwong/Downloads/TPCData/TestFloat.csv";
    char const* DIRECTORY = "/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/";
}


int main(int argc, char *argv[])
{
    setBackend(AF_BACKEND_CPU);
    auto f = loadStagingFinwire(DIR::DIRECTORY);
    auto i = loadIndustry(DIR::DIRECTORY);
    auto s = loadStatusType(DIR::DIRECTORY);
    auto date = loadDimDate(DIR::DIRECTORY);
    auto d = loadDimCompany(*f.company, i, s, date);
    auto sec = loadDimSecurity(*f.security, d, s);

    return 0;
}

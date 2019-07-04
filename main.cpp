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

char const* HOME = getenv("HOME");
char const* HR = "/Downloads/TPCData/HR3.csv";
char const* DATE = "/Downloads/TPCData/TestDate.csv";
char const* UINT = "/Downloads/TPCData/TestUint.csv";
char const* UCHAR = "/Downloads/TPCData/TestUchar.csv";
char const* INT = "/Downloads/TPCData/TestInt.csv";
char const* FLOAT = "/Downloads/TPCData/TestFloat.csv";
char const* DIRECTORY = "/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/";

int main(int argc, char *argv[])
{
    setBackend(AF_BACKEND_CPU);
//    auto f = loadStagingWatches(DIRECTORY);
//    af_print(f.data()[2].rows(0,20))
    int a1[] = {1 , 3, 4, 5, 2, 3, 1, 3, 4};
    int k[] = {1,2,3,4,5};
    auto a = array(9, a1);
    auto b = array(5, k);
    af_print(a)
    af_print(b)
    auto c = batchFunc(b, reorder(a,1,0),BatchFunctions::batchEqual);
    af_print(c)
    auto d = where(c) % b.dims(0);
    af_print(b(d))

    return 0;
}

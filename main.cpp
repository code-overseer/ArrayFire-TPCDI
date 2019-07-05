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

char const* HR = "/Users/bryanwong/Downloads/TPCData/HR3.csv";
char const* DATE = "/Users/bryanwong/Downloads/TPCData/TestDate.csv";
char const* UINT = "/Users/bryanwong/Downloads/TPCData/TestUint.csv";
char const* UCHAR = "/Users/bryanwong/Downloads/TPCData/TestUchar.csv";
char const* INT = "/Users/bryanwong/Downloads/TPCData/TestInt.csv";
char const* FLOAT = "/Users/bryanwong/Downloads/TPCData/TestFloat.csv";
char const* DIRECTORY = "/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/";

int main(int argc, char *argv[])
{
    setBackend(AF_BACKEND_CPU);
//    auto f = loadStagingWatches(DIRECTORY);
//    af_print(f.data()[2].rows(0,20))
//    int a1[] = {1 ,3, 4, 5, 2, 3, 1, 3, 4, 2,3,4,1,4,4,5,1,2,3,3,2,2,3,4};
//    int k[] = {1,2,3,4,5,1,2,3,4,5};
//    auto a = array(12,2, a1);
//    auto b = array(5,2, k);
//    auto ha = batchFunc(a, pow(7, range(dim4(1, a.dims(1)), 1, u32)), BatchFunctions::batchMul);
//    auto hb = batchFunc(b, pow(7, range(dim4(1, b.dims(1)), 1, u32)), BatchFunctions::batchMul);
//    hb = sum(hb,1) % 50000059;
//    ha = sum(ha,1) % 50000059;
//    af_print(ha)
//    af_print(hb)
//    auto idx = batchFunc(hb, moddims(ha,dim4(ha.dims(1),ha.dims(0))), BatchFunctions::batchEqual);
//    af_print(idx)
//    auto d = where(idx) % hb.dims(0);
//    af_print(d)
//    af_print(b(d, span))
//    af_print(a)
    test_String(FLOAT);

    return 0;
}

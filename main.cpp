#include <cstdio>
#include <cstdlib>
#include <rapidxml.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <OpenCL/opencl.h>
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
    char const* WORDS = "/Users/bryanwong/Documents/MPSI/words_alpha.txt";
    char const* NUMBERS = "/Users/bryanwong/Documents/MPSI/numbers.txt";
    char const* UUID = "/Users/bryanwong/Documents/MPSI/uuid.txt";
    char const* DIRECTORY = "/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/";
}

void experiment(int argc, char *argv[]);

int main(int argc, char *argv[])
{
    auto flipdims = AFDataFrame::flipdims;
    setBackend(AF_BACKEND_CPU);
    array lhs;
    array rhs;
    {
        int l[] = {2,3,3,5,5,6,6,6,6,8,8,9,9,9};
        int r[] = {2,3,4,4,5,5,5,6,7,7,7,9,9,11,12};
        lhs = array(14, l).as(u32);
        rhs = array(15, r).as(u32);
        lhs = flipdims(lhs);
        lhs = join(0, lhs, range(lhs.dims(), 1, u32));
        rhs = flipdims(rhs);
        rhs = join(0, rhs, range(rhs.dims(), 1, u32));
    }
    auto setrl = flipdims(setIntersect(setUnique(lhs.row(0)), setUnique(rhs.row(0)), true));


    unsigned int m = 6; // need to choose bucket size;
    unsigned int n;
    {
        auto set0 = setrl % m;
        array idx;
        sort(set0, idx, set0, 1);
        auto bin = join(1, constant(1,dim4(1),set0.type()), diff1(set0, 1));
        bin = bin > 0;
        bin = accum(bin, 1) - 1;
        auto set1 = flipdims(setUnique(set0, true));
        auto hist = flipdims(histogram(bin, set1.elements()));
        n = sum<unsigned int>(max(hist).as(u32)); // max collisions
        auto hash_table = constant(UINT32_MAX, dim4(n, m), u32);
        auto starts = batchFunc(set1 * n, range(dim4(n), 0, u32), BatchFunctions::batchAdd);
        starts = flipdims(starts(batchFunc(hist, range(dim4(n), 0, u32), BatchFunctions::batchSub) > 0));
        hash_table(starts) = flipdims(setrl(idx));
        auto h = batchFunc(lhs.row(0) % m * n, range(dim4(n), 0, u32), BatchFunctions::batchAdd);
        h = hash_table(h);
        h = moddims(h,dim4(n, h.elements() / n));
        h = batchFunc(h, lhs.row(0), BatchFunctions::batchEqual);
        h = flipdims(where(anyTrue(h, 0)));
        lhs = lhs(span,h);

        h = batchFunc(rhs.row(0) % m * n, range(dim4(n), 0, u32), BatchFunctions::batchAdd);
        h = hash_table(h);
        h = moddims(h,dim4(n, h.elements() / n));
        h = batchFunc(h, rhs.row(0), BatchFunctions::batchEqual);
        h = flipdims(where(anyTrue(h, 0)));
        rhs = rhs(span,h);
    }
    auto cl = flipdims(histogram(lhs.row(0), lhs.row(0).elements())).as(u32);
    cl = cl(cl > 0);
    auto il = scan(cl, 1, AF_BINARY_ADD, false);
    auto cr = flipdims(histogram(rhs.row(0), rhs.row(0).elements())).as(u32);
    cr = cr(cr > 0);
    auto ir = scan(cr, 1, AF_BINARY_ADD, false);

    auto outpos = cr * cl;
    auto out_size = sum<unsigned int>(sum(outpos,1));
    outpos = scan(outpos, 1, AF_BINARY_ADD, false);
    af::sync();

    auto x = setrl.elements();
    auto y = sum<unsigned int>(max(cl,1));
    auto z = sum<unsigned int>(max(cr,1));
    auto i = range(dim4(1, x * y * z), 1, u32);
    auto j = i / z % y;
    auto k = i % z;
    i = i / y / z;
    array l(1, out_size + 1, u32);
    array r(1, out_size + 1, u32);
    auto b = !(j / cl(i)) && !(k / cr(i));
    l(b * (outpos(i) + cl(i) * k + j) + !b * out_size) = il(i) + j;
    r(b * (outpos(i) + cr(i) * j + k) + !b * out_size) = ir(i) + k;
    l = l.cols(0,end - 1);
    r = r.cols(0,end - 1);
    l = lhs(1,l);
    r = rhs(1,r);
    l.eval();
    r.eval();
    af::sync();

    return 0;
}

void experiment(int argc, char *argv[]) {
    auto path = argc > 1 ? argv[1] : DIR::DIRECTORY;
    timer::start();
    auto batchDate = loadBatchDate(path);
    auto dimDate = loadDimDate(path);
    auto dimTime = loadDimTime(path);

    dimTime.flushToHost();
    print("industry");
    auto industry = loadIndustry(path);
    print("statusType");
    auto statusType = loadStatusType(path);
    print("taxRate");
    auto taxRate = loadTaxRate(path);
    taxRate.flushToHost();
    print("tradeType");
    auto tradeType = loadTradeType(path);
    tradeType.flushToHost();
    print("audit");
    auto audit = loadAudit(path);
    audit.flushToHost();
    print("finwire");
    auto finwire = loadStagingFinwire(path);
    print("s_prospect");
    auto s_prospect = loadStagingProspect(path);
    print("s_cash");
    auto s_cash = loadStagingCashBalances(path);
    print("s_watches");
    auto s_watches = loadStagingWatches(path);
    print("s_customer");
    auto s_customer = loadStagingCustomer(path);
    print("dimCompany");
    auto dimCompany = loadDimCompany(*finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    print("financial");
    auto financial = loadFinancial(*finwire.financial, dimCompany);
    financial.flushToHost();
    print("dimSecurity");
    auto dimSecurity = loadDimSecurity(*finwire.security, dimCompany, statusType);

    dimSecurity.flushToHost();
    dimCompany.flushToHost();
    statusType.flushToHost();
    finwire.security->clear();
    finwire.company->clear();
    finwire.financial->clear();
    print("prospect");
    auto prospect = loadProspect(s_prospect, batchDate);
    prospect.flushToHost();
    batchDate.flushToHost();
    s_prospect.clear();
    print("dimBroker");
    auto dimBroker = loadDimBroker(path, dimDate);
    dimBroker.flushToHost();
    dimDate.flushToHost();
    af::sync();
    char t[64];
    sprintf(t, "%f", timer::stop());

    if (argc < 3) return;
    std::ofstream outfile;
    outfile.open("results.csv", std::ios_base::app);
    outfile << argv[2] << ',' << t << '\n';
    outfile.close();
}
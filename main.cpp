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
    auto path = argc > 1 ? argv[1] : DIR::DIRECTORY;
    timer::start();
    auto batchDate = loadBatchDate(path);
    auto dimDate = loadDimDate(path);
    auto dimTime = loadDimTime(path);
    dimTime.flushToHost();
    auto industry = loadIndustry(path);
    auto statusType = loadStatusType(path);
    auto taxRate = loadTaxRate(path);
    taxRate.flushToHost();
    auto tradeType = loadTradeType(path);
    tradeType.flushToHost();
    auto audit = loadAudit(path);
    audit.flushToHost();

    auto finwire = loadStagingFinwire(path);
    auto s_prospect = loadStagingProspect(path);
    auto s_customer = loadStagingCustomer(path);
    auto s_cash = loadStagingCashBalances(path);
    auto s_watches = loadStagingWatches(path);

    auto dimCompany = loadDimCompany(*finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    auto financial = loadFinancial(*finwire.financial, dimCompany);
    financial.flushToHost();
    auto dimSecurity = loadDimSecurity(*finwire.security, dimCompany, statusType);
    dimSecurity.flushToHost();
    dimCompany.flushToHost();
    statusType.flushToHost();
    finwire.security.reset();
    finwire.company.reset();
    finwire.financial.reset();
    auto prospect = loadProspect(s_prospect, batchDate);
    prospect.flushToHost();
    batchDate.flushToHost();
    s_prospect.clear();
    auto dimBroker = loadDimBroker(path, dimDate);
    dimBroker.flushToHost();
    dimDate.flushToHost();
    printf("%f", timer::stop());
    return 0;
}

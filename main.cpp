#include <cstdio>
#include <fstream>
#include <string>
#include "BatchFunctions.h"
#include "Logger.h"
#include "TPC_DI.h"
#if USING_OPENCL
#include "OpenCL/opencl_kernels.h"
#elif USING_CUDA
#include "CUDA/cuda_kernels.h"
#endif

using namespace af;

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
    #if USING_OPENCL
        setBackend(AF_BACKEND_OPENCL);
    #elif USING_CUDA
        setBackend(AF_BACKEND_CUDA);
    #else
        setBackend(AF_BACKEND_CPU);
    #endif
    auto path = argc > 1 ? argv[1] : DIR::DIRECTORY;
    Logger::startTimer();
    print("DimDate");
    auto dimDate = loadDimDate(path);
    print("StatusType");
    auto statusType = loadStatusType(path);
    print("Finwire");
    auto finwire = loadStagingFinwire(path);
    print("industry");
    auto industry = loadIndustry(path);
    print("dimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    dimCompany.sortBy("CompanyID");
    Logger::logTime();
    Logger::sendToCSV();

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
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    print("financial");
    auto financial = loadFinancial(finwire.financial, dimCompany);
    financial.flushToHost();
    print("dimSecurity");
    auto dimSecurity = loadDimSecurity(finwire.security, dimCompany, statusType);

    dimSecurity.flushToHost();
    dimCompany.flushToHost();
    statusType.flushToHost();
    finwire.clear();
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
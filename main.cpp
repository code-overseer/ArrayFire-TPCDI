#include <cstring>
#include <string>
#include "BatchFunctions.h"
#include "Logger.h"
#include "TPC_DI.h"

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
    #if defined(IS_APPLE)
    char const* DIRECTORY = "/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/";
    #else
    char const* DIRECTORY = "/home/jw5514/data/5/Batch1/";
    #endif
}

void experiment();

int main(int argc, char *argv[])
{
    #if defined(USING_OPENCL)
        setBackend(AF_BACKEND_OPENCL);
    #elif defined(USING_CUDA)
        setBackend(AF_BACKEND_CUDA);
    #else
        setBackend(AF_BACKEND_CPU);
    #endif
    if (argc == 2 && !strcmp(argv[1], "-i")) {
        info();
        return 0;
    }

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i],"-f")) {
            DIR::DIRECTORY = argv[++i];
        } else if (!strcmp(argv[i],"-d")) {
            setDevice(atoi(argv[++i]));
        } else if (!strcmp(argv[i],"-o")) {
            Logger::output() = argv[++i];
        }
    }

    Logger::startTimer();
//    experiment();
    print("Staging Cash Balnces");
    auto s_cash = loadStagingCashBalances(DIR::DIRECTORY);
    Logger::logTime();
//    Logger::sendToCSV();

    return 0;
}

void experiment() {
    auto batchDate = loadBatchDate(DIR::DIRECTORY);
    print("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY);
    
    print("DimTime");
    auto dimTime = loadDimTime(DIR::DIRECTORY);
    dimTime.flushToHost();
    
    print("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY);
    
    print("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY);
    
    print("TaxRate");
    auto taxRate = loadTaxRate(DIR::DIRECTORY);
    taxRate.flushToHost();
    
    print("TradeType");
    auto tradeType = loadTradeType(DIR::DIRECTORY);
    tradeType.flushToHost();
    
    print("Audit");
    auto audit = loadAudit(DIR::DIRECTORY);
    audit.flushToHost();

    print("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY);
    
    print("Staging Prospect");
    auto s_prospect = loadStagingProspect(DIR::DIRECTORY);
    
    print("Staging Cash Balnces");
    auto s_cash = loadStagingCashBalances(DIR::DIRECTORY);
    
    print("Staging Watches");
    auto s_watches = loadStagingWatches(DIR::DIRECTORY);
    
    print("Staging Customer");
    auto s_customer = loadStagingCustomer(DIR::DIRECTORY);
    
    print("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    
    print("Financial");
    auto financial = loadFinancial(finwire.financial, dimCompany);
    financial.flushToHost();
    
    print("DimSecurity");
    auto dimSecurity = loadDimSecurity(finwire.security, dimCompany, statusType);

    dimSecurity.flushToHost();
    dimCompany.flushToHost();
    statusType.flushToHost();
    finwire.clear();
    
    print("Prospect");
    auto prospect = loadProspect(s_prospect, batchDate);
    prospect.flushToHost();
    batchDate.flushToHost();
    s_prospect.clear();

    print("DimBroker");
    auto dimBroker = loadDimBroker(DIR::DIRECTORY, dimDate);
    dimBroker.flushToHost();
    dimDate.flushToHost();
}
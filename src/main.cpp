#include <cstring>
#include <string>
#include "include/Logger.h"
#include "include/TPCDI.h"
#include "include/Tests.h"

namespace DIR {
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

void DimCompany();

void DimSecurity();

void Financial();

void DimBroker();

void FinWire();

int main(int argc, char *argv[]) {
    using namespace af;
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
            setDevice(std::stoi(argv[++i]));
        } else if (!strcmp(argv[i],"-o")) {
            Logger::output() = argv[++i];
        } else if (!strcmp(argv[i],"-I")) {
            info();
        }
    }

    Logger::startTimer();
    DimCompany();
//    DimBroker();
//    Financial();
    Logger::logTime();
    return 0;
}

void experiment() {
    auto batchDate = loadBatchDate(DIR::DIRECTORY);
    print("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY);
    
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
    s_cash.flushToHost();

    print("Staging Watches");
    auto s_watches = loadStagingWatches(DIR::DIRECTORY);
    s_watches.flushToHost();

    print("Staging Customer");
    auto s_customer = loadStagingCustomer(DIR::DIRECTORY);
    s_customer.flushToHost();

    print("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    
    print("Financial");
    auto financial = loadFinancial(std::move(finwire.financial), dimCompany);
    financial.flushToHost();
    
    print("DimSecurity");
    auto dimSecurity = loadDimSecurity(std::move(finwire.security), dimCompany, statusType);
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

void DimCompany() {
    print("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY);

    print("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY);

    print("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY);

    print("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY);

    print("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    statusType.flushToHost();
    dimDate.flushToHost();
    dimCompany.flushToHost();
}

void DimSecurity() {
    print("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY);

    print("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY);

    print("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY);

    print("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY);

    print("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    dimDate.flushToHost();
    finwire.financial.clear();
    finwire.company.clear();

    print("DimSecurity");
    auto dimSecurity = loadDimSecurity(std::move(finwire.security), dimCompany, statusType);
}

void Financial() {
    print("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY);

    print("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY);

    print("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY);

    print("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY);

    print("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    statusType.flushToHost();
    dimDate.flushToHost();
    finwire.security.clear();
    finwire.company.clear();

    print("Financial");
    auto financial = loadFinancial(std::move(finwire.financial), dimCompany);
}

void DimBroker() {
    print("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY);
    print("DimBroker");
    loadDimBroker(DIR::DIRECTORY, dimDate);
}

void FinWire() {
    auto batchDate = loadBatchDate(DIR::DIRECTORY);
    print("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY);

    print("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY);

    print("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY);

    print("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY);

    print("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    finwire.company.clear();

    print("DimSecurity");
    auto dimSecurity = loadDimSecurity(std::move(finwire.security), dimCompany, statusType);
    dimSecurity.flushToHost();
    statusType.flushToHost();
    finwire.security.clear();

    print("Financial");
    auto financial = loadFinancial(std::move(finwire.financial), dimCompany);
    financial.flushToHost();
    finwire.clear();

}



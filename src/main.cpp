#include <cstring>
#include <string>
#include <cstdio>
#include "Logger.h"
#include "TPCDI.h"
#include "Tests.h"

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
    std::string DIRECTORY = "/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/";
    #else
    std::string DIRECTORY = "/home/jw5514/data/5/Batch1/";
    #endif
}

void fullBenchmark();

void DimCompany();

void DimSecurity();

void Financial();

void DimBroker();

void FinWire();

void DailyMarket();

int main(int argc, char *argv[]) {
    using namespace af;
    #if defined(USING_OPENCL)
        setBackend(AF_BACKEND_OPENCL);
    #elif defined(USING_CUDA)
        setBackend(AF_BACKEND_CUDA);
    #else
        setBackend(AF_BACKEND_CPU);
    #endif
    int scale = 3;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i],"-f")) {           
            DIR::DIRECTORY = (std::string("/home/jw5514/data/") + argv[++i] + std::string("/Batch1/"));
            scale = (int)strtol(argv[i], nullptr, 10);
        } else if (!strcmp(argv[i],"-d")) {
            setDevice(std::stoi(argv[++i]));
        } else if (!strcmp(argv[i],"-o")) {
            Logger::directory(std::string(argv[++i]));
        } else if (!strcmp(argv[i],"-I")) {
            info();
        } else if (!strcmp(argv[i], "-i")) {
            info();
            return 0;
        }
    }

    
    af::deviceGC();
    Logger::startTimer();
    fullBenchmark();
    Logger::logTime();
    Logger::sendToCSV(scale);

    
    return 0;
}

void DailyMarket() {
    puts("Staging Market");
    auto d = loadStagingMarket(DIR::DIRECTORY.c_str());
    d.flushToHost();
}

void fullBenchmark() {
    auto batchDate = loadBatchDate(DIR::DIRECTORY.c_str());
    puts("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY.c_str());
    
    puts("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY.c_str());
    
    puts("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY.c_str());
    
    puts("TaxRate");
    auto taxRate = loadTaxRate(DIR::DIRECTORY.c_str());
    taxRate.flushToHost();

    puts("TradeType");
    auto tradeType = loadTradeType(DIR::DIRECTORY.c_str());
    tradeType.flushToHost();

    puts("Audit");
    auto audit = loadAudit(DIR::DIRECTORY.c_str());
    audit.flushToHost();
    
    puts("Staging Prospect");
    auto s_prospect = loadStagingProspect(DIR::DIRECTORY.c_str());
    puts("Prospect");
    auto prospect = loadProspect(s_prospect, batchDate);
    prospect.flushToHost();
    batchDate.flushToHost();
    s_prospect.clear();
    af::deviceGC();
    puts("Staging Cash Balnces");
    auto s_cash = loadStagingCashBalances(DIR::DIRECTORY.c_str());
    s_cash.flushToHost();

    puts("Staging Watches");
    auto s_watches = loadStagingWatches(DIR::DIRECTORY.c_str());
    s_watches.flushToHost();

    af::deviceGC();
    DailyMarket();
    
    puts("Staging Customer");
    auto s_customer = loadStagingCustomer(DIR::DIRECTORY.c_str());
    s_customer.flushToHost();

    puts("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY.c_str());

    puts("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    
    puts("Financial");
    auto financial = loadFinancial(std::move(finwire.financial), dimCompany);
    financial.flushToHost();
    
    puts("DimSecurity");
    auto dimSecurity = loadDimSecurity(std::move(finwire.security), dimCompany, statusType);
    dimSecurity.flushToHost();
    dimCompany.flushToHost();
    statusType.flushToHost();
    finwire.clear();
   

    puts("DimBroker");
    auto dimBroker = loadDimBroker(DIR::DIRECTORY.c_str(), dimDate);
    dimBroker.flushToHost();
    dimDate.flushToHost();

    

}

void DimCompany() {
    puts("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY.c_str());

    puts("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY.c_str());

    puts("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY.c_str());

    puts("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY.c_str());

    puts("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    statusType.flushToHost();
    dimDate.flushToHost();
    dimCompany.flushToHost();
}

void DimSecurity() {
    puts("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY.c_str());

    puts("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY.c_str());

    puts("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY.c_str());

    puts("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY.c_str());

    puts("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    dimDate.flushToHost();
    finwire.financial.clear();
    finwire.company.clear();

    puts("DimSecurity");
    auto dimSecurity = loadDimSecurity(std::move(finwire.security), dimCompany, statusType);
}

void Financial() {
    puts("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY.c_str());

    puts("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY.c_str());

    puts("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY.c_str());

    puts("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY.c_str());

    puts("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    statusType.flushToHost();
    dimDate.flushToHost();
    finwire.security.clear();
    finwire.company.clear();

    puts("Financial");
    auto financial = loadFinancial(std::move(finwire.financial), dimCompany);
}

void DimBroker() {
    puts("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY.c_str());
    puts("DimBroker");
    loadDimBroker(DIR::DIRECTORY.c_str(), dimDate);
}

void FinWire() {
    auto batchDate = loadBatchDate(DIR::DIRECTORY.c_str());
    puts("DimDate");
    auto dimDate = loadDimDate(DIR::DIRECTORY.c_str());

    puts("Industry");
    auto industry = loadIndustry(DIR::DIRECTORY.c_str());

    puts("StatusType");
    auto statusType = loadStatusType(DIR::DIRECTORY.c_str());

    puts("Finwire");
    auto finwire = loadStagingFinwire(DIR::DIRECTORY.c_str());

    puts("DimCompany");
    auto dimCompany = loadDimCompany(finwire.company, industry, statusType, dimDate);
    industry.flushToHost();
    finwire.company.clear();

    puts("DimSecurity");
    auto dimSecurity = loadDimSecurity(std::move(finwire.security), dimCompany, statusType);
    dimSecurity.flushToHost();
    statusType.flushToHost();
    finwire.security.clear();

    puts("Financial");
    auto financial = loadFinancial(std::move(finwire.financial), dimCompany);
    financial.flushToHost();
    finwire.clear();

}



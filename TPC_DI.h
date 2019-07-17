

//
// Created by Bryan Wong on 2019-06-28.
//

#ifndef ARRAYFIRE_TPCDI_TPC_DI_H
#define ARRAYFIRE_TPCDI_TPC_DI_H
#include "AFDataFrame.h"
#include "AFParser.hpp"
#include <utility>
#include <memory>

typedef std::shared_ptr<AFDataFrame> AFDF_ptr;
struct Finwire {
public:
    AFDF_ptr company;
    AFDF_ptr financial;
    AFDF_ptr security;
    Finwire(AFDF_ptr cmp, AFDF_ptr fin, AFDF_ptr sec) : company(std::move(cmp)),
    financial(std::move(fin)), security(std::move(sec)) {}
    Finwire(Finwire &&other) noexcept;
};

struct Customer {
public:
    AFDF_ptr newCust;
    AFDF_ptr addAcct;
    AFDF_ptr updAcct;
    AFDF_ptr closeAcct;
    AFDF_ptr updCust;
    AFDF_ptr inact;
    Customer(AFDF_ptr newC, AFDF_ptr addA, AFDF_ptr updA, AFDF_ptr closeA, AFDF_ptr updC, AFDF_ptr in) :
    newCust(std::move(newC)), addAcct(std::move(addA)), updAcct(std::move(updA)), closeAcct(std::move(closeA)),
    updCust(std::move(updC)), inact(std::move(in)) {}
    Customer(Customer &&other) noexcept;
};

AFDataFrame loadBatchDate(char const* directory);

AFDataFrame loadDimDate(char const* directory);

AFDataFrame loadDimTime(char const* directory);

AFDataFrame loadIndustry(char const* directory);

AFDataFrame loadStatusType(char const* directory);

AFDataFrame loadTaxRate(char const* directory);

AFDataFrame loadTradeType(char const* directory);

AFDataFrame loadAudit(char const* directory);

Finwire loadStagingFinwire(char const *directory);

AFDataFrame loadStagingProspect(char const *directory);

AFDataFrame loadDimBroker(char const* directory, AFDataFrame& dimDate);

AFDataFrame loadStagingCustomer(char const* directory);

AFDataFrame loadStagingCashBalances(char const* directory);

AFDataFrame loadStagingWatches(char const* directory);

AFDataFrame loadStagingTrade(char const* directory);

AFDataFrame loadStagingTradeHistory(char const* directory);

Customer splitCustomer(AFDataFrame &&s_Customer);

AFDataFrame loadDimCustomer(Customer &s_Customer, AFDataFrame &taxRate, AFDataFrame &prospect);

AFDataFrame loadDimAccount(Customer &stagingCustomer);

AFDataFrame loadDimCompany(AFDataFrame& s_Company, AFDataFrame& industry, AFDataFrame& statusType, AFDataFrame& dimDate);

AFDataFrame loadFinancial(AFDataFrame &s_Financial, AFDataFrame &dimCompany);

AFDataFrame loadDimSecurity(AFDataFrame &s_Security, AFDataFrame &dimCompany, AFDataFrame &StatusType);

AFDataFrame loadProspect(AFDataFrame &s_Prospect, AFDataFrame &batchDate);

af::array phoneNumberProcessing(af::array &ctry, af::array &area, af::array &local, af::array &ext);

#endif //ARRAYFIRE_TPCDI_TPC_DI_H

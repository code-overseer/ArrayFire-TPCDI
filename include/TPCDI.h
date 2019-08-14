

//
// Created by Bryan Wong on 2019-06-28.
//

#ifndef ARRAYFIRE_TPCDI_TPCDI_H
#define ARRAYFIRE_TPCDI_TPCDI_H
#include "AFDataFrame.h"
#include "AFParser.hpp"
#include "TPCDI_Utils.h"
#include "FinwireParser.h"
#include <utility>
#include <memory>

struct Customer {
public:
    AFDataFrame* newCust;
    AFDataFrame* addAcct;
    AFDataFrame* updAcct;
    AFDataFrame* closeAcct;
    AFDataFrame* updCust;
    AFDataFrame* inact;
    Customer(AFDataFrame* newC, AFDataFrame* addA, AFDataFrame* updA, AFDataFrame* closeA, AFDataFrame* updC, AFDataFrame* in) :
    newCust(newC), addAcct(addA), updAcct(updA), closeAcct(closeA), updCust(updC), inact(in) {}
    virtual ~Customer() { delete newCust; delete addAcct; delete updAcct; delete closeAcct; delete updCust; delete inact; }
};

AFDataFrame loadBatchDate(char const* directory);

AFDataFrame loadDimDate(char const* directory);

AFDataFrame loadDimTime(char const* directory);

AFDataFrame loadIndustry(char const* directory);

AFDataFrame loadStatusType(char const* directory);

AFDataFrame loadTaxRate(char const* directory);

AFDataFrame loadTradeType(char const* directory);

AFDataFrame loadAudit(char const* directory);

AFDataFrame loadStagingSecurity(char const* directory);
AFDataFrame loadStagingCompany(char const *directory);
AFDataFrame loadStagingFinancial(char const *directory);

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

AFDataFrame loadFinancial(AFDataFrame &&s_Financial, AFDataFrame const &dimCompany);

AFDataFrame loadDimSecurity(AFDataFrame &&s_Security, AFDataFrame &dimCompany, AFDataFrame &StatusType);

AFDataFrame loadProspect(AFDataFrame &s_Prospect, AFDataFrame &batchDate);

#endif //ARRAYFIRE_TPCDI_TPCDI_H

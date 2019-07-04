#include <utility>

#include <utility>

//
// Created by Bryan Wong on 2019-06-28.
//

#ifndef ARRAYFIRE_TPCDI_TPC_DI_H
#define ARRAYFIRE_TPCDI_TPC_DI_H
#include "AFDataFrame.h"
#include "AFParser.hpp"

typedef std::shared_ptr<AFDataFrame> AFDF_ptr;
struct Finwire {
public:
    AFDF_ptr company;
    AFDF_ptr financial;
    AFDF_ptr security;
    Finwire(AFDF_ptr cmp, AFDF_ptr fin, AFDF_ptr sec);
    Finwire(Finwire&& other);
};

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

AFDataFrame loadStagingCashBalances(char const* directory);

AFDataFrame loadStagingWatches(char const* directory);

AFDataFrame loadDimCustomer(char const* directory);

#endif //ARRAYFIRE_TPCDI_TPC_DI_H

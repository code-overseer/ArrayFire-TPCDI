//
// Created by Bryan Wong on 2019-06-28.
//

#ifndef ARRAYFIRE_TPCDI_TPC_DI_H
#define ARRAYFIRE_TPCDI_TPC_DI_H
#include "AFDataFrame.h"
#include "AFParser.hpp"

AFDataFrame loadDimDate(char const* filepath);

AFDataFrame loadDimTime(char const* filepath);

AFDataFrame loadIndustry(char const* filepath);

AFDataFrame loadStatusType(char const* filepath);

AFDataFrame loadTaxRate(char const* filepath);

AFDataFrame loadTradeType(char const* filepath);

AFDataFrame loadAudit(char const* folderpath);

AFDataFrame loadDimBroker(char const* filepath, AFDataFrame& dimDate);

AFDataFrame loadDimCustomer(char const* filepath);

#endif //ARRAYFIRE_TPCDI_TPC_DI_H

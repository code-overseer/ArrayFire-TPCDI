#ifndef ARRAYFIRE_TPCDI_COLUMNNAMES_H
#define ARRAYFIRE_TPCDI_COLUMNNAMES_H

#include "AFDataFrame.h"

void inline nameStagingCompany(AFDataFrame &company) {
    company.name("S_Company");
    company.nameColumn("PTS", 0);
    company.nameColumn("REC_TYPE", 1);
    company.nameColumn("COMPANY_NAME", 2);
    company.nameColumn("CIK", 3);
    company.nameColumn("STATUS", 4);
    company.nameColumn("INDUSTRY_ID", 5);
    company.nameColumn("SP_RATING", 6);
    company.nameColumn("FOUNDING_DATE", 7);
    company.nameColumn("ADDR_LINE_1", 8);
    company.nameColumn("ADDR_LINE_2", 9);
    company.nameColumn("POSTAL_CODE", 10);
    company.nameColumn("CITY", 11);
    company.nameColumn("STATE_PROVINCE", 12);
    company.nameColumn("COUNTRY", 13);
    company.nameColumn("CEO_NAME", 14);
    company.nameColumn("DESCRIPTION", 15);
}

void inline nameStagingFinancial(AFDataFrame &financial) {
    financial.name("S_Financial");
    financial.nameColumn("PTS", 0);
    financial.nameColumn("REC_TYPE", 1);
    financial.nameColumn("YEAR", 2);
    financial.nameColumn("QUARTER", 3);
    financial.nameColumn("QTR_START_DATE", 4);
    financial.nameColumn("POSTING_DATE", 5);
    financial.nameColumn("REVENUE", 6);
    financial.nameColumn("EARNINGS", 7);
    financial.nameColumn("EPS", 8);
    financial.nameColumn("DILUTED_EPS", 9);
    financial.nameColumn("MARGIN", 10);
    financial.nameColumn("INVENTORY", 11);
    financial.nameColumn("ASSETS", 12);
    financial.nameColumn("LIABILITIES", 13);
    financial.nameColumn("SH_OUT", 14);
    financial.nameColumn("DILUTED_SH_OUT", 15);
    financial.nameColumn("CO_NAME_OR_CIK", 16);
}

void inline nameStagingSecurity(AFDataFrame &security) {
    security.name("S_Security");
    security.nameColumn("PTS", 0);
    security.nameColumn("REC_TYPE", 1);
    security.nameColumn("SYMBOL", 2);
    security.nameColumn("ISSUE_TYPE", 3);
    security.nameColumn("STATUS", 4);
    security.nameColumn("NAME", 5);
    security.nameColumn("EX_ID", 6);
    security.nameColumn("SH_OUT", 7);
    security.nameColumn("FIRST_TRADE_DATE", 8);
    security.nameColumn("FIRST_TRADE_EXCHANGE", 9);
    security.nameColumn("DIVIDEND", 10);
    security.nameColumn("CO_NAME_OR_CIK", 11);
}

void inline nameStagingProspect(AFDataFrame &s_prospect) {
    s_prospect.name("StaginProspect");
    s_prospect.nameColumn("AgencyID", 0);
    s_prospect.nameColumn("LastName", 1);
    s_prospect.nameColumn("FirstName", 2);
    s_prospect.nameColumn("MiddleInitial", 3);
    s_prospect.nameColumn("Gender", 4);
    s_prospect.nameColumn("AddressLine1", 5);
    s_prospect.nameColumn("AddressLine2", 6);
    s_prospect.nameColumn("PostalCode", 7);
    s_prospect.nameColumn("City", 8);
    s_prospect.nameColumn("State", 9);
    s_prospect.nameColumn("Country", 10);
    s_prospect.nameColumn("Phone", 11);
    s_prospect.nameColumn("Income", 12);
    s_prospect.nameColumn("NumberCars", 13);
    s_prospect.nameColumn("NumberChildren", 14);
    s_prospect.nameColumn("MaritalStatus", 15);
    s_prospect.nameColumn("Age", 16);
    s_prospect.nameColumn("CreditRating", 17);
    s_prospect.nameColumn("OwnOrRentFlag", 18);
    s_prospect.nameColumn("Employer", 19);
    s_prospect.nameColumn("NumberCreditCards", 20);
    s_prospect.nameColumn("NetWorth", 21);
}

void inline nameDimCompany(AFDataFrame &dimCompany) {
    dimCompany.name("DimCompany");
    dimCompany.nameColumn("CompanyID", 1);
    dimCompany.nameColumn("Status", 2);
    dimCompany.nameColumn("Name", 3);
    dimCompany.nameColumn("Industry", 4);
    dimCompany.nameColumn("SPrating", 5);
    dimCompany.nameColumn("CEO", 7);
    dimCompany.nameColumn("AddressLine1", 8);
    dimCompany.nameColumn("AddressLine2", 9);
    dimCompany.nameColumn("PostalCode", 10);
    dimCompany.nameColumn("City", 11);
    dimCompany.nameColumn("StateProv", 12);
    dimCompany.nameColumn("Country", 13);
    dimCompany.nameColumn("Description", 14);
    dimCompany.nameColumn("FoundingDate", 15);
    dimCompany.nameColumn("IsCurrent", 16);
    dimCompany.nameColumn("BatchID", 17);
    dimCompany.nameColumn("EffectiveDate", 18);
    dimCompany.nameColumn("EndDate", 19);
}

void inline nameFinancial(AFDataFrame &financial) {
    financial.name("Financial");
    financial.nameColumn("SK_CompanyID", 0);
    financial.nameColumn("FI_YEAR", 1);
    financial.nameColumn("FI_QTR", 2);
    financial.nameColumn("FI_QTR_START_DATE", 3);
    financial.nameColumn("FI_REVENUE", 4);
    financial.nameColumn("FI_NET_EARN", 5);
    financial.nameColumn("FI_BASIC_EPS", 6);
    financial.nameColumn("FI_DILUT_EPS", 7);
    financial.nameColumn("FI_MARGIN", 8);
    financial.nameColumn("FI_INVENTORY", 9);
    financial.nameColumn("FI_ASSETS", 10);
    financial.nameColumn("FI_LIABILITY", 11);
    financial.nameColumn("FI_OUT_BASIC", 12);
    financial.nameColumn("FI_OUT_DILUT", 13);
}

void inline  nameDimSecurity(AFDataFrame &dimSecurity) {
    dimSecurity.name("DimSecurity");
    dimSecurity.nameColumn("SK_SecurityID", 0);
    dimSecurity.nameColumn("Symbol", 1);
    dimSecurity.nameColumn("Issue", 2);
    dimSecurity.nameColumn("Status", 3);
    dimSecurity.nameColumn("Name", 4);
    dimSecurity.nameColumn("ExchangeID", 5);
    dimSecurity.nameColumn("SK_CompanyID", 6);
    dimSecurity.nameColumn("SharesOutstanding", 7);
    dimSecurity.nameColumn("FirstTrade", 8);
    dimSecurity.nameColumn("FirstTradeOnExchange", 9);
    dimSecurity.nameColumn("Dividend", 10);
    dimSecurity.nameColumn("IsCurrent", 11);
    dimSecurity.nameColumn("BatchID", 12);
    dimSecurity.nameColumn("EffectiveDate", 13);
    dimSecurity.nameColumn("EndDate", 14);
}

#endif //ARRAYFIRE_TPCDI_COLUMNNAMES_H

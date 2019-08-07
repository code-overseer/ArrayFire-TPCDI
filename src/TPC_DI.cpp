#include <utility>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <rapidxml.hpp>
#include "include/TPC_DI.h"
#include "include/BatchFunctions.h"
#include "include/Logger.h"

namespace fs = boost::filesystem;
namespace xml = rapidxml;
using namespace af;
using namespace TPCDI_Utils;
using namespace BatchFunctions;

AFDataFrame loadBatchDate(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "BatchDate.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    frame.add(Column(constant(1, 1, u32), UINT));
    frame.add(parser.asDate(0, true, YYYYMMDD));

    return frame;
}

/* Independent Static Tables */
AFDataFrame loadDimDate(char const *directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Date.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asDate(1, true, YYYYMMDD));
    for (int i = 2;  i < 17; i += 2) {
        frame.add(parser.parse<char*>(i));
        frame.add(parser.parse<unsigned int>(i + 1));
    }
    frame.add(parser.parse<bool>(17));
    return frame;
}

AFDataFrame loadDimTime(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Time.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asTime(1, false));

    for (int i = 2;  i < 7; i += 2) {
        frame.add(parser.parse<unsigned int>(i));
        frame.add(parser.parse<char*>(i + 1));
    }
    frame.add(parser.parse<bool>(8));
    frame.add(parser.parse<bool>(9));
    return frame;
}

AFDataFrame loadIndustry(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Industry.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    for (int i = 0;  i < 3; ++i) frame.add(parser.parse<char*>(i));
    frame.name("Industry");
    frame.nameColumn("IN_ID", 0);
    frame.nameColumn("IN_NAME", 1);
    frame.nameColumn("IN_SC_ID", 2);
    return frame;
}

AFDataFrame loadStatusType(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "StatusType.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.parse<char*>(0), "ST_ID");
    frame.add(parser.parse<char*>(1), "ST_NAME");
    frame.name("StatusType");
    return frame;
}

AFDataFrame loadTaxRate(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TaxRate.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    frame.name("TaxRate");
    frame.add(parser.parse<char*>(0), "TX_ID");
    frame.add(parser.parse<char*>(1), "TX_NAME");
    frame.add(parser.parse<float>(2), "TX_RATE");
    return frame;
}

AFDataFrame loadTradeType(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TradeType.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    for (int i = 0;  i < 2; ++i) frame.add(parser.parse<char*>(i));
    for (int i = 2;  i < 4; ++i) frame.add(parser.parse<unsigned int>(i));
    return frame;
}

AFDataFrame loadAudit(char const* directory) {
    AFDataFrame frame;

    std::string dir(directory);
    std::vector<std::string> auditFiles;
    fs::directory_iterator end_itr;

    for( fs::directory_iterator i( directory ); i != end_itr; ++i ) {
        auto n = i->path().string().find("_audit.csv", dir.size());
        if( n == std::string::npos) continue;
        auditFiles.push_back( i->path().string() );
    }

    std::sort(auditFiles.begin(), auditFiles.end());

    AFParser parser(auditFiles, ',', true);
    frame.add(parser.parse<char*>(0));
    frame.add(parser.parse<unsigned int>(1));
    frame.add(parser.asDate(2, true, YYYYMMDD));
    frame.add(parser.parse<char*>(3));
    frame.add(parser.parse<int>(4));
    frame.add(parser.parse<double>(5));

    return frame;
}

inline void nameStagingCompany(AFDataFrame &company) {
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

inline void nameStagingFinancial(AFDataFrame &financial) {
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

inline void nameStagingSecurity(AFDataFrame &security) {
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

Finwire loadStagingFinwire(char const *directory) {
    std::string dir(directory);
    std::vector<std::string> finwireFiles;
    fs::directory_iterator end_itr;
    for( fs::directory_iterator i( directory ); i != end_itr; ++i )
    {
        auto n = i->path().string().find("FINWIRE", dir.size());
        if( n == std::string::npos) continue;
        n = i->path().string().find("_audit.csv", dir.size() + 13);
        if( n != std::string::npos) continue;
        finwireFiles.push_back( i->path().string() );
    }
    sort(finwireFiles.begin(), finwireFiles.end());
    FinwireParser parser(finwireFiles);
    Finwire finwire = parser.extractData();
    nameStagingCompany(finwire.company);
    nameStagingFinancial(finwire.financial);
    nameStagingSecurity(finwire.security);
    return finwire;
}

AFDataFrame loadStagingProspect(char const *directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Prospect.csv");
    AFDataFrame frame;
    AFParser parser(file, ',', false);

    for (int i = 0;  i < 12; ++i) frame.add(parser.parse<char*>(i));

    frame.add(parser.parse<unsigned long long>(12));

    for (int i = 13;  i < 15; ++i) frame.add(parser.parse<unsigned char>(i));

    frame.add(parser.parse<char*>(15));
    frame.add(parser.parse<unsigned short>(16));
    frame.add(parser.parse<unsigned int>(17));

    for (int i = 18;  i < 20; ++i) frame.add(parser.parse<char*>(i));

    frame.add(parser.parse<unsigned char>(20));
    frame.add(parser.parse<unsigned long long>(21));

    return frame;
}

AFDataFrame loadStagingCustomer(char const* directory) {

    std::string data = XML_Parser::flattenCustomerMgmt(directory);
    
    AFParser parser(data, '|', false);
    AFDataFrame frame;
    frame.add(parser.parse<char*>(0));

    frame.add(parser.asDateTime(1, true, YYYYMMDD));

    frame.add(parser.parse<unsigned long long>(2));

    for (int i = 3; i < 5; ++i) frame.add(parser.parse<char*>(i));

    frame.add(parser.parse<unsigned char>(5));
    frame.add(parser.asDate(6, true, YYYYMMDD));

    for (int i = 7; i < 32; ++i) frame.add(parser.parse<char*>(i));

    frame.add(parser.parse<unsigned long long>(32));
    frame.add(parser.parse<unsigned short>(33));
    frame.add(parser.parse<unsigned long long>(34));
    frame.add(parser.parse<char*>(35));

    return frame;
}

AFDataFrame loadDimBroker(char const* directory, AFDataFrame& dimDate) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "HR.csv");
    AFDataFrame frame;
    {
        AFParser parser(file, ',');
        frame.add(parser.parse<unsigned int>(0));
        frame.add(parser.parse<unsigned int>(1));
        for (int i = 2;  i <= 7; ++i) frame.add(parser.parse<char*>(i));
        auto idx = frame.column(5) == "314";
        frame = frame.select(idx, "DimBroker");
        frame.remove(5);
    }
    auto length = frame.columns()[0].dims(1);
    frame.insert(Column(range(dim4(1, length), 1, u64), ULONG), 0);
    frame.add(Column(constant(1, dim4(1, length), b8), BOOL));
    frame.add(Column(constant(1, dim4(1, length), u32), UINT));
    dimDate.sortBy(0);
    auto date = dimDate.column(1)(0);
    frame.add(Column(tile(date, dim4(1, length)), DATE));
    frame.add(Column(tile(Column::endDate(), dim4(1, length)), DATE));
    return frame;
}

AFDataFrame loadStagingCashBalances(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "CashTransaction.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asDateTime(1, true, YYYYMMDD));
    frame.add(parser.parse<double>(2));
    frame.add(parser.parse<char*>(3));
    return frame;
}

AFDataFrame loadStagingWatches(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "WatchHistory.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.parse<char*>(1));
    frame.add(parser.asDateTime(2, true, YYYYMMDD));
    frame.add(parser.parse<char*>(3));
    return frame;
}

inline void nameDimCompany(AFDataFrame &dimCompany) {
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

AFDataFrame loadDimCompany(AFDataFrame& s_Company, AFDataFrame& industry, AFDataFrame& statusType, AFDataFrame& dimDate) {
    auto dimCompany = s_Company.equiJoin(industry,5,0);

    dimCompany = dimCompany.equiJoin(statusType, 4, 0);
    {
        std::string order[15] = {
                "CIK","StatusType.ST_NAME","COMPANY_NAME", "Industry.IN_NAME","SP_RATING","CEO_NAME", "ADDR_LINE_1",
                "ADDR_LINE_2", "POSTAL_CODE","CITY","STATE_PROVINCE","COUNTRY","DESCRIPTION","FOUNDING_DATE", "PTS"
        };
        dimCompany = dimCompany.project(order, 15, "DimCompany");
    }
    dimCompany.column("PTS").toDate();
    dimCompany.nameColumn("EffectiveDate", "PTS");

    dimCompany.insert(Column(range(dim4(1, dimCompany.length()), 1, u64), ULONG), 0, "SK_CompanyID");
    dimCompany.insert(Column(constant(1, dim4(1, dimCompany.length()), b8), BOOL), dimCompany.columns().size() - 1, "IsCurrent");
    dimCompany.insert(Column(constant(1, dim4(1, dimCompany.length()), u32), UINT), dimCompany.columns().size() - 1, "BatchID");
    dimCompany.add(Column(tile(Column::endDate(), dim4(1, dimCompany.length())), DATE), "EndDate");

    {
        auto &rating = dimCompany.column("SP_RATING");
        auto lowGrade = rating((array)rating.index(0, af::span)) != 'A';
        array col = rating(batchFunc(rating.index(0, af::span), range(dim4(3), u32), batchAdd));
        lowGrade = lowGrade && anyTrue(batchFunc(col, array(3,1,"BBB").as(u8), BatchFunctions::batchNotEqual), 0);
        lowGrade.eval();
        dimCompany.insert(Column(lowGrade, BOOL), 6, "isLowGrade");
    }

    nameDimCompany(dimCompany);

    std::string order[3] = { "SK_CompanyID", "CompanyID", "EffectiveDate"};
    auto s0 = dimCompany.project(order, 3, "S0");
    s0 = s0.select(flip(flat(seq(s0.length())),0), "S0");
    s0.sortBy(order + 1, 2);

    auto s2 = s0.project(order + 1, 1, "S2");
    auto s1 = s0.select(range(dim4(s0.length() - 1), 0, u64) + 1, "S1");
    s2 = s2.select(range(dim4(s2.length() - 1), 0, u64), "S2");
    s1 = s1.zip(s2);
    s1 = s1.select(where(s1.column("CompanyID") == s1.column("S2.CompanyID")), "S1");
    auto end_date = s1.project(order + 2, 1, "EndDate");

    s0 = s0.project(order, 2, "S0");
    s1 = s0.project(order + 1, 1, "S1");
    s0 = s0.select(range(dim4(s0.length() - 1), 0, u64), "S0");
    s1 = s1.select(range(dim4(s1.length() - 1), 0, u64) + 1, "S1");
    s0 = s0.zip(s1);
    s0 = s0.select(where(s0.column("CompanyID") == s0.column("S1.CompanyID")), "S0");
    s0 = s0.project(order, 1, "S0");
    s0 = s0.zip(end_date);
    s0.nameColumn("EndDate", "EndDate.EffectiveDate");

    auto out = AFDataFrame::setCompare(dimCompany.column("SK_CompanyID").data(), s0.column("SK_CompanyID").data());
    dimCompany.column("IsCurrent")(out.first) = 0;
    dimCompany.column("EndDate")(span, out.first) = (array) s0.column("EndDate")(span, out.second);

    return dimCompany;
}

inline void nameFinancial(AFDataFrame &financial) {
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

AFDataFrame loadFinancial(AFDataFrame &s_Financial, AFDataFrame &dimCompany) {
    AFDataFrame financial;
    std::string columns[4] = {"SK_CompanyID", "CompanyID", "EffectiveDate", "EndDate"};
    auto tmp = dimCompany.project(columns, 4, "DC");

    auto fin1 = s_Financial.select(where(s_Financial.column("CO_NAME_OR_CIK").left(1) == '0'));
    fin1.column("CO_NAME_OR_CIK") = fin1.column("CO_NAME_OR_CIK").trim(0, 10);
    fin1.column("CO_NAME_OR_CIK").cast<unsigned long long>();

    financial = fin1.equiJoin(tmp, "CO_NAME_OR_CIK", "CompanyID");
    financial.remove("CO_NAME_OR_CIK");

    columns[1] = "Name";
    tmp = dimCompany.project(columns, 4, "DC");
    fin1 = s_Financial.select(where(s_Financial.column("CO_NAME_OR_CIK").left(1) != '0'));
    fin1 = fin1.equiJoin(tmp, "CO_NAME_OR_CIK", "Name");
    fin1.remove("CO_NAME_OR_CIK");
    if (!fin1.column(0).isempty()) financial = financial.concatenate(fin1);
    af::sync();
    fin1.clear();
    tmp.clear();
    financial.column("PTS").toDate();
    auto cond1 = financial.column("DC.EffectiveDate").hash() <= financial.column("PTS").hash();
    auto cond2 = financial.column("DC.EndDate").hash() > financial.column("PTS").hash();
    financial = financial.select(where(cond1 && cond2), "");

    std::string order[14] = {
            "DC.SK_CompanyID", "YEAR", "QUARTER", "QTR_START_DATE","REVENUE", "EARNINGS",
            "EPS", "DILUTED_EPS","MARGIN","INVENTORY","ASSETS","LIABILITIES","SH_OUT", "DILUTED_SH_OUT"
    };
    financial = financial.project(order, 14, "Financial");

    // Renames columns
    nameFinancial(financial);

    return financial;
}

inline void nameDimSecurity(AFDataFrame &dimSecurity) {
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

AFDataFrame loadDimSecurity(AFDataFrame &s_Security, AFDataFrame &dimCompany, AFDataFrame &StatusType) {
    AFDataFrame security;

    {
        std::string columns[4] = {"SK_CompanyID", "CompanyID", "EffectiveDate", "EndDate"};

        auto tmp = dimCompany.project(columns, 4, "DC");
        auto security1 = s_Security.select(where(s_Security.column("CO_NAME_OR_CIK").left(1) == '0'));
        security1.name(s_Security.name());
        security1.column("CO_NAME_OR_CIK") = security1.column("CO_NAME_OR_CIK").trim(0, 10);
        security1.column("CO_NAME_OR_CIK").cast<unsigned long long>();
        security = security1.equiJoin(tmp, "CO_NAME_OR_CIK", "CompanyID");
        security.remove("CO_NAME_OR_CIK");

        columns[1] = "Name";
        tmp = dimCompany.project(columns, 4, "DC");
        security1 = s_Security.select(where(s_Security.column("CO_NAME_OR_CIK").left(1) != '0'));
        security1.name(s_Security.name());
        security1 = security1.equiJoin(tmp, "CO_NAME_OR_CIK", "Name");
        security1.remove("CO_NAME_OR_CIK");
        security = security.concatenate(security1);
        columns[0] = "ST_ID";
        tmp = StatusType.project(columns, 1, "ST");
        security = security.equiJoin(tmp, "STATUS", "ST_ID");
    }
    security.column("PTS").toDate();
    security.nameColumn("EffectiveDate", "PTS");

    auto cond1 = security.column("DC.EffectiveDate").hash() <= security.column("EffectiveDate").hash();
    auto cond2 = security.column("DC.EndDate").hash() > security.column("EffectiveDate").hash();
    security = security.select(where(cond1 && cond2), "");
    {
        std::string order[11] = {
                "SYMBOL","ISSUE_TYPE", "STATUS","NAME","EX_ID", "DC.SK_CompanyID",
                "SH_OUT", "FIRST_TRADE_DATE","FIRST_TRADE_EXCHANGE","DIVIDEND", "EffectiveDate"
        };
        security = security.project(order, 11, "DimSecurity");
    }

    auto length = security.length();
    security.insert(Column(range(dim4(1, length), 1, u64), ULONG), 0, "SK_SecurityID");
    security.insert(Column(constant(1, dim4(1, length), b8), BOOL), security.columns().size() - 1, "IsCurrent");
    security.insert(Column(constant(1, dim4(1, length), u32), UINT), security.columns().size() - 1, "BatchID");
    security.add(Column(tile(Column::endDate(), dim4(1, length)), DATE), "EndDate");

    nameDimSecurity(security);
    std::string order[3] = { "SK_SecurityID", "Symbol", "EffectiveDate"};
    auto s0 = security.project(order, 3, "S0");
    s0.sortBy(order + 1, 2);

    auto s2 = s0.project(order + 1, 1, "S2");
    auto s1 = s0.select(range(dim4(s0.length() - 1), 0, u64) + 1);
    s2 = s2.select(range(dim4(s2.length() - 1), 0, u64));
    s1 = s1.zip(s2);
    s1 = s1.select(where(allTrue(s1.column("Symbol") == s1.column("S2.Symbol"), 0)));
    auto end_date = s1.project(order + 2, 1, "EndDate");
    if (end_date.isEmpty()) return security;

    s0 = s0.project(order, 2, "S0");
    s1 = s0.project(order + 1, 1, "S1");
    s0 = s0.select(range(dim4(s0.length() - 1), 0, u64));
    s1 = s1.select(range(dim4(s1.length() - 1), 0, u64) + 1);
    s0 = s0.zip(s1);
    s0 = s0.select(where(allTrue(s0.column("Symbol") == s0.column("S1.Symbol"))));
    s0.project(order, 1, "S0");
    s0 = s0.zip(end_date);
    s0.nameColumn("EndDate", "EndDate.EffectiveDate");

    auto out = AFDataFrame::setCompare(security.column("SK_SecurityID"), s0.column("SK_SecurityID"));
    security.column("IsCurrent")(out.first) = 0;
    security.column("EndDate")(span, out.first) = (array) s0.column("EndDate")(span, out.second);
    return security;
}

inline array marketingNameplate(array const &networth, array const &income, array const &cards, array const &children,
                         array const &age, array const &credit, array const &cars) {
    auto out = constant(0, 56, networth.dims(1), u8);
    array val(12, 6, "+HighValue\0\0+Expenses\0\0\0+Boomer\0\0\0\0\0+MoneyAlert\0+Spender\0\0\0\0+Inherited\0\0");
    val = val.as(u8);
    auto idx = constant(0, 6, networth.dims(1), b8);
    idx(0, where(networth > 1000000 || income > 200000)) = 1;
    idx(1, where(cards > 5 || children > 3)) = 1;
    idx(2, where(age > 45)) = 1;
    idx(3, where(credit < 600 || income < 5000 || networth < 100000)) = 1;
    idx(4, where(cars > 3 || cards > 7)) = 1;
    idx(5, where(age > 25 || networth > 1000000)) = 1;

    auto j = where(idx.row(0));
    out(seq(0,9,1), j) = tile(val(seq(0,9,1), 0), dim4(1, j.dims(0)));
    j = where(idx.row(1));
    out(seq(10,18,1), j) = tile(val(seq(0,8,1), 1), dim4(1, j.dims(0)));
    j = where(idx.row(2));
    out(seq(19,25,1), j) = tile(val(seq(0,6,1), 2), dim4(1, j.dims(0)));
    j = where(idx.row(3));
    out(seq(26,36,1), j) = tile(val(seq(0,10,1), 3), dim4(1, j.dims(0)));
    j = where(idx.row(4));
    out(seq(37,44,1), j) = tile(val(seq(0,7,1), 4), dim4(1, j.dims(0)));
    j = where(idx.row(5));
    out(seq(45,54,1), j) = tile(val(seq(0,9,1), 5), dim4(1, j.dims(0)));
    out(55, span) = '\n';
    out = out(where(out)).as(u32);
    idx = hflat(where(out == '\n'));
    out(idx) = 0;
    auto tmp = join(1, constant(0, 1, u32), idx.cols(0, end - 1) + 1);
    idx = join(0, tmp, idx);
    auto h = sum<uint32_t>(max(diff1(idx, 0))) + 1;
    tmp = batchFunc(idx.row(0), range(dim4(h), 0, u32), BatchFunctions::batchAdd);
    tmp(where(batchFunc(tmp, idx.row(1), BatchFunctions::batchGreater))) = UINT32_MAX;
    idx = where(tmp != UINT32_MAX);
    tmp(idx) = out;
    out = tmp;
    out(where(out == UINT32_MAX)) = 0;
    out = moddims(out, dim4(h, out.elements() / h)).as(u8);
    out.eval();

    return out;
}

void nameStagingProspect(AFDataFrame &s_prospect) {
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

AFDataFrame loadProspect(AFDataFrame &s_Prospect, AFDataFrame &batchDate) {
    nameStagingProspect(s_Prospect);
    AFDataFrame prospect(std::move(s_Prospect));

    auto dim = prospect.column("Age").dims();
    auto batchID = 1;
    prospect.name("Prospect");

    auto col = Column(tile(batchDate.column(1)(span, where(batchDate.column(0) == batchID)), dim), DATE);
    prospect.insert(Column(col.hash(), ULONG), 1, "SK_RecordDateID");
    prospect.insert(Column(array(prospect.column("SK_RecordDateID").data()), ULONG), 2, "SK_UpdateDateID");
    prospect.insert(Column(constant(1, dim, u32), UINT), 3, "BatchID");
    prospect.insert(Column(constant(0, dim, u8), BOOL), 4, "IsCustomer");
    auto tmp = marketingNameplate(prospect.column("NetWorth").data(), prospect.column("Income").data(),
                                  prospect.column("NumberCreditCards").data(),
                                  prospect.column("NumberChildren").data(), prospect.column("Age").data(),
                                  prospect.column("CreditRating").data(),
                                  prospect.column("NumberCars").data());
    prospect.add(Column(tmp, STRING), "MarketingNameplate");

    return prospect;
}

AFDataFrame loadStagingTrade(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Trade.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asDateTime(1, true, YYYYMMDD));
    frame.add(parser.parse<char*>(2));
    frame.add(parser.parse<char*>(3));
    frame.add(parser.parse<bool>(4));
    frame.add(parser.parse<char*>(5));
    frame.add(parser.parse<unsigned int>(6));
    frame.add(parser.parse<double>(7));
    frame.add(parser.parse<unsigned int>(8));
    frame.add(parser.parse<unsigned long long>(9));
    frame.add(parser.parse<double>(10));
    frame.add(parser.parse<double>(11));
    frame.add(parser.parse<double>(12));
    frame.add(parser.parse<double>(13));
    return frame;
}

AFDataFrame loadStagingTradeHistory(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TradeHistory.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asDateTime(1, true, YYYYMMDD));
    frame.add(parser.parse<char*>(2));
    return frame;
}

Customer splitCustomer(AFDataFrame &&s_Customer) {
    auto idx = s_Customer.stringMatch(0, "NEW");
    auto newC = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer.stringMatch(0, "ADDACCT");
    auto add = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer.stringMatch(0, "UPDACCT");
    auto uAcc = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer.stringMatch(0, "CLOSEACCT");
    auto cAcc = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer.stringMatch(0, "UPDCUST");
    auto uCus = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer.stringMatch(0, "INACT");
    auto inAc = new AFDataFrame(s_Customer.select(idx));

    return Customer(newC, add, uAcc, cAcc, uCus, inAc);
}

inline af::array phoneNumberProcessing(af::array const &ctry, af::array const &area, af::array const &local, af::array const &ext) {
    auto const cond1 = where(ctry.row(0) && area.row(0) && local.row(0));
    auto const cond2 = where(ctry.row(0) == 0 &&  area.row(0) && local.row(0));
    auto const cond3 = where(area.row(0) == 0 && local.row(0));
    auto const extNotNull = where(ext.row(0));
    auto const c = ctry.dims(0);
    auto const a = area.dims(0);
    auto const l = local.dims(0);
    auto const e = ext.dims(0);

    auto out = constant(0, dim4(c + a + l + e + 4, ctry.dims(1)), u8);

    out(0, cond1) = '+';

    auto idx = range(dim4(c), 0, u32) + 1;
    idx = batchFunc(idx, hflat(cond1) * out.dims(0) , BatchFunctions::batchAdd);
    out(idx) = flat(ctry(span, cond1));
    auto j = join(0, cond1, cond2);
    out(c + 1, j) = '(';
    idx = range(dim4(a), 0, u32) + c + 2;
    idx = batchFunc(idx, hflat(j) * out.dims(0) , BatchFunctions::batchAdd);
    out(idx) = flat(area(span, j));
    out(2 + c + a, j) = ')';

    j = join(0, j, cond3);
    idx = range(dim4(l), 0, u32) + c + a + 3;
    idx = batchFunc(idx, hflat(j) * out.dims(0) , BatchFunctions::batchAdd);
    out(idx) = flat(local(span, j));

    j = join(0, j, extNotNull);
    idx = range(dim4(e), 0, u32) + c + a + l + 3;
    idx = batchFunc(idx, hflat(j) * out.dims(0) , BatchFunctions::batchAdd);
    out(idx) = flat(ext(span, j));

    out(3 + c + a + l + e, span) = '\n';

    out = out(where(out)).as(u32);
    idx = hflat(where(out == '\n'));
    out(idx) = 0;
    auto tmp = join(1, constant(0, 1, u32), idx.cols(0, end - 1) + 1);
    idx = join(0, tmp, idx);
    auto h = sum<uint32_t>(max(diff1(idx, 0))) + 1;
    tmp = batchFunc(idx.row(0), range(dim4(h), 0, u32), BatchFunctions::batchAdd);
    tmp(where(batchFunc(tmp, idx.row(1), BatchFunctions::batchGreater))) = UINT32_MAX;
    idx = where(tmp != UINT32_MAX);
    tmp(idx) = out;
    out = tmp;
    out(where(out == UINT32_MAX)) = 0;
    out = moddims(out, dim4(h, out.elements() / h)).as(u8);
    out.eval();

    return out;
}

AFDataFrame loadDimCustomer(Customer &s_Customer, AFDataFrame &taxRate, AFDataFrame &prospect) {
    std::pair<int, char const*> input[] = {
            {2, "CustomerID"},
            {3, "TaxID"},
            {7, "LastName"},
            {8, "FirstName"},
            {9, "MiddleInitial"},
            {4, "Gender"},
            {5, "Tier"},
            {6, "DOB"},
            {10, "AddressLine1"},
            {11, "AddressLine2"},
            {12, "PostalCode"},
            {13, "City"},
            {14, "StateProvince"},
            {15, "Country"},
            {16, "Email1"},
            {17, "Email2"}
    };
    auto &frame = *s_Customer.newCust;
    auto dim = frame.column(2).dims();
    AFDataFrame dimCustomer;
    dimCustomer.name("DimCustomer");
    for (auto const &i: input) dimCustomer.add(std::move(frame.column(i.first)), i.second);

    dimCustomer.insert(Column(tile(array(dim4(7), "ACTIVE"), dim), STRING), 2, "Status");
    dimCustomer.insert(Column(phoneNumberProcessing(frame.column(18).data(), frame.column(19).data(),
                                                    frame.column(20).data(), frame.column(21).data()), STRING), 15, "Phone1");
    dimCustomer.insert(Column(phoneNumberProcessing(frame.column(22).data(), frame.column(23).data(),
                                                    frame.column(24).data(), frame.column(25).data()), STRING), 16, "Phone2");
    dimCustomer.insert(Column(phoneNumberProcessing(frame.column(26).data(), frame.column(27).data(),
                                                    frame.column(28).data(), frame.column(29).data()), STRING), 17, "Phone3");
    {
        AFDataFrame tmp;
        tmp.name("NationalTax");
        tmp.add(Column(frame.column(31)), "ID");
        tmp = tmp.equiJoin(taxRate, "ID", "TX_ID");
        dimCustomer.add(tmp.column("TaxRate.TX_NAME"), "NationalTaxRateDesc");
        dimCustomer.add(tmp.column("TaxRate.TX_NAME"), "NationalTaxRate");
    }

    {
        AFDataFrame tmp;
        tmp.name("LocalTax");
        tmp.add(Column(frame.column(30)), "ID");
        tmp = tmp.equiJoin(taxRate, "ID", "TX_ID");
        dimCustomer.add(tmp.column("TaxRate.TX_NAME"), "LocalTaxRateDesc");
        dimCustomer.add(tmp.column("TaxRate.TX_RATE"), "LocalTaxRate");
    }

    return dimCustomer;
}



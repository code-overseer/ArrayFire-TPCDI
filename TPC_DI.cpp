#include <utility>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <rapidxml.hpp>
#include "TPC_DI.h"
#include "BatchFunctions.h"
#include "Logger.h"

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

    frame.add(constant(1, 1, u32), UINT);
    frame.add(parser.asDate(0, true, YYYYMMDD), DATE);

    return frame;
}

/* Independent Static Tables */
AFDataFrame loadDimDate(char const *directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Date.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.asU64(0), U64);
    frame.add(parser.asDate(1, true, YYYYMMDD), DATE);
    for (int i = 2;  i < 17; i += 2) {
        frame.add(parser.asString(i), STRING);
        frame.add(parser.asUint(i + 1), UINT);
    }

    frame.add(parser.asBoolean(17), BOOL);
    return frame;
}

AFDataFrame loadDimTime(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Time.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    frame.add(parser.asU64(0), U64);
    frame.add(parser.asTime(1), TIME);

    for (int i = 2;  i < 7; i += 2) {
        frame.add(parser.asUint(i), UINT);
        frame.add(parser.asString(i + 1), STRING);
    }

    frame.add(parser.asBoolean(8), BOOL);
    frame.add(parser.asBoolean(9), BOOL);
    return frame;
}

AFDataFrame loadIndustry(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Industry.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    for (int i = 0;  i < 3; ++i) frame.add(parser.asString(i), STRING);
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
    for (int i = 0;  i < 2; ++i) frame.add(parser.asString(i), STRING);

    frame.name("StatusType");
    frame.nameColumn("ST_ID", 0);
    frame.nameColumn("ST_NAME", 1);
    return frame;
}

AFDataFrame loadTaxRate(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TaxRate.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    frame.name("TaxRate");
    frame.add(parser.asString(0), STRING, "TX_ID");
    frame.add(parser.asString(1), STRING, "TX_NAME");
    frame.add(parser.asFloat(2), FLOAT, "TX_RATE");
    return frame;
}

AFDataFrame loadTradeType(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TradeType.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    for (int i = 0;  i < 2; ++i) frame.add(parser.asString(i), STRING);
    for (int i = 2;  i < 4; ++i) frame.add(parser.asUint(i), UINT);
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
    frame.add(parser.asString(0), STRING);
    frame.add(parser.asUint(1), UINT);
    frame.add(parser.asDate(2, true, YYYYMMDD), DATE);
    frame.add(parser.asString(3), STRING);
    frame.add(parser.asInt(4), INT);
    frame.add(parser.asFloat(5), DOUBLE);

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

    for (int i = 0;  i < 12; ++i) frame.add(parser.asString(i), STRING);

    frame.add(parser.asU64(12), U64);

    for (int i = 13;  i < 15; ++i) frame.add(parser.asUchar(i), UCHAR);

    frame.add(parser.asString(15), STRING);
    frame.add(parser.asUshort(16), USHORT);
    frame.add(parser.asUint(17), UINT);

    for (int i = 18;  i < 20; ++i) frame.add(parser.asString(i), STRING);

    frame.add(parser.asUchar(20), UCHAR);
    frame.add(parser.asU64(21), U64);

    return frame;
}

AFDataFrame loadStagingCustomer(char const* directory) {

    std::string data = XML_Parser::flattenCustomerMgmt(directory);
    
    AFParser parser(data, '|', false);
    AFDataFrame frame;
    frame.add(parser.asString(0), STRING);

    frame.add(parser.asDateTime(1, true, YYYYMMDD), DATETIME);

    frame.add(parser.asU64(2), U64);

    for (int i = 3; i < 5; ++i) frame.add(parser.asString(i), STRING);

    frame.add(parser.asUchar(5), UCHAR);
    frame.add(parser.asDate(6, true, YYYYMMDD), DATE);

    for (int i = 7; i < 32; ++i) frame.add(parser.asString(i), STRING);

    frame.add(parser.asU64(32), U64);
    frame.add(parser.asUshort(33), USHORT);
    frame.add(parser.asU64(34), U64);
    frame.add(parser.asString(35), STRING);

    return frame;
}

AFDataFrame loadDimBroker(char const* directory, AFDataFrame& dimDate) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "HR.csv");
    AFDataFrame frame;
    {
        AFParser parser(file, ',');
        frame.add(parser.asUint(0), UINT);
        frame.add(parser.asUint(1), UINT);
        for (int i = 2;  i <= 7; ++i) frame.add(parser.asString(i), STRING);
        auto idx = frame.stringMatch(5, "314");
        print("HERE");
        frame = frame.select(idx);
        frame.remove(5);
    }
    auto length = frame.data()[0].dims(1);
    frame.insert(range(dim4(1, length), 1, u64), U64, 0);
    frame.add(constant(1, dim4(1, length), b8), BOOL);
    frame.add(constant(1, dim4(1, length), u32), UINT);
    auto date = sort(dimDate.data(0),1);
    date = date(0);
    frame.add(tile(dehashDate(date, YYYYMMDD), dim4(1, length)), DATE);
    frame.add(tile(endDate(), dim4(1, length)), DATE);
    return frame;
}

AFDataFrame loadStagingCashBalances(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "CashTransaction.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.asU64(0), U64);
    frame.add(parser.asDateTime(1, true, YYYYMMDD), DATE);
    frame.add(parser.asDouble(2), DOUBLE);
    frame.add(parser.asString(3), STRING);
    return frame;
}

AFDataFrame loadStagingWatches(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "WatchHistory.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.asU64(0), U64);
    frame.add(parser.asString(1), STRING);
    frame.add(parser.asDateTime(2, true, YYYYMMDD), DATE);
    frame.add(parser.asString(3), STRING);
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
    print(dimCompany.length());
    return dimCompany;

    dimCompany = dimCompany.equiJoin(statusType, 4, 0);
    print(dimCompany.length());
    {
        std::string order[15] = {
                "CIK","StatusType.ST_NAME","COMPANY_NAME", "Industry.IN_NAME","SP_RATING","CEO_NAME", "ADDR_LINE_1",
                "ADDR_LINE_2", "POSTAL_CODE","CITY","STATE_PROVINCE","COUNTRY","DESCRIPTION","FOUNDING_DATE", "PTS"
        };
        dimCompany = dimCompany.project(order, 15, "DimCompany");
    }

    dimCompany.data("PTS") = dimCompany.data("PTS")(range(dim4(3), 0, u32), span);
    dimCompany.types("PTS") = DATE;
    dimCompany.nameColumn("EffectiveDate", "PTS");

    dimCompany.insert(range(dim4(1, dimCompany.length()), 1, u64), U64, 0, "SK_CompanyID");
    dimCompany.insert(constant(1, dim4(1, dimCompany.length()), b8), BOOL, dimCompany.data().size() - 1, "IsCurrent");
    dimCompany.insert(constant(1, dim4(1, dimCompany.length()), u32), UINT, dimCompany.data().size() - 1, "BatchID");
    dimCompany.add(tile(endDate(), dim4(1, dimCompany.length())), DATE, "EndDate");

    {
        auto lowGrade = dimCompany.data(5).row(0) != 'A';
        array col = dimCompany.data(5).rows(0,2);
        lowGrade = lowGrade && anyTrue(batchFunc(col, array(3,1,"BBB").as(u8), BatchFunctions::batchNotEqual), 0);
        lowGrade.eval();
        dimCompany.insert(lowGrade, BOOL, 6, "isLowGrade");
    }

    nameDimCompany(dimCompany);
    dimCompany.data("CompanyID") = stringToNum(dimCompany.data("CompanyID"), u64);
    dimCompany.types("CompanyID") = U64;

    std::string order[3] = { "SK_CompanyID", "CompanyID", "EffectiveDate"};
    auto s0 = dimCompany.project(order, 3, "S0");
    s0.sortBy(order + 1, 2);

    auto s2 = s0.project(order + 1, 1, "S2");
    auto s1 = s0.select(range(dim4(s0.length() - 1), 0, u64) + 1);
    s2 = s2.select(range(dim4(s2.length() - 1), 0, u64));
    s1 = s1.zip(s2);
    s1 = s1.select(where(s1.data("CompanyID") == s1.data("S2.CompanyID")));
    auto end_date = s1.project(order + 2, 1, "EndDate");

    s0 = s0.project(order, 2, "S0");
    s1 = s0.project(order + 1, 1, "S1");
    s0 = s0.select(range(dim4(s0.length() - 1), 0, u64));
    s1 = s1.select(range(dim4(s1.length() - 1), 0, u64) + 1);
    s0 = s0.zip(s1);
    s0 = s0.select(where(s0.data("CompanyID") == s0.data("S1.CompanyID")));
    s0.project(order, 1, "S0");
    s0 = s0.zip(end_date);
    s0.nameColumn("EndDate", "EndDate.EffectiveDate");

    auto out = AFDataFrame::setCompare(dimCompany.data("SK_CompanyID"), s0.data("SK_CompanyID"));
    dimCompany.data("IsCurrent")(out.first) = 0;
    dimCompany.data("EndDate")(span, out.first) = (array)s0.data("EndDate")(span, out.second);

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
    auto fin1 = s_Financial.select(where(s_Financial.data("CO_NAME_OR_CIK")(0,span) == '0'));
    fin1.data("CO_NAME_OR_CIK") = stringToNum(fin1.data("CO_NAME_OR_CIK"), u64);
    fin1.types("CO_NAME_OR_CIK") = U64;

    financial = fin1.equiJoin(tmp, "CO_NAME_OR_CIK", "CompanyID");
    financial.remove("CO_NAME_OR_CIK");

    columns[1] = "Name";
    tmp = dimCompany.project(columns, 4, "DC");
    fin1 = s_Financial.select(where(s_Financial.data("CO_NAME_OR_CIK")(0,span) != '0'));

    fin1 = fin1.equiJoin(tmp, "CO_NAME_OR_CIK", "Name");
    fin1.remove("CO_NAME_OR_CIK");
    if (!fin1.data(0).isempty()) financial = financial.concatenate(fin1);
    af::sync();
    fin1.clear();
    tmp.clear();

    auto cond1 = dateHash(financial.data("DC.EffectiveDate"))
                 <= dateHash(financial.data("PTS").rows(0, 2));
    auto cond2 = dateHash(financial.data("DC.EndDate"))
                 > dateHash(financial.data("PTS").rows(0, 2));
    financial = financial.select(where(cond1 && cond2));

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
        auto security1 = s_Security.select(where(s_Security.data("CO_NAME_OR_CIK")(0,span) == '0'));
        security1.name(s_Security.name());
        security1.data("CO_NAME_OR_CIK") = stringToNum(security1.data("CO_NAME_OR_CIK"), u64);
        security1.types("CO_NAME_OR_CIK") = U64;
        security = security1.equiJoin(tmp, "CO_NAME_OR_CIK", "CompanyID");
        security.remove("CO_NAME_OR_CIK");

        columns[1] = "Name";
        tmp = dimCompany.project(columns, 4, "DC");
        security1 = s_Security.select(where(s_Security.data("CO_NAME_OR_CIK")(0,span) != '0'));
        security1.name(s_Security.name());
        security1 = security1.equiJoin(tmp, "CO_NAME_OR_CIK", "Name");
        security1.remove("CO_NAME_OR_CIK");
        security = security.concatenate(security1);
        columns[0] = "ST_ID";
        tmp = StatusType.project(columns, 1, "ST");
        security = security.equiJoin(tmp, "STATUS", "ST_ID");
    }
    print(dimCompany.length());
    print(security.length());
    auto cond1 = dateHash(security.data("DC.EffectiveDate")) <= dateHash(security.data("PTS").rows(0, 2));
    auto cond2 = dateHash(security.data("DC.EndDate")) > dateHash(security.data("PTS").rows(0, 2));
    security = security.select(where(cond1 && cond2));
    print(security.length());
    {
        std::string order[11] = {
                "SYMBOL","ISSUE_TYPE", "STATUS","NAME","EX_ID", "DC.SK_CompanyID",
                "SH_OUT", "FIRST_TRADE_DATE","FIRST_TRADE_EXCHANGE","DIVIDEND", "PTS"
        };
        security = security.project(order, 11, "DimSecurity");
    }

    security.data("PTS") = security.data("PTS")(seq(3), span);
    security.types("PTS") = DATE;
    security.nameColumn("EffectiveDate", "PTS");
    auto length = security.length();
    security.insert(range(dim4(1, length), 1, u64), U64, 0, "SK_SecurityID");
    security.insert(constant(1, dim4(1, length), b8), BOOL, security.data().size() - 1, "IsCurrent");
    security.insert(constant(1, dim4(1, length), u32), UINT, security.data().size() - 1, "BatchID");
    security.add(tile(endDate(), dim4(1, length)), DATE, "EndDate");

    nameDimSecurity(security);
    std::string order[3] = { "SK_SecurityID", "Symbol", "EffectiveDate"};
    auto s0 = security.project(order, 3, "S0");
    print("HERE");
    s0.sortBy(order + 1, 2);

    auto s2 = s0.project(order + 1, 1, "S2");
    auto s1 = s0.select(range(dim4(s0.length() - 1), 0, u64) + 1);
    s2 = s2.select(range(dim4(s2.length() - 1), 0, u64));
    s1 = s1.zip(s2);
    s1 = s1.select(where(allTrue(s1.data("Symbol") == s1.data("S2.Symbol"), 0)));
    auto end_date = s1.project(order + 2, 1, "EndDate");
    if (end_date.isEmpty()) return security;

    s0 = s0.project(order, 2, "S0");
    s1 = s0.project(order + 1, 1, "S1");
    s0 = s0.select(range(dim4(s0.length() - 1), 0, u64));
    s1 = s1.select(range(dim4(s1.length() - 1), 0, u64) + 1);
    s0 = s0.zip(s1);
    s0 = s0.select(where(allTrue(s0.data("Symbol") == s0.data("S1.Symbol"))));
    s0.project(order, 1, "S0");
    s0 = s0.zip(end_date);
    s0.nameColumn("EndDate", "EndDate.EffectiveDate");

    auto out = AFDataFrame::setCompare(security.data("SK_SecurityID"), s0.data("SK_SecurityID"));
    security.data("IsCurrent")(out.first) = 0;
    security.data("EndDate")(span, out.first) = (array)s0.data("EndDate")(span, out.second);
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
    idx = flipdims(where(out == '\n'));
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

    auto dim = prospect.data("Age").dims();
    auto batchID = 1;
    prospect.name("Prospect");

    auto tmp = tile(batchDate.data(1)(span, where(batchDate.data(0) == batchID)),dim);
    prospect.insert(dateHash(tmp), UINT, 1, "SK_RecordDateID");
    prospect.insert(array(prospect.data("SK_RecordDateID")), UINT, 2, "SK_UpdateDateID");
    prospect.insert(constant(1, dim, u32), UINT, 3, "BatchID");
    prospect.insert(constant(0, dim, u8), BOOL, 4, "IsCustomer");
    tmp = marketingNameplate(prospect.data("NetWorth"), prospect.data("Income"), prospect.data("NumberCreditCards"),
                             prospect.data("NumberChildren"), prospect.data("Age"), prospect.data("CreditRating"),
                             prospect.data("NumberCars"));
    prospect.add(tmp, STRING, "MarketingNameplate");

    return prospect;
}

AFDataFrame loadStagingTrade(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Trade.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.asU64(0), U64);
    frame.add(parser.asDateTime(1, YYYYMMDD), DATETIME);
    frame.add(parser.asString(2), STRING);
    frame.add(parser.asString(3), STRING);
    frame.add(parser.asBoolean(4), BOOL);
    frame.add(parser.asString(5), STRING);
    frame.add(parser.asUint(6), UINT);
    frame.add(parser.asDouble(7), DOUBLE);
    frame.add(parser.asUint(8), UINT);
    frame.add(parser.asU64(9), U64);
    frame.add(parser.asDouble(10), DOUBLE);
    frame.add(parser.asDouble(11), DOUBLE);
    frame.add(parser.asDouble(12), DOUBLE);
    frame.add(parser.asDouble(13), DOUBLE);
    return frame;
}

AFDataFrame loadStagingTradeHistory(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TradeHistory.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.asU64(0), U64);
    frame.add(parser.asDateTime(1, YYYYMMDD), DATE);
    frame.add(parser.asString(2), STRING);
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

inline af::array phoneNumberProcessing(af::array &ctry, af::array &area, af::array &local, af::array &ext) {
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
    idx = batchFunc(idx, flipdims(cond1) * out.dims(0) , BatchFunctions::batchAdd);
    out(idx) = flat(ctry(span, cond1));
    auto j = join(0, cond1, cond2);
    out(c + 1, j) = '(';
    idx = range(dim4(a), 0, u32) + c + 2;
    idx = batchFunc(idx, flipdims(j) * out.dims(0) , BatchFunctions::batchAdd);
    out(idx) = flat(area(span, j));
    out(2 + c + a, j) = ')';

    j = join(0, j, cond3);
    idx = range(dim4(l), 0, u32) + c + a + 3;
    idx = batchFunc(idx, flipdims(j) * out.dims(0) , BatchFunctions::batchAdd);
    out(idx) = flat(local(span, j));

    j = join(0, j, extNotNull);
    idx = range(dim4(e), 0, u32) + c + a + l + 3;
    idx = batchFunc(idx, flipdims(j) * out.dims(0) , BatchFunctions::batchAdd);
    out(idx) = flat(ext(span, j));

    out(3 + c + a + l + e, span) = '\n';

    out = out(where(out)).as(u32);
    idx = flipdims(where(out == '\n'));
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
    auto dim = frame.data(2).dims();
    AFDataFrame dimCustomer;
    dimCustomer.name("DimCustomer");
    for (auto const &i: input) dimCustomer.add(frame.data(i.first), frame.types(i.first), i.second);

    dimCustomer.insert(tile(array(dim4(7), "ACTIVE"), dim), STRING, 2, "Status");
    dimCustomer.insert(phoneNumberProcessing(frame.data(18), frame.data(19), frame.data(20), frame.data(21)),
                    STRING, 15, "Phone1");
    dimCustomer.insert(phoneNumberProcessing(frame.data(22), frame.data(23), frame.data(24), frame.data(25)),
                    STRING, 16, "Phone2");
    dimCustomer.insert(phoneNumberProcessing(frame.data(26), frame.data(27), frame.data(28), frame.data(29)),
                    STRING, 17, "Phone3");
    {
        AFDataFrame tmp;
        tmp.name("NationalTax");
        tmp.add(frame.data(31), STRING, "ID");
        tmp = tmp.equiJoin(taxRate, "ID", "TX_ID");
        dimCustomer.add(tmp.data("TaxRate.TX_NAME"), STRING, "NationalTaxRateDesc");
        dimCustomer.add(tmp.data("TaxRate.TX_RATE"), FLOAT, "NationalTaxRate");
    }

    {
        AFDataFrame tmp;
        tmp.name("LocalTax");
        tmp.add(frame.data(30), STRING, "ID");
        tmp = tmp.equiJoin(taxRate, "ID", "TX_ID");
        dimCustomer.add(tmp.data("TaxRate.TX_NAME"), STRING, "LocalTaxRateDesc");
        dimCustomer.add(tmp.data("TaxRate.TX_RATE"), FLOAT, "LocalTaxRate");
    }

    return dimCustomer;
}



#include <utility>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <rapidxml.hpp>
#include "TPC_DI.h"
#include "FinwireParser.h"
#include "XMLFlattener.h"
#include "BatchFunctions.h"

namespace fs = boost::filesystem;
namespace xml = rapidxml;
using namespace af;

Finwire::Finwire(Finwire&& other) noexcept : company(nullptr), financial(nullptr), security(nullptr) {
    company = std::move(other.company);
    financial = std::move(other.financial);
    security = std::move(other.security);
    other.company = nullptr;
    other.financial = nullptr;
    other.security = nullptr;
}

/* Independent Static Tables */
AFDataFrame loadDimDate(char const *directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Date.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    frame.add(parser.asU64(0), U64);
    frame.add(parser.asDate(1, YYYYMMDD, false), DATE);
    for (int i = 2;  i < 17; i += 2) {
        frame.add(parser.asString(i), STRING);
        frame.add(parser.asUint(i + 1), UINT);
    }

    frame.add(parser.stringToBoolean(17), BOOL);
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

    frame.add(parser.stringToBoolean(8), BOOL);
    frame.add(parser.stringToBoolean(9), BOOL);
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

    for (int i = 0;  i < 2; ++i) frame.add(parser.asString(i), STRING);

    frame.add(parser.asFloat(2), FLOAT);
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
    for( fs::directory_iterator i( directory ); i != end_itr; ++i )
    {
        auto n = i->path().string().find("_audit.csv", dir.size());
        if( n == std::string::npos) continue;
        auditFiles.push_back( i->path().string() );
    }
    std::sort(auditFiles.begin(), auditFiles.end());

    for (auto const &path : auditFiles) {
        AFParser parser(path.c_str(), ',', true);
        if (frame.isEmpty()) {
            frame.add(parser.asString(0), STRING);
            frame.add(parser.asUchar(1), UINT);
            frame.add(parser.asDate(2, YYYYMMDD, false), DATE);
            frame.add(parser.asString(3), STRING);
            frame.add(parser.asInt(4), LONG);
            frame.add(parser.asFloat(5), DOUBLE);
        } else {
            AFDataFrame tmp;
            tmp.add(parser.asString(0), STRING);
            tmp.add(parser.asUchar(1), UINT);
            tmp.add(parser.asDate(2, YYYYMMDD, false), DATE);
            tmp.add(parser.asString(3), STRING);
            tmp.add(parser.asInt(4), LONG);
            tmp.add(parser.asFloat(5), DOUBLE);
            frame.concatenate(tmp);
        }
    }
    return frame;
}

void nameStagingCompany(AFDataFrame &company) {
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

void nameStagingFinancial(AFDataFrame &financial) {
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

void nameStagingSecurity(AFDataFrame &security) {
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

/* Staging Tables */
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
    AFDF_ptr stagingCompany = nullptr;
    AFDF_ptr stagingFinancial = nullptr;
    AFDF_ptr stagingSecurity = nullptr;
    for (auto const &path : finwireFiles) {
        FinwireParser finwire(path.c_str());
        if (!stagingCompany && !stagingFinancial && !stagingSecurity) {
            stagingCompany = std::make_shared<AFDataFrame>(finwire.extractCmp());
            stagingFinancial = std::make_shared<AFDataFrame>(finwire.extractFin());
            stagingSecurity = std::make_shared<AFDataFrame>(finwire.extractSec());
        } else {
            stagingCompany->concatenate(finwire.extractCmp());
            stagingFinancial->concatenate(finwire.extractFin());
            stagingSecurity->concatenate(finwire.extractSec());
        }
    }
    nameStagingCompany(*stagingCompany);
    nameStagingFinancial(*stagingFinancial);
    nameStagingSecurity(*stagingSecurity);
    return Finwire(stagingCompany, stagingFinancial, stagingSecurity);
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
    std::string data = flattenCustomerMgmt(directory);
    AFDataFrame frame;
    AFParser parser(data, '|', false);
    for (int i = 0; i < 2; ++i) frame.add(parser.asString(i), STRING);

    frame.add(parser.asU64(2), U64);

    for (int i = 3; i < 5; ++i) frame.add(parser.asString(i), STRING);

    frame.add(parser.asUchar(5), UCHAR);
    frame.add(parser.asDate(6, YYYYMMDD, false), DATE);

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
        AFParser parser(directory, ',');
        frame.add(parser.asUint(0), UINT);
        frame.add(parser.asUint(1), UINT);
        for (int i = 2;  i <= 7; ++i) {
            frame.add(parser.asString(i), STRING);
        }
    }
    frame.stringMatchSelect(5, "314");
    frame.remove(5);
    auto length = frame.data()[0].dims(1);
    frame.insert(range(dim4(1, length), 1, u64), U64, 0);
    frame.add(constant(1, dim4(1, length), b8), BOOL);
    frame.add(constant(1, dim4(1, length), u32), UINT);
    auto date = sort(dimDate.data(1),1);
    date = date(0);
    frame.add(tile(date, dim4(1, length)), DATE);
    frame.add(tile(AFDataFrame::endDate(), dim4(1, length)), DATE);
    return frame;
}

AFDataFrame loadStagingCashBalances(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "CashTransaction.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.asU64(0), U64);
    frame.add(parser.asDateTime(1, YYYYMMDD), DATE);
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
    frame.add(parser.asDateTime(2, YYYYMMDD), DATE);
    frame.add(parser.asString(3), STRING);
    return frame;
}

void nameDimCompany(AFDataFrame &dimCompany) {
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

    dimCompany.data("PTS") = dimCompany.data("PTS")(range(dim4(3), 0, u32), span);
    dimCompany.types("PTS") = DATE;
    dimCompany.nameColumn("EffectiveDate", "PTS");

    auto length = dimCompany.data(0).dims(1);
    dimCompany.insert(range(dim4(1, length), 1, u64), U64, 0);
    dimCompany.nameColumn("SK_CompanyID", 0);
    dimCompany.insert(constant(1, dim4(1, length), b8), BOOL, dimCompany.data().size() - 1);
    dimCompany.nameColumn("IsCurrent", dimCompany.data().size() - 2);
    dimCompany.insert(constant(1, dim4(1, length), u32), UINT, dimCompany.data().size() - 1);
    dimCompany.nameColumn("BatchID", dimCompany.data().size() - 2);
    dimCompany.add(tile(AFDataFrame::endDate(), dim4(1, length)), DATE);
    dimCompany.nameColumn("EndDate", dimCompany.data().size() - 1);
    {
        auto lowGrade = dimCompany.data(5).row(0) != 'A';
        array col = dimCompany.data(5).rows(0,2);
        lowGrade = lowGrade && anyTrue(batchFunc(col, array(3,1,"BBB").as(u8), BatchFunctions::batchNotEqual), 0);
        lowGrade.eval();
        dimCompany.insert(lowGrade, BOOL, 6);
        dimCompany.nameColumn("isLowGrade", 6);
    }
    nameDimCompany(dimCompany);
    dimCompany.data("CompanyID") = FinwireParser::stringToNum(dimCompany.data("CompanyID"), u64);
    dimCompany.types("CompanyID") = U64;

    AFDataFrame s1;
    {
        std::string order[3] = { "SK_CompanyID", "CompanyID", "EffectiveDate"};
        s1 = dimCompany.project(order, 3, "S1");
    }
    {
        auto const& l = s1.data("CompanyID");
        auto idx = range(dim4(1, l.dims(1)), 1, u64);
        array sorting;
        sort(sorting,idx,l, 1); // first column
        s1.select(idx);
        auto date = AFDataFrame::dateOrTimeHash(s1.data("EffectiveDate"));
        {
            auto eq = sum(batchFunc(l, setUnique(l, true), BatchFunctions::batchEqual),1);
            eq = moddims(eq, dim4(eq.dims(1), eq.dims(0)));
            eq = join(1, constant(0,dim4(1),u32), eq);
            eq = accum(eq, 1);
            idx = join(0, eq.cols(0, end - 1), eq.cols(1, end) - 1);
        }

        auto h = max(diff1(idx,0)).scalar<uint32_t>() + 1;
        auto tmp2 = batchFunc(idx(0,span), range(dim4(h, idx.dims(1)), 0, u32), BatchFunctions::batchAdd);
        auto tmp3 = tmp2;
        tmp3(where(batchFunc(tmp3,idx(1,span), BatchFunctions::batchGreater))) = UINT32_MAX;
        {
            auto tmp4 = where(tmp3!=UINT32_MAX);
            array tmp5 = tmp3(tmp4);
            tmp3(tmp4) = reorder(date(tmp5),1,0);
        }
        sort(tmp3,idx,tmp3, 0);
        idx += range(idx.dims(), 1, u32) * idx.dims(0);
        idx = idx(where(tmp3!=UINT32_MAX));
        idx = tmp2(idx);
        s1.select(idx);
    }

    s1.add(range(dim4(1, length), 1, u64), U64);
    s1.nameColumn("RN", s1.data().size() - 1);
    std::string order[2] = {"RN", "CompanyID"};
    {
        auto s2 = s1.project(order,2,"S2");
        s2.data("RN") = s2.data("RN") - 1;
        s1 = s1.equiJoin(s2, "RN", "RN");
    }
    {
        auto idx = where(s1.data("CompanyID") == s1.data("S2.CompanyID"));
        s1.select(idx);
    }
    order[0] = "SK_CompanyID";
    order[1] = "EffectiveDate";
    s1 = s1.project(order, 2, "candidate");
    auto out = AFDataFrame::innerJoin(dimCompany.data("SK_CompanyID"), s1.data("SK_CompanyID"));
    dimCompany.data("IsCurrent")(out.first) = 0;
    dimCompany.data("EndDate")(span, out.first) = s1.data("EffectiveDate");

    return dimCompany;
}

void nameFinancial(AFDataFrame &financial) {
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
    using namespace af;
    using namespace BatchFunctions;

    AFDataFrame financial;
    {
        std::string columns[4] = {"SK_CompanyID", "CompanyID", "EffectiveDate", "EndDate"};
        auto tmp = dimCompany.project(columns, 4, "DimCompany");
        financial = s_Financial.equiJoin(tmp, "CO_NAME_OR_CIK", "CompanyID");
        columns[1] = "Name";
        tmp = dimCompany.project(columns, 4, "DimCompany");
        financial.concatenate(s_Financial.equiJoin(tmp, "CO_NAME_OR_CIK", "Name"));
    }

    auto cond1 = AFDataFrame::dateOrTimeHash(financial.data("DimCompany.EffectiveDate"))
                 <= AFDataFrame::dateOrTimeHash(financial.data("PTS").rows(0,2));
    auto cond2 = AFDataFrame::dateOrTimeHash(financial.data("DimCompany.EndDate"))
                 > AFDataFrame::dateOrTimeHash(financial.data("PTS").rows(0,2));
    financial.select(where(cond1 && cond2));

    std::string order[14] = {
            "DimCompany.SK_CompanyID", "YEAR", "QUARTER", "QTR_START_DATE","REVENUE", "EARNINGS",
            "EPS", "DILUTED_EPS","MARGIN","INVENTORY","ASSETS","LIABILITIES","SH_OUT", "DILUTED_SH_OUT"
    };

    financial = financial.project(order, 14, "Financial");
    print(financial.data().size());

    // Renames columns
    nameFinancial(financial);

    return financial;
}

void nameDimSecurity(AFDataFrame &dimSecurity) {
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
        security = s_Security.equiJoin(tmp, "CO_NAME_OR_CIK", "CompanyID");
        columns[1] = "Name";
        tmp = dimCompany.project(columns, 4, "DC");
        security.concatenate(s_Security.equiJoin(tmp, "CO_NAME_OR_CIK", "Name"));
        columns[0] = "ST_ID";
        tmp = StatusType.project(columns, 1, "ST");
        security = security.equiJoin(tmp, "STATUS", "ST_ID");
    }

    auto cond1 = AFDataFrame::dateOrTimeHash(security.data("DC.EffectiveDate"))
                 <= AFDataFrame::dateOrTimeHash(security.data("PTS").rows(0,2));
    auto cond2 = AFDataFrame::dateOrTimeHash(security.data("DC.EndDate"))
                 > AFDataFrame::dateOrTimeHash(security.data("PTS").rows(0,2));
    security.select(where(cond1 && cond2));

    std::string order[11] = {
            "SYMBOL","ISSUE_TYPE", "STATUS","NAME","EX_ID", "DC.SK_CompanyID",
            "SH_OUT", "FIRST_TRADE_DATE","FIRST_TRADE_EXCHANGE","DIVIDEND", "PTS"
    };

    security = security.project(order, 11, "DimSecurity");

    security.data("PTS") = security.data("PTS")(range(dim4(3), 0, u32), span);
    security.types("PTS") = DATE;
    security.nameColumn("EffectiveDate", "PTS");
    auto length = security.data(0).dims(1);
    security.insert(range(dim4(1, length), 1, u64), U64, 0);
    security.nameColumn("SK_SecurityID", 0);
    security.insert(constant(1, dim4(1, length), b8), BOOL, security.data().size() - 1);
    security.nameColumn("IsCurrent", security.data().size() - 2);
    security.insert(constant(1, dim4(1, length), u32), UINT, security.data().size() - 1);
    security.nameColumn("BatchID", security.data().size() - 2);
    security.add(tile(AFDataFrame::endDate(), dim4(1, length)), DATE);
    security.nameColumn("EndDate", security.data().size() - 1);

    nameDimSecurity(security);
    //TODO do scd
    return security;
}


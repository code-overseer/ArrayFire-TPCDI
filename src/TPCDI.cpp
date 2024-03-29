#include <utility>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <rapidxml.hpp>
#include "TPCDI.h"
#include "BatchFunctions.h"
#include "Logger.h"
#include "ColumnNames.h"
#ifdef ITT_ENABLED
    #include <ittnotify.h>
#endif
namespace fs = boost::filesystem;
namespace xml = rapidxml;
using namespace af;
using namespace Utils;
using namespace BatchFunctions;

std::vector<std::string> inline collectFinwireFiles(char const *directory) {
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
    return finwireFiles;
}

AFDataFrame loadBatchDate(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "BatchDate.txt");
    AFParser parser(file, '|', false);

    AFDataFrame frame;
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
    Logger::startTimer("DimDate");
    AFParser parser(file, '|', false);

    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asDate(1, true, YYYYMMDD));
    for (int i = 2;  i < 17; i += 2) {
        frame.add(parser.parse<char*>(i));
        frame.add(parser.parse<unsigned int>(i + 1));
    }
    frame.add(parser.parse<bool>(17));
    Logger::logTime("DimDate", false);
    callGC();
    return frame;
}

AFDataFrame loadDimTime(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Time.txt");
    AFDataFrame frame;
    Logger::startTimer("DimTime");
    AFParser parser(file, '|', false);
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asTime(1, true));

    for (int i = 2;  i < 7; i += 2) {
        frame.add(parser.parse<unsigned int>(i));
        frame.add(parser.parse<char*>(i + 1));
    }
    frame.add(parser.parse<bool>(8));
    frame.add(parser.parse<bool>(9));
    Logger::logTime("DimTime", false);
    callGC();
    return frame;
}

AFDataFrame loadIndustry(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Industry.txt");

    Logger::startTimer("Industry");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    for (int i = 0;  i < 3; ++i) frame.add(parser.parse<char*>(i));
    Logger::logTime("Industry", false);

    frame.name("Industry");
    frame.nameColumn("IN_ID", 0);
    frame.nameColumn("IN_NAME", 1);
    frame.nameColumn("IN_SC_ID", 2);
    callGC();
    return frame;
}

AFDataFrame loadStatusType(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "StatusType.txt");

    Logger::startTimer("StatusType");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    frame.add(parser.parse<char*>(0), "ST_ID");
    frame.add(parser.parse<char*>(1), "ST_NAME");
    Logger::logTime("StatusType", false);
    frame.name("StatusType");
    callGC();
    return frame;
}

AFDataFrame loadTaxRate(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TaxRate.txt");

    Logger::startTimer("TaxRate");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    frame.name("TaxRate");
    frame.add(parser.parse<char*>(0), "TX_ID");
    frame.add(parser.parse<char*>(1), "TX_NAME");
    frame.add(parser.parse<float>(2), "TX_RATE");
    Logger::logTime("TaxRate", false);
    callGC();
    return frame;
}

AFDataFrame loadTradeType(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TradeType.txt");

    Logger::startTimer("TradeType");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    for (int i = 0;  i < 2; ++i) frame.add(parser.parse<char*>(i));
    for (int i = 2;  i < 4; ++i) frame.add(parser.parse<unsigned int>(i));

    Logger::logTime("TradeType", false);
    callGC();
    return frame;
}

AFDataFrame loadStagingMarket(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "DailyMarket.txt");
    AFDataFrame frame;
    // Logger::startCollection();
    Logger::startTimer("DailyMarket");
    AFParser parser(file, '|', false);

    af::deviceGC();
    frame.add(parser.parse<char*>(0), "DM_DATE");
    frame(0).toDate(true);
    af::deviceGC();
    frame.add(parser.parse<char*>(1), "DM_S_SYMB");
    frame.add(parser.parse<float>(2), "DM_CLOSE");
    frame.add(parser.parse<float>(3), "DM_HIGH");
    af::deviceGC();
    frame.add(parser.parse<float>(4), "DM_LOW");
    frame.add(parser.parse<unsigned long long>(5), "DM_VOL");
    Logger::logTime("DailyMarket", false);
    // Logger::pauseCollection();
    callGC();
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

    // Logger::startCollection();
    // Logger::startTask("Audit Load");
    Logger::startTimer("Audit");
    AFParser parser(auditFiles, ',', true);
    // Logger::endLastTask();

    // Logger::startTask("Audit Parse");
    frame.add(parser.parse<char*>(0));
    frame.add(parser.parse<unsigned int>(1));
    frame.add(parser.asDate(2, true, YYYYMMDD));
    frame.add(parser.parse<char*>(3));
    frame.add(parser.parse<int>(4));
    frame.add(parser.parse<double>(5));
    Logger::logTime("Audit", false);
    // Logger::endLastTask();

    // Logger::pauseCollection();
    callGC();
    return frame;
}

AFDataFrame loadStagingSecurity(char const *directory) {
    std::vector<std::string> finwireFiles = collectFinwireFiles(directory);
    // Logger::startCollection();
    Logger::startTimer("StagingSecurity");
    FinwireParser parser(finwireFiles);
    auto sec = parser.extractSec();
    Logger::logTime("StagingSecurity", false);
    // Logger::pauseCollection();
    nameStagingSecurity(sec);
    return sec;
}

AFDataFrame loadStagingCompany(char const *directory) {
    std::vector<std::string> finwireFiles = collectFinwireFiles(directory);
    // Logger::startCollection();
    Logger::startTimer("StagingCompany");
    FinwireParser parser(finwireFiles);
    auto cmp = parser.extractCmp();
    Logger::logTime("StagingCompany", false);
    // Logger::pauseCollection();
    nameStagingCompany(cmp);
    return cmp;
}

AFDataFrame loadStagingFinancial(char const *directory) {
    std::vector<std::string> finwireFiles = collectFinwireFiles(directory);
    // Logger::startCollection();
    Logger::startTimer("StagingFinancial");
    FinwireParser parser(finwireFiles);
    auto fin = parser.extractFin();
    Logger::logTime("StagingFinancial", false);
    // Logger::pauseCollection();
    nameStagingFinancial(fin);
    return fin;
}

Finwire loadStagingFinwire(char const *directory) {
    std::vector<std::string> finwireFiles = collectFinwireFiles(directory);
    //    // Logger::startCollection();
    Logger::startTimer("Finwire Ingestion");
    Finwire finwire = FinwireParser(finwireFiles).extractData();
    Logger::logTime("Finwire Ingestion", false);

    Logger::startTimer("StagingCompany");
    nameStagingCompany(finwire.company);
    Logger::logTime("StagingCompany", false);

    Logger::startTimer("StagingFinancial");
    nameStagingFinancial(finwire.financial);
    Logger::logTime("StagingFinancial", false);

    Logger::startTimer("StagingSecurity");
    nameStagingSecurity(finwire.security);
    Logger::logTime("StagingSecurity", false);
    // // Logger::pauseCollection();
    callGC();
    return finwire;
}

AFDataFrame loadStagingProspect(char const *directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Prospect.csv");
    Logger::startTimer("StagingProspect");
    AFDataFrame frame;
    // Logger::startCollection();

    // Logger::startTask("Staging Prospect Load");
    AFParser parser(file, ',', false);
    // Logger::endLastTask();

    // Logger::startTask("Staging Prospect Parse");
    for (int i = 0;  i < 12; ++i) frame.add(parser.parse<char*>(i));

    frame.add(parser.parse<unsigned long long>(12));

    for (int i = 13;  i < 15; ++i) frame.add(parser.parse<unsigned char>(i));

    frame.add(parser.parse<char*>(15));
    frame.add(parser.parse<unsigned short>(16));
    frame.add(parser.parse<unsigned int>(17));

    for (int i = 18;  i < 20; ++i) frame.add(parser.parse<char*>(i));

    frame.add(parser.parse<unsigned char>(20));
    frame.add(parser.parse<unsigned long long>(21));
    // Logger::endLastTask();
    Logger::logTime("StagingProspect", false);
    // Logger::pauseCollection();
    nameStagingProspect(frame);
    callGC();
    return frame;
}

AFDataFrame loadStagingCustomer(char const* directory) {
    AFDataFrame frame;
    // Logger::startCollection();

    // Logger::startTask("Flattening XML");

    std::string data = Utils::flattenCustomerMgmt(directory);
    // Logger::endLastTask();

    // Logger::startTask("Customer Load");
    Logger::startTimer("StagingCustomer");
    AFParser parser(data, '|', false);
    // Logger::endLastTask();

    // Logger::startTask("Customer Parse");
    frame.add(parser.parse<char*>(0));
    frame.add(parser.asDateTime(1, YYYYMMDD));
    frame.add(parser.parse<unsigned long long>(2));

    for (int i = 3; i < 5; ++i) frame.add(parser.parse<char*>(i));

    frame.add(parser.parse<unsigned char>(5));
    frame.add(parser.asDate(6, true, YYYYMMDD));
    callGC();
    for (int i = 7; i < 32; ++i) frame.add(parser.parse<char*>(i));
    callGC();
    frame.add(parser.parse<unsigned long long>(32));
    frame.add(parser.parse<unsigned short>(33));
    frame.add(parser.parse<unsigned long long>(34));
    frame.add(parser.parse<char*>(35));
    Logger::logTime("StagingProspect", false);
    // Logger::endLastTask();

    callGC();
    nameStagingProspect(frame);
    return frame;
}

AFDataFrame loadDimBroker(char const* directory, AFDataFrame& dimDate) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "HR.csv");
    AFDataFrame dimBroker;
    // Logger::startCollection();

    // Logger::startTask("HR Load");
    Logger::startTimer("StagingBroker");
    AFParser parser(file, ',');
    // Logger::endLastTask();

    // Logger::startTask("HR Parse");
    dimBroker.add(parser.parse<unsigned int>(0));
    dimBroker.add(parser.parse<unsigned int>(1));
    for (int i = 2;  i <= 7; ++i) dimBroker.add(parser.parse<char*>(i));
    // Logger::endLastTask();
    Logger::logTime("StagingBroker", false);

    Logger::startTimer("DimBroker");
    // Logger::startTask("Broker Select");
    dimBroker = dimBroker.select(dimBroker(5) == "314", "DimBroker");
    // Logger::endLastTask();

    dimBroker.remove(5);
    auto length = dimBroker.rows();
    dimBroker.insert(Column(range(dim4(1, length), 1, u64)), 0);
    dimBroker.add(Column(constant(1, dim4(1, length), b8)));
    dimBroker.add(Column(constant(1, dim4(1, length), u32)));
    dimDate.sortBy(0);
    auto date = dimDate(1)(af::span, 0);
    dimBroker.add(Column(tile(date, dim4(1, length)), DATE));
    dimBroker.add(Utils::endDate(length));
    Logger::logTime("DimBroker", false);
    // Logger::pauseCollection();
    callGC();
    return dimBroker;
}

AFDataFrame loadStagingCashBalances(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "CashTransaction.txt");
    AFDataFrame frame;
    // Logger::startCollection();

    // Logger::startTask("Cash Load");
    Logger::startTimer("StagingCashBalances");
    AFParser parser(file, '|', false);
    // Logger::endLastTask();
    
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asDateTime(1, YYYYMMDD));
    frame.add(parser.parse<double>(2));
    frame.add(parser.parse<char*>(3));
    Logger::logTime("StagingCashBalances", false);

    // Logger::pauseCollection();
    callGC();
    return frame;
}

AFDataFrame loadStagingWatches(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "WatchHistory.txt");
    // Logger::startCollection();
    Logger::startTimer("StagingWatches");
    AFDataFrame frame;
    AFParser parser(file, '|', false);

    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.parse<char*>(1));
    frame.add(parser.asDateTime(2, YYYYMMDD));
    frame.add(parser.parse<char*>(3));
    Logger::logTime("StagingWatches", false);

    // Logger::pauseCollection();
    callGC();
    return frame;
}

AFDataFrame loadDimCompany(AFDataFrame &&s_Company, AFDataFrame &industry, AFDataFrame &statusType) {
    // Logger::startCollection();
    Logger::startTimer("DimCompany");
    // Logger::startTask("DimCompany Industry Join");
    auto dimCompany = s_Company.equiJoin(industry,5,0);
    // Logger::endLastTask();

    // Logger::startTask("DimCompany Status Join");
    dimCompany = dimCompany.equiJoin(statusType,4,0);
    // Logger::endLastTask();

    dimCompany = dimCompany.project(
            { "CIK","StatusType.ST_NAME","COMPANY_NAME", "Industry.IN_NAME","SP_RATING",
              "CEO_NAME", "ADDR_LINE_1","ADDR_LINE_2", "POSTAL_CODE","CITY","STATE_PROVINCE",
              "COUNTRY","DESCRIPTION","FOUNDING_DATE", "PTS"
            }, "DimCompany");
    dimCompany("PTS").toDate();
    dimCompany.nameColumn("EffectiveDate", "PTS");
    dimCompany.insert(Column(range(dim4(1, dimCompany.rows()), 1, u64)), 0, "SK_CompanyID");
    dimCompany.insert(Column(constant(1, dim4(1, dimCompany.rows()), b8)), dimCompany.columns() - 1, "IsCurrent");
    dimCompany.insert(Column(constant(1, dim4(1, dimCompany.rows()), u32)), dimCompany.columns() - 1, "BatchID");
    dimCompany.add(Utils::endDate(dimCompany.rows()), "EndDate");

    // Logger::startTask("DimCompany LowGrade Addition");
    dimCompany.insert(Column(dimCompany("SP_RATING").left(1) != "A" && dimCompany("SP_RATING").left(3) != "BBB", BOOL), 6, "isLowGrade");
    // Logger::endLastTask();
    nameDimCompany(dimCompany);
    Logger::logTime("DimCompany", false);

    Logger::startTimer("DimCompany SCD");
    // Logger::startTask("DimCompany Update EndDate");
    auto s0 = dimCompany.project({ "SK_CompanyID", "CompanyID", "EffectiveDate"}, "S0");
    s0.sortBy({ "CompanyID", "EffectiveDate" });

    auto s2 = s0.project({ "CompanyID" }, "S2");
    auto s1 = s0.select(range(dim4(s0.rows() - 1), 0, u64) + 1);
    s2 = s2.select(range(dim4(s2.rows() - 1), 0, u64), "S2");
    s1 = s1.zip(s2);
    s1 = s1.select(s1("CompanyID") == s1("S2.CompanyID"), "S1");
    auto end_date = s1.project({ "EffectiveDate" }, "EndDate");

    s0 = s0.project({ "SK_CompanyID", "CompanyID" }, "S0");
    s1 = s0.project({ "CompanyID" }, "S1");
    s0 = s0.select(range(dim4(s0.rows() - 1), 0, u64), "S0");
    s1 = s1.select(range(dim4(s1.rows() - 1), 0, u64) + 1, "S1");
    s0 = s0.zip(s1);
    s0 = s0.select(s0("CompanyID") == s0("S1.CompanyID"), "S0");
    s0 = s0.project({ "SK_CompanyID" }, "S0");

    s0 = s0.zip(end_date);
    s0.nameColumn("EndDate", "EndDate.EffectiveDate");
    // Logger::startTask("DimCompany EndDate ID");
    auto out = AFDataFrame::hashCompare(dimCompany("SK_CompanyID").data(), s0("SK_CompanyID").data());
    dimCompany("IsCurrent")(out.first) = 0;
    dimCompany("EndDate")(span, out.first) = (array) s0("EndDate")(span, out.second);
    Logger::logTime("DimCompany SCD", false);
    // Logger::endLastTask();
    // Logger::endLastTask();

    // Logger::pauseCollection();
    callGC();
    return dimCompany;
}

AFDataFrame loadFinancial(AFDataFrame &&s_Financial, AFDataFrame const &dimCompany) {
    AFDataFrame financial;
    // Logger::startCollection();
    Logger::startTimer("Financial");
    // Logger::startTask("Financial CIK Trim");
    auto cik = s_Financial("CO_NAME_OR_CIK").left(1) == "0";
    // Logger::endLastTask();
    // Logger::startTask("Financial CIK Select And Cast");
    auto fin1 = s_Financial.select(cik);
    fin1("CO_NAME_OR_CIK").cast<unsigned long long>();
    // Logger::endLastTask();
    // Logger::startTask("Financial CIK Join");
    financial = fin1.equiJoin(
            dimCompany.project({"SK_CompanyID", "CompanyID", "EffectiveDate", "EndDate"}, "DC"),
            "CO_NAME_OR_CIK", "CompanyID");
    // Logger::endLastTask();
    financial.remove("CO_NAME_OR_CIK");
    financial.remove("DC.CompanyID");

    // Logger::startTask("Financial Name Select");
    fin1 = s_Financial.select(!cik);
    // Logger::endLastTask();

    // Logger::startTask("Financial Name Join");
    fin1 = fin1.equiJoin(
            dimCompany.project({"SK_CompanyID", "Name", "EffectiveDate", "EndDate"}, "DC"), "CO_NAME_OR_CIK", "Name");
    // Logger::endLastTask();
    fin1.remove("CO_NAME_OR_CIK");
    fin1.remove("DC.Name");

    if (!fin1.isEmpty()) financial = financial.unionize(std::move(fin1));
    af::sync();

    financial("PTS").toDate();
    
    financial = financial.select(
            financial("DC.EffectiveDate") <= financial("PTS") &&
            financial("DC.EndDate") > financial("PTS")
            );

    financial = financial.project({ "DC.SK_CompanyID", "YEAR", "QUARTER", "QTR_START_DATE","REVENUE", "EARNINGS",
                                   "EPS", "DILUTED_EPS","MARGIN","INVENTORY","ASSETS","LIABILITIES","SH_OUT",
                                   "DILUTED_SH_OUT" }, "Financial");
    // Logger::pauseCollection();
    // Renames columns
    Logger::logTime("Financial", false);
    nameFinancial(financial);

    callGC();
    return financial;
}

AFDataFrame loadDimSecurity(AFDataFrame &&s_Security, AFDataFrame &dimCompany, AFDataFrame &StatusType) {
    AFDataFrame security;
    // Logger::startCollection();
    Logger::startTimer("DimSecurity");
    {
        // Logger::startTask("Financial CIK Trim");
        auto cik = s_Security("CO_NAME_OR_CIK").left(1) == "0";
        // Logger::endLastTask();

        // Logger::startTask("DimSecurity CIK Join");
        auto part1 = s_Security.select(cik);
        part1("CO_NAME_OR_CIK").cast<unsigned long long>();
        part1 = part1.equiJoin(
                dimCompany.project({"SK_CompanyID", "CompanyID", "EffectiveDate", "EndDate"}, "DC"),
                "CO_NAME_OR_CIK",
                "CompanyID");
        part1("PTS").toDate();
        part1.nameColumn("EffectiveDate", "PTS");
        part1 = part1.select(
                part1("DC.EffectiveDate") <= part1("EffectiveDate") &&
                part1("DC.EndDate") > part1("EffectiveDate"));
        part1.remove("CO_NAME_OR_CIK");
        part1.remove("DC.CompanyID");
        // Logger::endLastTask();
        // Logger::startTask("DimSecurity Name Join");
        auto part2 = s_Security.select(!cik);
        part2 = part2.equiJoin(
                dimCompany.project({"SK_CompanyID", "Name", "EffectiveDate", "EndDate"}, "DC"),
                "CO_NAME_OR_CIK",
                "Name");
        part2.remove("CO_NAME_OR_CIK");
        part2.remove("DC.Name");

        part2("PTS").toDate();
        part2.nameColumn("EffectiveDate", "PTS");
        part2 = part2.select(
                part2("DC.EffectiveDate") <= part2("EffectiveDate") &&
                part2("DC.EndDate") > part2("EffectiveDate"));
        
        security = part1.unionize(part2);

        // Logger::endLastTask();
    }
    // Logger::startTask("DimSecurity Status Join");

    security = security.equiJoin(StatusType, "STATUS", "ST_ID").project(
            {"SYMBOL","ISSUE_TYPE", "StatusType.ST_NAME" ,"NAME","EX_ID", "DC.SK_CompanyID",
             "SH_OUT", "FIRST_TRADE_DATE","FIRST_TRADE_EXCHANGE","DIVIDEND", "EffectiveDate" }, "DimSecurity");
    // Logger::endLastTask();

    auto length = security.rows();
    security.insert(Column(range(dim4(1, length), 1, u64)), 0, "SK_SecurityID");
    security.insert(Column(constant(1, dim4(1, length), b8)), security.columns() - 1, "IsCurrent");
    security.insert(Column(constant(1, dim4(1, length), u32)), security.columns() - 1, "BatchID");
    security.add(Utils::endDate(length), "EndDate");

    Logger::logTime("DimSecurity", false);
    nameDimSecurity(security);
    Logger::startTimer("DimSecurity SCD");
    // Logger::startTask("DimSecurity Update EndDate");
    auto s0 = security.project({ "SK_SecurityID", "Symbol", "EffectiveDate"}, "S0");
    s0.sortBy({"Symbol", "EffectiveDate"});

    auto s2 = s0.project({"Symbol"}, "S2");
    auto s1 = s0.select(range(dim4(s0.rows() - 1), 0, u64) + 1);
    s2 = s2.select(range(dim4(s2.rows() - 1), 0, u64));
    s1 = s1.zip(std::move(s2));

    s1 = s1.select(s1("Symbol") == s1("S2.Symbol"));
    auto end_date = s1.project({ "EffectiveDate" }, "EndDate");
    if (end_date.isEmpty()) return security;

    s0 = s0.project({ "SK_SecurityID", "Symbol" }, "S0");
    s1 = s0.project({"Symbol"}, "S1");
    s0 = s0.select(range(dim4(s0.rows() - 1), 0, u64));
    s1 = s1.select(range(dim4(s1.rows() - 1), 0, u64) + 1);
    s0 = s0.zip(std::move(s1));

    s0 = s0.select(s0("Symbol") == s0("S1.Symbol"));
    s0.project({ "SK_SecurityID" }, "S0");
    s0 = s0.zip(std::move(end_date));

    s0.nameColumn("EndDate", "EndDate.EffectiveDate");
    // Logger::startTask("DimSecurity EndDate ID");
    auto out = AFDataFrame::hashCompare(security("SK_SecurityID"), s0("SK_SecurityID"));
    security("IsCurrent")(out.first) = 0;
    security("EndDate")(span, out.first) = (array) s0("EndDate")(span, out.second);
    // Logger::endLastTask();
    // Logger::endLastTask();
    Logger::logTime("DimSecurity SCD", false);
    // Logger::pauseCollection();
    callGC();
    return security;
}

Column inline marketingNameplate(array const &networth, array const &income, array const &cards, array const &children,
                         array const &age, array const &credit, array const &cars) {
    array val(12, 6, "+HighValue\0\0+Expenses\0\0\0+Boomer\0\0\0\0\0+MoneyAlert\0+Spender\0\0\0\0+Inherited\0\0");
    val = val.as(u8);
    // Logger::startCollection();
    // Logger::startTask("Marketing Nameplate Column");
    auto idx = constant(0, 6, networth.dims(1), b8);
    idx(0, networth > 1000000 || income > 200000) = 1;
    idx(1, cards > 5 || children > 3) = 1;
    idx(2, age > 45) = 1;
    idx(3, credit < 600 || income < 5000 || networth < 100000) = 1;
    idx(4, cars > 3 || cards > 7) = 1;
    idx(5, age > 25 || networth > 1000000) = 1;

    auto out = constant(0, 56, networth.dims(1), u8);

    auto j = where(idx.row(0));
    out(seq(0,9), j) = tile(val(seq(10), 0), dim4(1, j.dims(0)));
    j = where(idx.row(1));
    out(seq(10,18), j) = tile(val(seq(9), 1), dim4(1, j.dims(0)));
    j = where(idx.row(2));
    out(seq(19,25), j) = tile(val(seq(7), 2), dim4(1, j.dims(0)));
    j = where(idx.row(3));
    out(seq(26,36), j) = tile(val(seq(11), 3), dim4(1, j.dims(0)));
    j = where(idx.row(4));
    out(seq(37,44), j) = tile(val(seq(8), 4), dim4(1, j.dims(0)));
    j = where(idx.row(5));
    out(seq(45,54), j) = tile(val(seq(10), 5), dim4(1, j.dims(0)));
    out(55, span) = '\n';
    out = out(out > 0);
    out(out == '\n') = 0;
    // Logger::endLastTask();

    // Logger::pauseCollection();
    callGC();
    return Column(out, STRING);
}

AFDataFrame loadProspect(AFDataFrame &s_Prospect, AFDataFrame &batchDate) {
    Logger::startTimer("Prospect");
    AFDataFrame prospect(std::move(s_Prospect));

    auto dim = prospect("Age").dims();
    auto batchID = 1;
    prospect.name("Prospect");

    auto col = Column(tile(batchDate(1)(span, batchDate(0).data() == batchID), dim), DATE);
    prospect.insert(Column(col.hash()), 1, "SK_RecordDateID");
    prospect.insert(Column(array(prospect("SK_RecordDateID").data())), 2, "SK_UpdateDateID");
    prospect.insert(Column(constant(1, dim, u32)), 3, "BatchID");
    prospect.insert(Column(constant(0, dim, b8)), 4, "IsCustomer");
    callGC();
    auto tmp = marketingNameplate(prospect("NetWorth").data(), prospect("Income").data(),
                                  prospect("NumberCreditCards").data(),
                                  prospect("NumberChildren").data(), prospect("Age").data(),
                                  prospect("CreditRating").data(),
                                  prospect("NumberCars").data());
    prospect.add(std::move(tmp), "MarketingNameplate");
    Logger::logTime("Prospect", false);
    return prospect;
}

AFDataFrame loadStagingTrade(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Trade.txt");
    Logger::startTimer("StagingTrade");
    AFDataFrame frame;
    // Logger::startCollection();

    // Logger::startTask("Trade Load");
    AFParser parser(file, '|', false);
    // Logger::endLastTask();

    // Logger::startTask("Trade Parse");
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asDateTime(1, YYYYMMDD));
    frame.add(parser.parse<char*>(2));
    frame.add(parser.parse<char*>(3));
    frame.add(parser.parse<bool>(4));
    frame.add(parser.parse<char*>(5));
    frame.add(parser.parse<unsigned int>(6));
    callGC();
    frame.add(parser.parse<double>(7));
    frame.add(parser.parse<unsigned int>(8));
    frame.add(parser.parse<unsigned long long>(9));
    frame.add(parser.parse<double>(10));
    frame.add(parser.parse<double>(11));
    frame.add(parser.parse<double>(12));
    frame.add(parser.parse<double>(13));
    // Logger::endLastTask();
    Logger::logTime("StagingTrade", false);
    // Logger::pauseCollection();
    callGC();
    return frame;
}

AFDataFrame loadStagingTradeHistory(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TradeHistory.txt");
    Logger::startTimer("StagingTradeHistory");
    AFDataFrame frame;
    // Logger::startCollection();

    // Logger::startTask("Trade History Load");
    AFParser parser(file, '|', false);
    // Logger::endLastTask();

    // Logger::startTask("Trade History Parse");
    frame.add(parser.parse<unsigned long long>(0));
    frame.add(parser.asDateTime(1, YYYYMMDD));
    frame.add(parser.parse<char*>(2));
    // Logger::endLastTask();
    Logger::logTime("StagingTradeHistory", false);
    // Logger::pauseCollection();
    callGC();
    return frame;
}

Customer splitCustomer(AFDataFrame &&s_Customer) {
    auto idx = s_Customer(0) == "NEW";
    auto newC = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer(0) == "ADDACCT";
    auto add = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer(0) == "UPDACCT";
    auto uAcc = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer(0) == "CLOSEACCT";
    auto cAcc = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer(0) == "UPDCUST";
    auto uCus = new AFDataFrame(s_Customer.select(idx));
    idx = s_Customer(0) == "INACT";
    auto inAc = new AFDataFrame(s_Customer.select(idx));

    return Customer(newC, add, uAcc, cAcc, uCus, inAc);
}

#ifdef CUSTOMER
inline Column phoneNumberProcessing(Column const &ctry, Column const &area, Column const &local, Column const &ext) {
    auto const cond1 = where(ctry.irow(1) && area.irow(1) && local.irow(1));
    auto const cond2 = where(ctry.irow(1) == 0 &&  area.irow(1) && local.irow(1));
    auto const cond3 = where(area.irow(1) == 0 && local.irow(1));
    auto const extNotNull = where(ext.irow(1));
    auto const c = 3;
    auto const a = 3;
    auto const l = 10;
    auto const e = 5;

    auto out = constant(0, dim4(c + a + l + e + 4, ctry.length()), u8);
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

    idx = af::diff1(af::join(1, af::constant(0, 1, u64), hflat(where64(out == 0))), 1);
    idx(0) += 1;
    idx = join(0, af::scan(idx, 1, AF_BINARY_ADD, false), idx);
    return Column(out, idx);
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
                                                    frame.column(20).data(), frame.column(21).data()), 15, "Phone1");
    dimCustomer.insert(Column(phoneNumberProcessing(frame.column(22).data(), frame.column(23).data(),
                                                    frame.column(24).data(), frame.column(25).data()), 16, "Phone2");
    dimCustomer.insert(Column(phoneNumberProcessing(frame.column(26).data(), frame.column(27).data(),
                                                    frame.column(28).data(), frame.column(29).data()), 17, "Phone3");
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
#endif


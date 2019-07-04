#include <utility>

//
// Created by Bryan Wong on 2019-06-28.
//

#include "TPC_DI.h"
#include "FinwireParser.h"
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>

namespace fs = boost::filesystem;

Finwire::Finwire(AFDF_ptr cmp, AFDF_ptr fin, AFDF_ptr sec) :
company(std::move(cmp)), financial(std::move(fin)), security(std::move(sec)) {}

Finwire::Finwire(Finwire&& other) : company(nullptr), financial(nullptr), security(nullptr) {
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
    auto lval = parser.asU64(0);
    frame.add(lval, AFDataFrame::U64);
    lval = parser.asDate(1, YYYYMMDD);
    frame.add(lval, AFDataFrame::DATE);
    for (int i = 2;  i < 17; i += 2) {
        lval = parser.asString(i);
        frame.add(lval, AFDataFrame::STRING);
        lval = parser.asUint(i + 1);
        frame.add(lval, AFDataFrame::UINT);
    }
    lval = parser.stringToBoolean(17);
    frame.add(lval, AFDataFrame::BOOL);
    return frame;
}

AFDataFrame loadDimTime(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Time.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    auto lval = parser.asU64(0);
    frame.add(lval, AFDataFrame::U64);
    lval = parser.asTime(1);
    frame.add(lval, AFDataFrame::TIME);

    for (int i = 2;  i < 7; i += 2) {
        lval =parser.asUint(i);
        frame.add(lval, AFDataFrame::UINT);
        lval = parser.asString(i + 1);
        frame.add(lval, AFDataFrame::STRING);
    }
    lval = parser.stringToBoolean(8);
    frame.add(lval, AFDataFrame::BOOL);
    lval = parser.stringToBoolean(9);
    frame.add(lval, AFDataFrame::BOOL);
    return frame;
}

AFDataFrame loadIndustry(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Industry.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    for (int i = 0;  i < 3; ++i) {
        auto lval = parser.asString(i);
        frame.add(lval, AFDataFrame::STRING);
    }
    return frame;
}

AFDataFrame loadStatusType(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "StatusType.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    for (int i = 0;  i < 2; ++i) {
        auto lval = parser.asString(i);
        frame.add(lval, AFDataFrame::STRING);
    }
    return frame;
}

AFDataFrame loadTaxRate(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TaxRate.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    for (int i = 0;  i < 2; ++i) {
        auto lval = parser.asString(i);
        frame.add(lval, AFDataFrame::STRING);
    }
    auto lval = parser.asFloat(2);
    frame.add(lval, AFDataFrame::FLOAT);
    return frame;
}

AFDataFrame loadTradeType(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "TradeType.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    for (int i = 0;  i < 2; ++i) {
        auto lval = parser.asString(i);
        frame.add(lval, AFDataFrame::STRING);
    }
    for (int i = 2;  i < 4; ++i) {
        auto lval = parser.asUint(i);
        frame.add(lval, AFDataFrame::UINT);
    }
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
            auto lval = parser.asString(0);
            frame.add(lval, AFDataFrame::STRING);
            lval = parser.asUchar(1);
            frame.add(lval, AFDataFrame::UINT);
            lval = parser.asDate(2, YYYYMMDD);
            frame.add(lval, AFDataFrame::DATE);
            lval = parser.asString(3);
            frame.add(lval, AFDataFrame::STRING);
            lval = parser.asInt(4);
            frame.add(lval, AFDataFrame::LONG);
            lval = parser.asFloat(5);
            frame.add(lval, AFDataFrame::DOUBLE);
        } else {
            AFDataFrame tmp;
            auto lval = parser.asString(0);
            tmp.add(lval, AFDataFrame::STRING);
            lval = parser.asUchar(1);
            tmp.add(lval, AFDataFrame::UINT);
            lval = parser.asDate(2, YYYYMMDD);
            tmp.add(lval, AFDataFrame::DATE);
            lval = parser.asString(3);
            tmp.add(lval, AFDataFrame::STRING);
            lval = parser.asInt(4);
            tmp.add(lval, AFDataFrame::LONG);
            lval = parser.asFloat(5);
            tmp.add(lval, AFDataFrame::DOUBLE);
            frame.concatenate(tmp);
        }
    }
    return frame;
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
            stagingCompany = finwire.extractData(FinwireParser::CMP);
            stagingFinancial = finwire.extractData(FinwireParser::FIN);
            stagingSecurity = finwire.extractData(FinwireParser::SEC);
        } else {
            auto tmp = finwire.extractData(FinwireParser::CMP);
            stagingCompany->concatenate(*tmp);
            tmp = finwire.extractData(FinwireParser::FIN);
            stagingFinancial->concatenate(*tmp);
            tmp = finwire.extractData(FinwireParser::SEC);
            stagingSecurity->concatenate(*tmp);
        }
    }
    return Finwire(stagingCompany, stagingFinancial, stagingSecurity);
}

AFDataFrame loadStagingProspect(char const *directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "Prospect.csv");
    AFDataFrame frame;
    AFParser parser(file, ',', false);
    af::array lval;
    for (int i = 0;  i < 12; ++i) {
        lval = parser.asString(i);
        frame.add(lval, AFDataFrame::STRING);
    }
    lval = parser.asU64(12);
    frame.add(lval, AFDataFrame::U64);
    for (int i = 13;  i < 15; ++i) {
        lval = parser.asUchar(i);
        frame.add(lval, AFDataFrame::UCHAR);
    }
    lval = parser.asString(15);
    frame.add(lval, AFDataFrame::STRING);
    lval = parser.asUshort(16);
    frame.add(lval, AFDataFrame::USHORT);
    lval = parser.asUint(17);
    frame.add(lval, AFDataFrame::UINT);
    for (int i = 18;  i < 20; ++i) {
        lval = parser.asString(i);
        frame.add(lval, AFDataFrame::STRING);
    }
    lval = parser.asUchar(20);
    frame.add(lval, AFDataFrame::UCHAR);
    lval = parser.asU64(21);
    frame.add(lval, AFDataFrame::U64);
    return frame;
}

AFDataFrame loadDimBroker(char const* directory, AFDataFrame& dimDate) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "HR.csv");
    AFDataFrame frame;
    {
        AFParser parser(directory, ',');
        auto lval = parser.asUint(0);
        frame.add(lval, AFDataFrame::UINT);
        lval = parser.asUint(1);
        frame.add(lval, AFDataFrame::UINT);

        for (int i = 2;  i <= 7; ++i) {
            auto lval = parser.asString(i);
            frame.add(lval, AFDataFrame::STRING);
        }
    }
    frame.stringMatch(5, "314");
    frame.remove(5);
    auto length = frame.data()[0].dims(0);
    auto lval = af::range(af::dim4(length), 0, u64);
    frame.insert(lval, AFDataFrame::U64, 0);
    lval = af::constant(1, af::dim4(length), b8);
    frame.add(lval, AFDataFrame::BOOL);
    dimDate.dateSort(1);
    lval = af::tile(dimDate.data()[1](0,af::span), length);
    frame.add(lval, AFDataFrame::DATE);
    lval = af::tile(AFDataFrame::endDate(), length);
    frame.add(lval, AFDataFrame::DATE);
    return frame;
}

AFDataFrame loadStagingCashBalances(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "CashTransaction.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    af::array lval;
    lval = parser.asU64(0);
    frame.add(lval, AFDataFrame::U64);
    lval = parser.asDateTime(1, YYYYMMDD);
    frame.add(lval, AFDataFrame::DATE);
    lval = parser.asDouble(2);
    frame.add(lval, AFDataFrame::DOUBLE);
    lval = parser.asString(3);
    frame.add(lval, AFDataFrame::STRING);
    return frame;
}

AFDataFrame loadStagingWatches(char const* directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "WatchHistory.txt");
    AFDataFrame frame;
    AFParser parser(file, '|', false);
    af::array lval;
    lval = parser.asU64(0);
    frame.add(lval, AFDataFrame::U64);
    lval = parser.asString(1);
    frame.add(lval, AFDataFrame::STRING);
    lval = parser.asDateTime(2, YYYYMMDD);
    frame.add(lval, AFDataFrame::DATE);
    lval = parser.asString(3);
    frame.add(lval, AFDataFrame::STRING);
    return frame;
}

AFDataFrame loadDimCompany(AFDataFrame& s_Company, AFDataFrame& s_Industry, AFDataFrame& s_StatusType);
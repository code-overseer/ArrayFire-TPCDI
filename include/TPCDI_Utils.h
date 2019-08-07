#ifndef ARRAYFIRE_TPCDI_TPCDI_UTILS_H
#define ARRAYFIRE_TPCDI_TPCDI_UTILS_H
#include <arrayfire.h>
#include <iostream>
#include <tuple>
#include <rapidxml.hpp>
#include <unordered_map>
#include "Enums.h"

template<typename T>
inline void print(T i, std::ostream &out = std::cout) { af::sync(); out << i << std::endl; }
void printStr(af::array str_array, std::ostream &out = std::cout);

namespace TPCDI_Utils {
    std::string loadFile(char const *filename);
    std::string collect(std::vector<std::string> const &files, bool hasHeader = false);
    af::array flipdims(af::array const &arr);
    af::array stringToDate(af::array const &datestr, bool isDelimited = false, DateFormat dateFormat = YYYYMMDD);
    af::array stringToTime(af::array const &timestr, bool isDelimited = false);
    af::array stringToDateTime(af::array &datetimestr, bool isDelimited = false, DateFormat dateFormat = YYYYMMDD);;
    af::array where64(af::array const &input);
}

namespace XML_Parser {
    typedef std::unordered_map<std::string, int> StrToInt;
    typedef std::string String;
    typedef rapidxml::xml_node<> Node;
    void fillBlanks(int &count, std::string fieldName, StrToInt &tracker, String &data, bool isAtt = false);

    void depthFirstAppend(String &data, Node *node, StrToInt &tracker,  String branch, Node *root);

    void learnFieldNames(Node *node, StrToInt &tracker, String branch, Node *root);

    std::string flattenCustomerMgmt(char const *directory);
}

#endif //ARRAYFIRE_TPCDI_TPCDI_UTILS_H

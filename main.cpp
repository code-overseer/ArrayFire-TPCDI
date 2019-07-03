#include <cstdio>
#include <cstdlib>
#include <rapidxml.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "BatchFunctions.h"
#include "Tests.h"
#include "TPC_DI.h"
#include "XMLFlattener.h"

using namespace af;
using namespace rapidxml;

char const* HOME = getenv("HOME");
char const* HR = "/Downloads/TPCData/HR3.csv";
char const* DATE = "/Downloads/TPCData/TestDate.csv";
char const* UINT = "/Downloads/TPCData/TestUint.csv";
char const* UCHAR = "/Downloads/TPCData/TestUchar.csv";
char const* INT = "/Downloads/TPCData/TestInt.csv";
char const* FLOAT = "/Downloads/TPCData/TestFloat.csv";

int main(int argc, char *argv[])
{
//    std::string data;
//    xml_document<> doc;
//    {
//        std::ifstream file("/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/CustomerMgmt.xml");
//        std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
//        data.reserve(buffer.size());
//        file.close();
//        buffer.push_back('\0');
//        // Parse the buffer using the xml file parsing library into doc
//        doc.parse<0>(&buffer[0]);
//    }
//    xml_node<>* root = doc.first_node();
//    std::unordered_map<std::string, int> fieldTracker;
//    auto node = root->first_node();
//    std::string branch;
//    learnFieldNames(node, fieldTracker, branch, node);
//
//    while (node) {
//        depthFirstAppend(data, node, fieldTracker, branch, node);
//        node = node->next_sibling();
//    }
//
//    print(data);
    auto a = loadAudit("/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/");

    return 0;
}

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

using namespace af;
using namespace rapidxml;

char const* HOME = getenv("HOME");
char const* HR = "/Downloads/TPCData/HR3.csv";
char const* DATE = "/Downloads/TPCData/Date.txt";
char const* UINT = "/Downloads/TPCData/TestUint.csv";
char const* INT = "/Downloads/TPCData/TestInt.csv";
char const* FLOAT = "/Downloads/TPCData/TestFloat.csv";

void dfa(std::string &data, xml_node<> *node, std::unordered_map<char const *, int> &tracker,
                      xml_node<> *const root);

void lfn(xml_node<> *node, std::unordered_map<char const *, int> &tracker, xml_node<> *const root);

int main(int argc, char *argv[])
{
    std::string data;
    xml_document<> doc;
    {
        std::ifstream file("/Users/bryanwong/Documents/MPSI/DIGen/Data/Batch1/CustomerMgmt.xml");
        std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
        data.reserve(buffer.size());
        file.close();
        buffer.push_back('\0');
        // Parse the buffer using the xml file parsing library into doc
        doc.parse<0>(&buffer[0]);
    }
    xml_node<>* root = doc.first_node();
    std::unordered_map<char const*, int> fieldTracker;
    auto a = root->first_node();
    lfn(a, fieldTracker, a);
    print(data);
    for (std::pair<char const*, int> element : fieldTracker)
    {
        std::cout << element.first << " :: " << element.second << std::endl;
    }

    return 0;
}


void dfa(std::string &data, xml_node<> *node, std::unordered_map<char const *, int> &tracker,
                      xml_node<> *const root) {
    static int count = 0;
    if (node == root) count = 0;
    auto att = node->first_attribute();
    // if count != tracker int, loop for tracker int to add number of '|', increasing count along the way
    while (att) {
        data += att->value();
        data += '|';
        ++count;
        att = att->next_attribute();
    }
    auto child = node->first_node();
    // if count != tracker int, loop for tracker int to add number of '|', increasing count along the way
    if (!child) {
        data += node->value();
        data += '|';
        ++count;
        return;
    }

    while (child) {
        dfa(data, child, tracker, root);
        child = child->next_sibling();
    }
    // if count != endCount (36), loop to add number of '|'
    // end at 36
    if (node == root) data.back() = '\n';
}

void lfn(xml_node<> *node, std::unordered_map<char const *, int> &tracker, xml_node<> *const root) {
    static int count = 0;
    if (node == root) count = 0;
    auto att = node->first_attribute();
    // if count != tracker int, loop for tracker int to add number of '|', increasing count along the way
    while (att) {
        tracker.insert(std::make_pair(att->name(), count));
        ++count;
        att = att->next_attribute();
    }
    auto child = node->first_node();
    // if count != tracker int, loop for tracker int to add number of '|', increasing count along the way
    if (!child) {
        auto name = (!strcmp(node->name(),"\0")) ? node->parent()->name() : node->name();
        tracker.insert(std::make_pair(name, count));
        ++count;
        return;
    }

    while (child) {
        lfn(child, tracker, root);
        child = child->next_sibling();
    }
    if (node == root) tracker.insert(std::make_pair("End", count));
}
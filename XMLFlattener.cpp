//
// Created by Bryan Wong on 2019-07-01.
//

#include "XMLFlattener.h"
#include <cstring>
using namespace rapidxml;
typedef std::unordered_map<std::string, int> SIMap;

void
fillBlanks(int &count, std::string fieldName, SIMap &tracker, std::string &data, bool isAtt) {
    if (!tracker.count(fieldName)) {
        char msg[32];
        char const* type = (isAtt) ? "attribute" : "element";
        sprintf(msg, "Could not find %s [%s]", type, fieldName.c_str());
        throw std::runtime_error(msg);
    }
    for (int i = count; i < tracker.at(fieldName); ++i) {
        data += "|";
        ++count;
    }
}

void depthFirstAppend(std::string &data, xml_node<> *node, SIMap &tracker, std::string branch, xml_node<> *const root) {
    static int count = 0;
    if (node == root) {
        count = 0;
        branch = "";
    }

    auto att = node->first_attribute();
    branch += node->name();
    while (att) {
        auto name = branch + att->name();
        fillBlanks(count, name, tracker, data, true);
        data += att->value();
        data += '|';
        ++count;
        att = att->next_attribute();
    }
    auto child = node->first_node();

    if (!child) {
        if (!strcmp(node->value(),"\0")) return;
        fillBlanks(count, branch, tracker, data);
        data += node->value();
        data += '|';
        ++count;
        return;
    }

    while (child) {
        depthFirstAppend(data, child, tracker, branch, root);
        child = child->next_sibling();
    }

    if (node == root) {
        fillBlanks(count, std::string("End"), tracker, data);
        data.back() = '\n';
    }
}

void learnFieldNames(xml_node<> *node, SIMap &tracker, std::string branch, xml_node<> *const root) {
    static int count = 0;
    if (node == root) {
        count = 0;
        branch = "";
    }

    auto att = node->first_attribute();
    branch += node->name();
    while (att) {
        auto name = branch + att->name();
        tracker.insert(std::make_pair(name, count));
        ++count;
        att = att->next_attribute();
    }
    auto child = node->first_node();

    if (!child) {
        tracker.insert(std::make_pair(branch, count));
        ++count;
        return;
    }

    while (child) {
        learnFieldNames(child, tracker, branch, root);
        child = child->next_sibling();
    }
    if (node == root) tracker.insert(std::make_pair(std::string("End"), count));
}

std::string flattenCustomerMgmt(char const *directory) {
    char file[128];
    strcpy(file, directory);
    strcat(file, "CustomerMgmt.xml");
    std::string data;
    xml_document<> doc;
    
    std::ifstream inFile(file);
    std::vector<char> buffer((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    data.reserve(buffer.size());
    inFile.close();
    buffer.push_back('\0');
    // Parse the buffer using the xml file parsing library into doc
    doc.parse<0>(&buffer[0]);
    
    xml_node<>* root = doc.first_node();
    std::unordered_map<std::string, int> fieldTracker;
    auto node = root->first_node();
    std::string branch;
    learnFieldNames(node, fieldTracker, branch, node);
    
    while (node) {
        depthFirstAppend(data, node, fieldTracker, branch, node);
        node = node->next_sibling();
    }

    return data;
}

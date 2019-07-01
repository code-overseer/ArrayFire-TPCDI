//
// Created by Bryan Wong on 2019-07-01.
//

#include "XMLFlattener.h"

using namespace rapidxml;

void
fillBlanks(int count, char const *fieldName, std::unordered_map<char const *, int> &tracker, std::string &data) {
    if (!tracker.count(fieldName)) {
        char buffer[32];
        sprintf(buffer, "Could not find field [%s]", fieldName);
        throw std::runtime_error(buffer);
    }
    for (int i = count; i < tracker.at(fieldName); ++i) {
        data += "|";
        ++count;
    }
}

void depthFirstAppend(std::string &data, xml_node<> *node, std::unordered_map<char const *, int> &tracker,
                      xml_node<> *const root) {
    static int count = 0;
    if (node == root) count = 0;
    auto att = node->first_attribute();
    // if count != tracker int, loop for tracker int to add number of '|', increasing count along the way
    while (att) {
        fillBlanks(count, att->name(), tracker, data);
        data += att->value();
        data += '|';
        ++count;
        att = att->next_attribute();
    }
    auto child = node->first_node();
    // if count != tracker int, loop for tracker int to add number of '|', increasing count along the way
    if (!child) {
        auto name = (!strcmp(node->name(),"\0")) ? node->parent()->name() : node->name();
        fillBlanks(count, name, tracker, data);
        data += node->value();
        data += '|';
        ++count;
        return;
    }

    while (child) {
        depthFirstAppend(data, child, tracker, root);
        child = child->next_sibling();
    }

    if (node == root) {
        fillBlanks(count, "End", tracker, data);
        data.back() = '\n';
    }
}

void learnFieldNames(xml_node<> *node, std::unordered_map<char const *, int> &tracker, xml_node<> *const root) {
    static int count = 0;
    if (node == root) count = 0;
    auto att = node->first_attribute();

    while (att) {
        tracker.insert(std::make_pair(att->name(), count));
        ++count;
        att = att->next_attribute();
    }
    auto child = node->first_node();

    if (!child) {
        auto name = (!strcmp(node->name(),"\0")) ? node->parent()->name() : node->name();
        tracker.insert(std::make_pair(name, count));
        ++count;
        return;
    }

    while (child) {
        learnFieldNames(child, tracker, root);
        child = child->next_sibling();
    }
    if (node == root) tracker.insert(std::make_pair("End", count));
}
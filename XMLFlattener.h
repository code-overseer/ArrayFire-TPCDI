//
// Created by Bryan Wong on 2019-07-01.
//

#ifndef ARRAYFIRE_TPCDI_XMLFLATTENER_H
#define ARRAYFIRE_TPCDI_XMLFLATTENER_H

#include <cstdio>
#include <cstdlib>
#include <rapidxml.hpp>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include "AFParser.hpp"

void
fillBlanks(int &count, std::string fieldName, std::unordered_map<std::string, int> &tracker, std::string &data,
           bool isAtt = false);

void depthFirstAppend(std::string &data, rapidxml::xml_node<> *node, std::unordered_map<std::string, int> &tracker,
        std::string branch, rapidxml::xml_node<> *root);

void learnFieldNames(rapidxml::xml_node<> *node, std::unordered_map<std::string, int> &tracker, std::string branch,
                     rapidxml::xml_node<> *root);

std::string flattenCustomerMgmt(char const *directory);

#endif //ARRAYFIRE_TPCDI_XMLFLATTENER_H

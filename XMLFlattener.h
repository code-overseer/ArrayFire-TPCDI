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


void
fillBlanks(int count, char const *fieldName, std::unordered_map<char const *, int> &tracker, std::string &data);

void depthFirstAppend(std::string &data, rapidxml::xml_node<> *node, std::unordered_map<char const *, int> &tracker,
                      rapidxml::xml_node<> *const root);

void learnFieldNames(rapidxml::xml_node<> *node, std::unordered_map<char const *, int> &tracker,
                     rapidxml::xml_node<> *const root);

#endif //ARRAYFIRE_TPCDI_XMLFLATTENER_H

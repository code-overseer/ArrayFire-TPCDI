#include <iostream>
#include <fstream>
#include <rapidxml.hpp>
#include <string.h>
#include <stdio.h>
#include <vector>

using namespace rapidxml;

void rapid() {
  xml_document<> doc;
  xml_node<>* root;
  std::ifstream file("/Users/bryanwong/Documents/MPSI/Data/Batch1/CustomerMgmt.xml");
  std::vector<char> buffer((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
  file.close();
  buffer.push_back('\0');
  // Parse the buffer using the xml file parsing library into doc
  doc.parse<0>(&buffer[0]);
  // Find our root node
  root = doc.first_node("TPCDI:Actions");
  // Iterate over the brewerys
  for (xml_node<> * node = root->first_node("TPCDI:Action"); node; node = node->next_sibling())
  {
    std::cout<<node->first_attribute()->name()<< " : ";
    std::cout<<node->first_attribute()->value()<<std::endl;
  }
}

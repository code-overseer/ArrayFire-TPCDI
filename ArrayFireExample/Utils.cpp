//
//  Utils.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "Utils.hpp"

std::string textToString(char const *filename) {
  std::ifstream file(filename);
  std::string data;
  file.seekg(0, std::ios::end);
  data.reserve(file.tellg());
  file.seekg(0, std::ios::beg);
  data.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  file.close();
  return data;
}


//
//  Utils.hpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#ifndef Utils_hpp
#define Utils_hpp

#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include "AFParser.hpp"
#include <arrayfire.h>

std::string ParseAndTrim(af::Backend backend, const int device = 0,
        const unsigned int runs = 1, const unsigned int scale = 0);
AFParser load_DimDate();
AFParser load_DimBroker();
AFParser test_SignedInt();
AFParser test_Float();
template<typename T>
void print(T i) {std::cout << i << std::endl;}
#endif /* Utils_hpp */

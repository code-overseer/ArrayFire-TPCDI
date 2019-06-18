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
#include "AFCSVParser.hpp"
#include <arrayfire.h>

std::string textToString(char const *filename);
void experiment();
void single_run(af::Backend const backend);
#endif /* Utils_hpp */

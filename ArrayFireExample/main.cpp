/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include <arrayfire.h>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include "AFTest.hpp"
#include "CSVObject.hpp"
#include <DataFrame/DataFrame.h>
#include <af/opencl.h>

using namespace af;
using namespace hmdf;
char const* CSTMGMT = "/Users/bryanwong/Documents/MPSI/Data/Batch1/CustomerMgmt.xml";
char const* HR = "/Users/bryanwong/Documents/MPSI/Data/Batch1/HR.csv";
typedef unsigned short ushort;
int main(int argc, char *argv[])
{
  auto o = CSVObject::parse(HR, false);
  auto v = o.select(5, [](std::string a){return !a.compare("314");});
  auto p = CSVObject(o, v);
  o.trim(v);

}


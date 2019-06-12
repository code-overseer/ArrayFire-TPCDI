/*******************************************************
 * Copyright (c) 2014, ArrayFire
 * All rights reserved.
 *
 * This file is distributed under 3-clause BSD license.
 * The complete license agreement can be obtained at:
 * http://arrayfire.com/licenses/BSD-3-Clause
 ********************************************************/

#include "CSVObject.hpp"
#include "AFCSVParser.hpp"
#include "Utils.hpp"
#include <af/opencl.h>
#include <arrayfire.h>
#include <cstdio>

using namespace af;
char const* HR = "/Users/bryanwong/Downloads/Data/HR1280.csv";

array match(int col, array const* data, array const* indexer, std::string match) {
  int const len = (int)match.size();
  auto starts = indexer->col(col) + constant((int)!!col, indexer->dims(0),u32);
  
  auto i = indexer->col(col + 1) - starts;
  auto out = i == constant(len, i.elements(), u32);
  auto c = where(out);
  // out contains the row indices where the string length matches the target string
  starts = starts(c);
  // starts contains the data indices where the string length matches the target string
  i = tile(range(dim4(1,len),1,u32), starts.dims());
  // i contains the character index of the target string
  
  auto str = tile(array(1, len, match.c_str()), starts.dims());
  i = tile(starts, 1, len) + i;
  i = reorder((*data)(i), 1, 0);
  i = moddims(i, i.elements()/len, len);
  i -= str;
  i = where(!anyTrue(i,1));
  i = c(i);

  out = constant(0, out.elements(), b8);
  out(i) = 1;
  
  return where(out);
}

int main(int argc, char *argv[])
{
  setBackend(AF_BACKEND_CPU);
  setDevice(0);
  auto cpu_o = AFCSVParser::parse(HR, false);
  auto open_cl_o = cpu_o.getData()->host<char>();
  auto open_cl_i = cpu_o.getIndexer()->host<unsigned int>();
  auto dims_o = cpu_o.getData()->dims();
  auto dims_i = cpu_o.getIndexer()->dims();
  timer::start();
  auto cpu_x = match(5, cpu_o.getData(), cpu_o.getIndexer(), "314");
  cpu_o.trim(cpu_x);
  printf("CPU elapsed seconds: %g\n", timer::stop());
  
//  setBackend(AF_BACKEND_OPENCL);
//  setDevice(0);
//  auto o = array(dims_o, open_cl_o);
//  auto i = array(dims_i, open_cl_i);
//  freeHost(open_cl_i);
//  freeHost(open_cl_o);
//  auto opencl_o = AFCSVParser(o, i);
//
//  timer::start();
//  auto opencl_x = match(5, opencl_o.getData(), opencl_o.getIndexer(), "314");
//  opencl_o.trim(opencl_x);
//  printf("OpenCL elapsed seconds: %g\n", timer::stop());
//
  return 0;
  
}


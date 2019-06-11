//
//  AFCSVParser.cpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#include "AFCSVParser.hpp"
#include "Utils.hpp"

using String = std::string;
template<typename T>
using Vector = std::vector<T>;
template<typename T>
using Predicate = std::function<bool(T)>;
typedef unsigned long ulong;
using namespace af;

array findChar(char c, array &csv) {
  auto r = constant((int)c, 1, csv.elements());
  r = csv - r;
  r = iszero(r);
  return where(r);
}

AFCSVParser AFCSVParser::parse(const char *filename, bool header) {
  String str = textToString(filename);

  AFCSVParser output;
  auto data = output._data;
  auto csv = array(1,str.size() + 1, str.c_str());
  
  auto row_end = findChar('\n', csv);
  auto row_start = row_end + constant(1, row_end.elements(), u32);
  row_start.row(0) = 0;
  row_start.row(0) = 0;
  auto col_end = findChar(',', csv);
  col_end = moddims(col_end, col_end.elements()/row_end.elements(), row_end.elements());
  col_end = reorder(col_end, 1, 0);
  auto slices = join(1,row_start, col_end, row_end);
  
  //TODO get slices into _data vector
  
  return output;
}



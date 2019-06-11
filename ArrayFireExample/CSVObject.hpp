//
//  CSVObject.hpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#ifndef CSVObject_hpp
#define CSVObject_hpp

#include <stdio.h>
#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>

typedef unsigned long ulong;

struct Slice {
  ulong start;
  ulong length;
  Slice(ulong s, ulong e) {
    start = s;
    length = e - s;
  }
};

class CSVObject {
private:
  static std::vector<Slice> _findRowSlice(std::string const csv);
  static std::vector<Slice> _findColSlice(std::string const csv, Slice const row);
  std::unordered_map<std::string, ulong> _columnNames;
  std::vector<std::vector<std::string>>* _data = new std::vector<std::vector<std::string>>();
public:
  static CSVObject parse(char const* filename, bool header);
  std::vector<ulong> select(int column, std::function<bool(std::string)> predicate);
  CSVObject() {};
  CSVObject(CSVObject& src, std::vector<ulong> selected);
  virtual ~CSVObject() {
    delete _data;
  }
  bool nameColumn(std::string name, ulong idx);
  bool nameColumn(std::string name, std::string old);
  void printRow(std::ostream& str, ulong row);
  void printColumn(std::ostream& str, ulong col);
  std::string get(ulong row, ulong col);
  
};

#endif /* CSVObject_hpp */

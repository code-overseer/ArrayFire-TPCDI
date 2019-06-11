//
//  AFCSVParser.hpp
//  ArrayFireExample
//
//  Created by Bryan Wong on 6/11/19.
//  Copyright © 2019 Bryan Wong. All rights reserved.
//

#ifndef AFCSVParser_hpp
#define AFCSVParser_hpp

#include <arrayfire.h>
#include <unordered_map>

class AFCSVParser {
private:
  static af::array _findRowSlice(std::string const &csv);
  static af::array _findColSlice(std::string const &csv, unsigned long start, unsigned long length);
  std::unordered_map<std::string, unsigned long> _columnNames;
  std::vector<std::vector<std::string>>* _data = new std::vector<std::vector<std::string>>();
public:
  static AFCSVParser parse(char const* filename, bool header);
  af::array select(int column, std::function<bool(std::string)> predicate) const;
  AFCSVParser() {};
  AFCSVParser(AFCSVParser &src, af::array &selected);
  virtual ~AFCSVParser() {
    delete _data;
  }
  bool nameColumn(std::string name, unsigned long idx);
  bool nameColumn(std::string name, std::string old);
  void printRow(std::ostream& str, unsigned long row) const;
  void printColumn(std::ostream& str, unsigned long col) const;
  void trim(af::array &selected);
  std::string get(unsigned long row, unsigned long col) const;
  unsigned long length() const;
  unsigned long width() const;
};


#endif /* AFCSVParser_hpp */

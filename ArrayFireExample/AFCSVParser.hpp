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
  std::unordered_map<std::string, unsigned long> _columnNames;
  af::array _data;
  af::array _indexer;
  unsigned long _length;
  unsigned long _width;
  std::string _getString() const;
public:
  static AFCSVParser parse(char const* filename, bool header);
  static af::array findChar(char c, af::array &csv);
  AFCSVParser() {};
  AFCSVParser(af::array &data, af::array &indexer) : _data(data), _indexer(indexer) {};
  AFCSVParser(AFCSVParser &src, af::array &selected_rows);
  bool nameColumn(std::string name, unsigned long idx);
  bool nameColumn(std::string name, std::string old);
  void printRow(std::ostream& str, unsigned long row) const;
  void printColumn(std::ostream& str, unsigned long col) const;
  /* trim columns */
  void trim(af::array &selected_rows);
  /* Returns specific field in csv */
  std::string get(dim_t row, dim_t col) const;
  /* Returns number of rows */
  unsigned long length() const { return _length; }
  /* Returns number of columns */
  unsigned long width() const { return _width; }
  /* Returns ASCII integer array */
  af::array const* getData() const { return &_data; }
  /* Returns position of newline and commas */
  af::array const* getIndexer() const { return &_indexer; }
};


#endif /* AFCSVParser_hpp */

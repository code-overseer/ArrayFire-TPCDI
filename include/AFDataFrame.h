#ifndef ARRAYFIRE_TPCDI_AFDATAFRAME_H
#define ARRAYFIRE_TPCDI_AFDATAFRAME_H
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <arrayfire.h>
#include "Enums.h"
#include "BatchFunctions.h"
#include "Column.h"

class AFDataFrame {
private:
    std::vector<Column> _data;
    std::string _name;
    std::unordered_map<std::string, unsigned int> _nameToIdx;
    std::unordered_map<unsigned int, std::string> _idxToName;
public:
    AFDataFrame() = default;
    virtual ~AFDataFrame() = default;
    AFDataFrame(AFDataFrame&& other) noexcept;
    AFDataFrame(AFDataFrame const &other);
    AFDataFrame& operator=(AFDataFrame&& other) noexcept;
    AFDataFrame& operator=(AFDataFrame const &other) noexcept;
    void add(Column &column, const char *name = nullptr);
    void add(Column &&column, const char *name = nullptr);
    void insert(Column &column, unsigned int index, const char *name = nullptr);
    void insert(Column &&column, unsigned int index, const char *name = nullptr);
    void remove(unsigned int index);
    AFDataFrame select(af::array const &index, std::string const &name = "") const;
    AFDataFrame project(int const *columns, int size, std::string const &name = "") const;
    AFDataFrame project(std::string const *names, int size, std::string const &name = "") const;
    AFDataFrame unionize(AFDataFrame &frame) const;
    AFDataFrame unionize(AFDataFrame &&frame) const { return unionize(frame); }
    AFDataFrame zip(AFDataFrame const &rhs) const;
    static std::pair<af::array, af::array> setCompare(Column const &lhs, Column const &rhs);
    static std::pair<af::array, af::array> setCompare(af::array const &lhs, af::array const &rhs);
    void sortBy(unsigned int col, bool isAscending = true);
    void sortBy(unsigned int const *columns, unsigned int size, const bool *isAscending = nullptr);
    void sortBy(std::string const *columns, unsigned int size, bool const *isAscending = nullptr);
    AFDataFrame equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const;
    void nameColumn(const std::string& name, unsigned int column);
    std::string name(const std::string& str);
    void printAllNames() {for (auto const& i : _nameToIdx) printf("%s : %d\n", i.first.c_str(), i.second); }
    void flushToHost();
    void clear();
    inline void remove(std::string const &name) { remove(_nameToIdx[name]); }
    inline AFDataFrame zip(AFDataFrame &&rhs) const { return zip(rhs); }
    inline bool isEmpty() { return _data.empty() || _data[0].isempty(); }
    inline std::vector<Column> &columns() { return _data; }
    inline std::vector<Column> const &columns_() const { return _data; }
    inline Column &operator()(unsigned int column) { return _data[column]; }
    inline Column &operator()(std::string const &name) { return (*this)(_nameToIdx.at(name)); }
    inline Column const &column_(unsigned int i) const { return _data[i]; }
    inline Column const &column_(std::string const &name) const { return column_(_nameToIdx.at(name)); }
    inline AFDataFrame equiJoin(AFDataFrame const &rhs, std::string const &lName, std::string const &rName) const { return equiJoin(rhs, _nameToIdx.at(lName), rhs._nameToIdx.at(rName)); }
    inline uint64_t length() const { return _data.empty() ? 0 : _data[0].length(); }
    inline void nameColumn(const std::string& name, const std::string &old) { nameColumn(name, _nameToIdx.at(old)); }
    inline std::string name() const { return _name; }
    inline void sortBy(std::string const &name, bool isAscending = true) { sortBy(_nameToIdx.at(name), isAscending); }
};
#endif //ARRAYFIRE_TPCDI_AFDATAFRAME_H

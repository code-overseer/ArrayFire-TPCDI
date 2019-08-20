#ifndef ARRAYFIRE_TPCDI_AFDATAFRAME_H
#define ARRAYFIRE_TPCDI_AFDATAFRAME_H

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <vector>
#include <unordered_map>
#include <initializer_list>
#include <arrayfire.h>
#include "Enums.h"
#include "Column.h"

class AFDataFrame {
private:
    typedef std::initializer_list<std::string> str_list;
    typedef std::initializer_list<bool> bool_list;
    std::vector<Column> _data;
    std::string _name;
    std::unordered_map<std::string, unsigned int> _nameToCol;
    std::unordered_map<unsigned int, std::string> _colToName;
public:
    AFDataFrame() = default;

    virtual ~AFDataFrame() {
        _data.clear();
        af::deviceGC();
    }

    AFDataFrame(AFDataFrame &&other) noexcept;

    AFDataFrame(AFDataFrame const &other) = default;

    AFDataFrame &operator=(AFDataFrame &&other) noexcept;

    AFDataFrame &operator=(AFDataFrame const &other) noexcept;

    void add(Column &column, std::string const &name = "");

    void add(Column &&column, std::string const &name = "");

    void insert(Column &column, unsigned int index, std::string const &name = "");

    void insert(Column &&column, unsigned int index, std::string const &name = "");

    void remove(unsigned int index);

    AFDataFrame select(af::array const &index, std::string const &name = "") const;

    AFDataFrame project(int const *columns, int size, std::string const &name = "") const;

    AFDataFrame project(std::string const *columns, int size, std::string const &name = "") const;

    AFDataFrame project(str_list columns, std::string const &name = "") const;

    AFDataFrame unionize(AFDataFrame &frame) const;

    AFDataFrame unionize(AFDataFrame &&frame) const { return unionize(frame); }

    AFDataFrame zip(AFDataFrame const &rhs) const;

    void sortBy(unsigned int col, bool isAscending = true);

    void sortBy(unsigned int const *columns, unsigned int size, const bool *isAscending = nullptr);

    void sortBy(std::string const *columns, unsigned int size, bool const *isAscending = nullptr);

    void sortBy(str_list columns, bool_list isAscending = bool_list());

    AFDataFrame equiJoin(AFDataFrame const &rhs, int lhs_column, int rhs_column) const;

    void nameColumn(const std::string &name, unsigned int column);

    std::string name(const std::string &str);

    void flushToHost();

    void clear();

    static std::pair<af::array, af::array> hashCompare(Column const &lhs, Column const &rhs);

    static std::pair<af::array, af::array> hashCompare(af::array const &left, af::array const &right);

    static std::pair<af::array, af::array> crossCompare(Column const &lhs, Column const &rhs);

    static std::pair<af::array, af::array> crossCompare(const af::array &left, const af::array &right);

    inline void printAllNames() { for (auto const &i : _nameToCol) printf("%s : %d\n", i.first.c_str(), i.second); }

    inline void remove(std::string const &name) { remove(_nameToCol[name]); }

    inline AFDataFrame zip(AFDataFrame &&rhs) const { return zip(rhs); }

    inline bool isEmpty() { return _data.empty() || _data[0].isempty(); }

    inline size_t columns() { return _data.size(); }

    inline size_t rows() const { return _data.empty() ? 0 : _data[0].length(); }

    inline Column &operator()(unsigned int column) { return _data[column]; }

    inline Column &operator()(std::string const &name) { return (*this)(_nameToCol.at(name)); }

    inline AFDataFrame equiJoin(AFDataFrame const &rhs, std::string const &lName, std::string const &rName) const {
        return equiJoin(rhs, _nameToCol.at(lName), rhs._nameToCol.at(rName));
    }

    inline void nameColumn(const std::string &name, const std::string &old) { nameColumn(name, _nameToCol.at(old)); }

    inline std::string name() const { return _name; }

    inline void sortBy(std::string const &name, bool isAscending = true) { sortBy(_nameToCol.at(name), isAscending); }
};

#endif //ARRAYFIRE_TPCDI_AFDATAFRAME_H

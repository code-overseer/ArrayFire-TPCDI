#include "AFParser.h"
#include "BatchFunctions.h"
#include "Utils.h"
#include "Column.h"
#include "KernelInterface.h"
#include "AFTypes.h"
#include "Logger.h"
#include <sstream>
#include <utility>
using namespace af;
using namespace BatchFunctions;
using namespace Utils;

AFParser::AFParser(char const *filename, char const delimiter, bool const hasHeader) : _filename(filename), _delimiter(delimiter) {
    Logger::startTimer("CPU Ingestion");
    std::string txt = Utils::loadFile(_filename);
    if (txt.back() != '\n') txt += '\n';
    Logger::logTime("CPU Ingestion", false);
    Logger::startTimer("GPU Ingestion");
    _data = array(txt.size() + 1, txt.c_str()).as(u8);
    af::sync();

    txt = "";
    _generateIndexer(hasHeader);
    callGC();
    Logger::logTime("GPU Ingestion", false);
}

AFParser::AFParser(const std::vector<std::string> &files, char const delimiter, bool const hasHeader) : _delimiter(delimiter) {
    Logger::startTimer("CPU Ingestion");
    auto text = collect(files, hasHeader);
    Logger::logTime("CPU Ingestion", false);
    Logger::startTimer("GPU Ingestion");
    _data = array(text.size() + 1, text.c_str()).as(u8);
    _generateIndexer(false);
    callGC();
    Logger::logTime("GPU Ingestion", false);
}

AFParser::AFParser(std::string const &text, char const delimiter, bool const hasHeader) : _delimiter(delimiter) {
    Logger::startTimer("GPU Ingestion");
    _data = array(text.size() + 1, text.c_str()).as(u8);
    _generateIndexer(hasHeader);
    callGC();
    Logger::logTime("GPU Ingestion", false);
}

AFParser::~AFParser() {
    callGC();
}

void AFParser::_generateIndexer(bool hasHeader) {
    _indexer = where64(_data == '\n');
    _indexer = hflat(_indexer);
    _length = _indexer.elements();
    
    {
        auto col_end = where64(_data == _delimiter);
        _width = col_end.elements() / _length;
        col_end = moddims(col_end, _width++, _length);
        col_end.eval();
        _indexer = join(0, col_end, _indexer);
    }
    
    if (!_indexer.isempty()) {
        auto row_start = constant(0, 1, _indexer.type());
        if (_length > 1) row_start = join(1, row_start, _indexer.row(end).cols(0, end - 1) + 1);
        row_start.eval();
        _data(_indexer) = 0;
        _indexer = join(0, row_start, _indexer);
    }

    if (hasHeader) {
        _indexer = _indexer.dims(1) <= 1 ? array(1, 0, _indexer.type()) : _indexer.cols(1, end);
        _indexer.eval();
        --_length;
    }
}

template<typename T>
Column AFParser::parse(int column) const {
    if (!_length) return Column(array(0, GetAFType<T>().af_type),  GetAFType<T>().df_type);
    unsigned int const i = column != 0;

    auto idx = _indexer.row(column) + i;
    idx = join(0, idx, (_indexer.row(column + 1) - idx) + 1);

    return Column(numericParse<T>(_data, idx), GetAFType<T>().df_type);
}
template Column AFParser::parse<unsigned char>(int column) const;
template Column AFParser::parse<short>(int column) const;
template Column AFParser::parse<unsigned short>(int column) const;
template Column AFParser::parse<int>(int column) const;
template Column AFParser::parse<unsigned int>(int column) const;
template Column AFParser::parse<long long>(int column) const;
template Column AFParser::parse<double>(int column) const;
template Column AFParser::parse<float>(int column) const;
template Column AFParser::parse<unsigned long long>(int column) const;
template<> Column AFParser::parse<char*>(int column) const {
    if (!_length) return Column(array(0, u8), array(0,u64));
    unsigned int const i = column != 0;
    
    auto idx = _indexer.row(column) + i;
    idx = join(0, idx, (_indexer.row(column + 1) - idx) + 1);

    auto out = stringGather(_data, idx);
    return Column(std::move(out), std::move(idx));
}
template<> Column AFParser::parse<bool>(int column) const {
    if (!_length) return Column(array(0, b8), BOOL);
    auto out = _indexer.row(column) + (column != 0);
    out = _data(out);
    if (!out.dims(1)) return Column(array(0, b8), BOOL);
    out = out.row(0);
    out(out == 'T' || out == 't' || out == '1' || out =='Y' || out == 'y' || out != 0 || out != _delimiter) = 1;
    out(out != 1) = 0;
    out = moddims(out, dim4(out.dims(1), out.dims(0))).as(b8);
    out.eval();
    return Column(std::move(out), BOOL);
}
Column AFParser::asTime(int column, bool const isDelimited) const {
    auto out = parse<char*>(column);
    out.toTime(isDelimited);
    return out;
}
Column AFParser::asDate(int column, bool isDelimited, DateFormat inputFormat) const {
    auto out = parse<char*>(column);
    out.toDate(isDelimited, inputFormat);
    return out;
}
Column AFParser::asDateTime(int column, DateFormat inputFormat) const {
    auto out = parse<char*>(column);
    out.toDateTime(inputFormat);
    return out;
}






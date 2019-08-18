#include "include/FinwireParser.h"
#include "include/BatchFunctions.h"
#include "include/Utils.h"
#include "include/KernelInterface.h"
#include <fstream>
#include <include/Logger.h>

using namespace af;
using namespace Utils;

FinwireParser::FinwireParser(std::vector<std::string> const &files) {
    Logger::startTask("Finwire files concat");
    auto text = collect(files);
    if (text.back() != '\n') text += '\n';
    Logger::endLastTask();
    Logger::startTask("Finwire load");
    _data = array(text.size(), text.c_str()).as(u8);
    auto row_end = hflat(where64(_data == '\n'));
    auto row_start = join(1, constant(0, 1, row_end.type()), row_end.cols(0, end - 1) + 1);
    _indexer = join(0, row_start, row_end);
    _data.eval();
    _indexer.eval();
    Logger::endLastTask();
}

Column FinwireParser::_extract(array &start, unsigned int const length, FinwireParser::RecordType const &type) const {
    af::array idx;
    if (type != CMP && length == 60) {
        idx = join(0, start, hflat((_data(start) != '0') * 50 + 11).as(start.type()));
    } else {
        idx = join(0, start, constant(length + 1, start.dims(), start.type()));
        start += length;
    }
    auto out = stringGather(_data, idx);
    out.eval();
    idx.eval();
    return Column(std::move(out), std::move(idx));
}

af::array FinwireParser::filterRowsByCategory(const FinwireParser::RecordType &type) const {
    using namespace BatchFunctions;
    auto recType = af::batchFunc(_indexer.row(0), range(dim4(3), 0, u64) + 15, batchAdd);
    af::array tmp = _data(recType);
    tmp = moddims(tmp, dim4(3, tmp.elements() / 3));
    return allTrue(batchFunc(tmp, array(3, _search[type]), batchEqual), 0);
}

AFDataFrame FinwireParser::extractCmp() const {
    callGC();
    AFDataFrame output;
    ull const *lengths = _CMPLengths;

    Logger::startTask("CMP columns extraction");
    auto rows = filterRowsByCategory(CMP);
    af::array start = _indexer(0, rows);

    while(*lengths) {
        output.add(_extract(start, *lengths, CMP));
        ++lengths;
    }
    Logger::endLastTask();
    Logger::startTask("CMP columns parsing");
    output(0).toDateTime(YYYYMMDD);
    output(3).cast<unsigned long long>();
    output(7).toDate(false, YYYYMMDD);
    Logger::endLastTask();
    return output;
}

AFDataFrame FinwireParser::extractFin() const {
    callGC();
    AFDataFrame output;
    ull const *lengths = _FINLengths;

    Logger::startTask("FIN columns extraction");
    auto rows = filterRowsByCategory(FIN);
    af::array start = _indexer(0, rows);
    while(*lengths) {
        output.add(_extract(start, *lengths, FIN));
        ++lengths;
    }
    Logger::endLastTask();

    Logger::startTask("FIN columns parsing");
    output(0).toDateTime(YYYYMMDD);
    output(2).cast<unsigned short>();
    output(3).cast<unsigned char>();
    output(4).toDate(false, YYYYMMDD);
    output(5).toDate(false, YYYYMMDD);
    for (int i = 6; i < 14; ++i) output(i).cast<double>();
    output(14).cast<unsigned long long>();
    output(15).cast<unsigned long long>();
    Logger::endLastTask();
    return output;
}

AFDataFrame FinwireParser::extractSec() const {
    callGC();
    ull const *lengths = _SECLengths;
    AFDataFrame output;

    Logger::startTask("SEC columns extraction");
    auto rows = filterRowsByCategory(SEC);
    af::array start = _indexer(0, rows);
    while(*lengths) {
        output.add(_extract(start, *lengths, SEC));
        ++lengths;
    }
    Logger::endLastTask();
    Logger::startTask("SEC columns parsing");
    output(0).toDateTime(YYYYMMDD);
    output(7).cast<unsigned long long>();
    output(8).toDate(false, YYYYMMDD);
    output(9).toDate(false, YYYYMMDD);
    output(10).cast<double>();
    Logger::endLastTask();
    return output;
}

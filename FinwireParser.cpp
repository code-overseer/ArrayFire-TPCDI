//
// Created by Bryan Wong on 2019-07-03.
//

#include "FinwireParser.h"
#include "BatchFunctions.h"
using namespace af;
using namespace BatchFunctions;

FinwireParser::FinwireParser(char const *filename) {

    auto text = loadFile(filename);
    if (text.back() != '\n') text += '\n';
    _finwireData = array(text.size() + 1, text.c_str()).as(u8);
    _finwireData = _finwireData(where(_finwireData != '\r'));
    _finwireData.eval();

    auto row_end = where(_finwireData == '\n');
    row_end = moddims(row_end, dim4(1, row_end.elements()));
    auto row_start = join(1, constant(0, 1, u32), row_end.cols(0, end - 1) + 1);
    _indexer = join(0, row_start, row_end);
    _maxRowWidth = max(diff1(_indexer, 0)).scalar<uint32_t>();
}

af::array FinwireParser::_extract(af::array const &start, int const length) const {
    array column;
    column = batchFunc(start.row(0), range(dim4(length, 1), 0, u32), batchAdd);
    column(where(batchFunc(column, start.row(1), batchGE))) = UINT32_MAX;
    {
        auto cond = where(column != UINT32_MAX);
        array tmp = column(cond);
        tmp = _finwireData(tmp).as(u32);
        tmp.eval();
        column(cond) = tmp;
    }
    column(where(column == UINT32_MAX)) = 0;
    column = join(0, column.as(u8), constant(0, dim4(1, column.dims(1)), u8));
    column.eval();
    return column;
}

std::shared_ptr<AFDataFrame> FinwireParser::extractData(RecordType const type) const {
    std::shared_ptr<AFDataFrame> output = std::make_shared<AFDataFrame>();
    array rows;
    {
        auto recType = batchFunc(_indexer.row(0), range(dim4(3, 1), 0, u32) + 15, batchAdd);
        array tmp = _finwireData(recType);
        tmp = moddims(tmp, dim4(3, tmp.dims(0) / 3));
        tmp.eval();
        rows = where(allTrue(batchFunc(tmp, array(3, 1, _search[type]), batchEqual), 0));
        if (rows.isempty()) {
            for (int i = 0; i < _widths[type]; ++i) {
                auto lval = array(1, 0, u8);
                output->add(lval, AFDataFrame::STRING);
            }
            return output;
        }
    }
    int const *lengths;
    switch (type) {
        case CMP:
            lengths = _CMPLengths;
            break;
        case FIN:
            lengths = _FINLengths;
            break;
        case SEC:
            lengths = _SECLengths;
            break;
        default:
            throw std::runtime_error("Invalid type");
    }

    array start = _indexer(span, rows);
    for (; *lengths != -1; lengths++) {
        auto lval = _extract(start, *lengths);
        output->add(lval, AFDataFrame::STRING);
        start(0, span) += *lengths;
    }
    return output;
}
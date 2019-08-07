#include "include/TPCDI_Utils.h"
#include "include/AFDataFrame.h"
#include "include/BatchFunctions.h"
#include <fstream>
#include <string>
#include <thread>
#include <cstring>
using namespace af;
using namespace BatchFunctions;

void printStr(array str_array, std::ostream &out) {
    str_array(str_array == 0) = ',';
    str_array = join(0, flat(str_array), af::constant(0, 1, u8));
    str_array.eval();
    char *d = (char*) malloc(str_array.bytes());
    str_array.host(d);
    print(d);
    af::freeHost(d);
}

std::string TPCDI_Utils::loadFile(char const *filename) {
    std::ifstream file(filename);
    std::string text;
    file.seekg(0, std::ios::end);
    text.reserve(((size_t)file.tellg()) + 1);
    file.seekg(0, std::ios::beg);
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
    return text;
}

void inline bulk_loader(char const *file, std::string *data, unsigned long *sizes, bool const hasHeader) {
    *data = TPCDI_Utils::loadFile(file);
    if (hasHeader) {
        auto pos = data->find_first_of('\n');
        data->erase(0, (pos != std::string::npos) ? pos + 1 : std::string::npos);
    }
    if (!data->empty() && data->back() != '\n') *data += '\n';
    *sizes = data->size();
}

std::string TPCDI_Utils::collect(std::vector<std::string> const &files, bool const hasHeader) {
    auto const num = files.size();
    auto const limit = std::thread::hardware_concurrency() - 1;
    std::vector<std::string> data((size_t)num);
    std::vector<unsigned long> sizes((size_t)num);

    std::thread threads[limit];
    for (unsigned long i = 0; i < num / limit + 1; ++i) {
        auto jlim = (i == num / limit) ? num % limit : limit;
        for (decltype(i) j = 0; j < jlim; ++j) {
            auto n = i * limit + j;
            auto d = data.data() + n;
            auto s = sizes.data() + n;
            threads[j] = std::thread(bulk_loader, files[n].c_str(), d, s, hasHeader);
        };
        for (decltype(i) j = 0; j < jlim; ++j) threads[j].join();
    }

    size_t total_size = 0;
    for (int i = 0; i < num; ++i) total_size += sizes[i];
    std::string output;
    output.reserve(total_size + 1);
    for (int i = 0; i < num; ++i) output.append(data[i]);

    return output;
}

af::array TPCDI_Utils::flipdims(af::array const &arr) {
    return moddims(arr, af::dim4(arr.dims(1), arr.dims(0)));
}

af::array TPCDI_Utils::stringToDate(af::array const &datestr, bool const isDelimited, DateFormat dateFormat) {
    if (!datestr.isempty()) return array(3, 0, u16);

    auto out = (isDelimited) ?
              moddims(datestr, dim4(11, datestr.elements() / 11)) :
              moddims(datestr, dim4(9, datestr.elements() / 9));
    out = out.rows(0, end - 1);
    out(span, where(allTrue(out == ' ', 0))) = '0';
    out(where(out >= '0' && out <='9')) -= '0';

    if (isDelimited) out = out(out >= 0 && out <= 9);

    out = moddims(out, dim4(8, out.elements() / 8));
    auto th = flip(pow(10, range(dim4(4), 0, u16)), 0);
    auto tens = flip(pow(10, range(dim4(2), 0, u16)), 0);

    switch (dateFormat) {
        case YYYYMMDD:
            out = join(0, sum(batchFunc(out.rows(0, 3), th, batchMult), 0),
                       sum(batchFunc(out.rows(4, 5), tens, batchMult), 0),
                       sum(batchFunc(out.rows(6, 7), tens, batchMult), 0)).as(u16);
            break;
        case YYYYDDMM:
            out = join(0, sum(batchFunc(out.rows(0, 3), th, batchMult), 0),
                       sum(batchFunc(out.rows(6, 7), tens, batchMult), 0),
                       sum(batchFunc(out.rows(4, 5), tens, batchMult), 0)).as(u16);
            break;
        case MMDDYYYY:
            out = join(0, sum(batchFunc(out.rows(4, 7), th, batchMult), 0),
                       sum(batchFunc(out.rows(0, 1), tens, batchMult), 0),
                       sum(batchFunc(out.rows(2, 3), tens, batchMult), 0)).as(u16);
            break;
        case DDMMYYYY:
            out = join(0, sum(batchFunc(out.rows(4, 7), th, batchMult), 0),
                       sum(batchFunc(out.rows(2, 3), tens, batchMult), 0),
                       sum(batchFunc(out.rows(0, 1), tens, batchMult), 0)).as(u16);
            break;
        default:
            throw std::runtime_error("No such date format");
    }

    out.eval();

    return out;
}

array TPCDI_Utils::stringToTime(af::array const &timestr, bool isDelimited) {
    if (!timestr.isempty()) return array(3, 0, u16);
    auto out = (isDelimited) ?
               moddims(timestr, dim4(9, timestr.elements() / 9)) :
               moddims(timestr, dim4(7, timestr.elements() / 7));
    out = out.rows(0, end - 1);
    out(span, where(allTrue(out == ' ', 0))) = '0';
    out(out >= '0' && out <='9') -= '0';
    if (isDelimited) out(seq(2, 5, 3), span) = 255;
    out = out(out >= 0 && out <= 9);
    out = moddims(out, dim4(6, out.elements() / 6));
    auto tens = flip(pow(10, range(dim4(2), 0, u16)), 0);
    out = join(0, sum(batchFunc(out.rows(0, 1), tens, batchMult), 0),
            sum(batchFunc(out.rows(2, 3), tens, batchMult), 0),
            sum(batchFunc(out.rows(4, 5), tens, batchMult), 0)).as(u16);
    out.eval();

    return out;
}

af::array TPCDI_Utils::stringToDateTime(af::array &datetimestr, bool const isDelimited, DateFormat dateFormat) {
    if (datetimestr.isempty()) return array(6, 0, u16);
    auto length = sum<unsigned int>(max(diff1(where(datetimestr == 0),0),0));
    if (!length) return array(6, 0, u16);
    auto out = moddims(datetimestr, dim4(length, datetimestr.elements() / length));

    af::array date = out.rows(0, isDelimited ? 10 : 8);
    date = stringToDate(date, isDelimited, dateFormat);

    af::array time = out.rows(end - (isDelimited ? 8 : 6), end);
    time = stringToTime(time, isDelimited);

    return join(0, date, time);
}

af::array TPCDI_Utils::where64(af::array const &input) {
    auto b = flat(input > 0);
    auto output = b * range(b.dims(), 0, u64);
    return output(b);
}

void XML_Parser::fillBlanks(int &count, String fieldName, StrToInt &tracker, String &data, bool isAtt) {
    if (!tracker.count(fieldName)) {
        char msg[32];
        char const* type = (isAtt) ? "attribute" : "element";
        sprintf(msg, "Could not find %s [%s]", type, fieldName.c_str());
        throw std::runtime_error(msg);
    }
    for (int i = count; i < tracker.at(fieldName); ++i) {
        data += "|";
        ++count;
    }
}

void XML_Parser::depthFirstAppend(String &data, Node *node, StrToInt &tracker, String branch, Node *const root) {
    static int count = 0;
    if (node == root) {
        count = 0;
        branch = "";
    }

    auto att = node->first_attribute();
    branch += node->name();
    while (att) {
        auto name = branch + att->name();
        fillBlanks(count, name, tracker, data, true);
        data += att->value();
        data += '|';
        ++count;
        att = att->next_attribute();
    }
    auto child = node->first_node();

    if (!child) {
        if (!strcmp(node->value(),"\0")) return;
        fillBlanks(count, branch, tracker, data);
        data += node->value();
        data += '|';
        ++count;
        return;
    }

    while (child) {
        depthFirstAppend(data, child, tracker, branch, root);
        child = child->next_sibling();
    }

    if (node == root) {
        fillBlanks(count, std::string("End"), tracker, data);
        data.back() = '\n';
    }
}

void XML_Parser::learnFieldNames(Node* node, StrToInt &tracker, String branch, Node* const root) {
    static int count = 0;
    if (node == root) {
        count = 0;
        branch = "";
    }

    auto att = node->first_attribute();
    branch += node->name();
    while (att) {
        auto name = branch + att->name();
        tracker.insert(std::make_pair(name, count));
        ++count;
        att = att->next_attribute();
    }
    auto child = node->first_node();

    if (!child) {
        tracker.insert(std::make_pair(branch, count));
        ++count;
        return;
    }

    while (child) {
        learnFieldNames(child, tracker, branch, root);
        child = child->next_sibling();
    }
    if (node == root) tracker.insert(std::make_pair(std::string("End"), count));
}

std::string XML_Parser::flattenCustomerMgmt(char const *directory) {
    using namespace rapidxml;
    char file[128];
    strcpy(file, directory);
    strcat(file, "CustomerMgmt.xml");
    std::string data;
    xml_document<> doc;

    std::ifstream inFile(file);
    std::vector<char> buffer((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
    data.reserve(buffer.size());
    inFile.close();
    buffer.push_back('\0');
    // Parse the buffer using the xml file parsing library into doc
    doc.parse<0>(&buffer[0]);

    xml_node<>* root = doc.first_node();
    std::unordered_map<std::string, int> fieldTracker;
    auto node = root->first_node();
    std::string branch;
    learnFieldNames(node, fieldTracker, branch, node);

    while (node) {
        depthFirstAppend(data, node, fieldTracker, branch, node);
        node = node->next_sibling();
    }

    return data;
}
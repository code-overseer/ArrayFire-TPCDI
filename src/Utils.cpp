#include "Utils.h"
#include "BatchFunctions.h"
#include "Column.h"
#include <fstream>
#include <string>
#include <thread>
#include <cstring>
#include <Logger.h>

#define GC_RATIO 8
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

std::string Utils::loadFile(char const *filename) {
    std::ifstream file(filename);
    std::string text;
    file.seekg(0, std::ios::end);
    text.reserve(((size_t)file.tellg()) + 1);
    file.seekg(0, std::ios::beg);
    text.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
    file.close();
    return text;
}

static void bulk_loader(char const *file, std::string *data, unsigned long *sizes, bool const hasHeader) {
    *data = Utils::loadFile(file);
    if (hasHeader) {
        auto pos = data->find_first_of('\n');
        data->erase(0, (pos != std::string::npos) ? pos + 1 : std::string::npos);
    }
    if (!data->empty() && data->back() != '\n') *data += '\n';
    *sizes = data->size();
}

std::string Utils::collect(std::vector<std::string> const &files, bool const hasHeader) {
    auto const num = files.size();
    auto const limit = std::thread::hardware_concurrency() - 1;
    std::vector<std::string> data((size_t)num);
    std::vector<unsigned long> sizes((size_t)num);

    std::thread threads[limit];
    auto remainder = num % limit;
    auto ilim = num / limit + (remainder ? 1 : 0);
    for (unsigned long i = 0; i < ilim; ++i) {
        auto jlim = ((i == ilim - 1) && remainder) ? remainder : limit;
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

af::array Utils::where64(af::array const &input) {
    auto b = flat(input > 0);
    auto output = b * range(b.dims(), 0, u64);
    return output(b);
}

Column Utils::endDate(int length) {
    auto a = join(0, af::constant(9999, 1, u16), af::constant(12, 1, u16), af::constant(31, 1, u16));
    a = tile(a, dim4(1, length));
    return Column(std::move(a), DATE);
}

void Utils::fillBlanks(int &count, String fieldName, StrToInt &tracker, String &data, bool isAtt) {
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

void Utils::depthFirstAppend(String &data, Node *node, StrToInt &tracker, String branch, Node *const root) {
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

void Utils::learnFieldNames(Node* node, StrToInt &tracker, String branch, Node* const root) {
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

std::string Utils::flattenCustomerMgmt(char const *directory) {
    using namespace rapidxml;
    Logger::startTimer("XML flattening");
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
    Logger::logTime("XML flattening", false);
    return data;
}

void Utils::callGC() {
    size_t alloc;
    size_t locked;

    deviceMemInfo(&alloc, nullptr, &locked, nullptr);
    if (alloc / locked > GC_RATIO) deviceGC();
}

void Utils::MemInfo() {
    size_t alloc;
    size_t locked;

    deviceMemInfo(&alloc, nullptr, &locked, nullptr);
    printf("Allocated: %zu \t Locked: %zu\n", alloc, locked);
}

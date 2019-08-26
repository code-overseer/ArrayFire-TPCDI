#include <sstream>
#include <string>
#include <algorithm>
#include <cstdio>
#include "Logger.h"

void Logger::startTimer(std::string const &name) {
    af::sync();
    _timers[name] = af::timer::start();
}

void Logger::logTime(std::string const &name, bool show) {
    af::sync();
    if (!_timers.count(name)) {
        char buffer[128];
        sprintf(buffer, "Timer %s does not exists, start one first", name.c_str());
        std::cerr << buffer << std::endl;
        return;
    }
    _times[name].emplace_back(af::timer::stop(_timers.at(name)));
    if (show) std::cout << name << ": " << _times[name].back() << std::endl;
}

void Logger::sendToCSV(int const scale) {
    std::stringstream ss;
    auto info = std::string(af::infoString());
    std::replace(info.begin(), info.end(), ',', ' ');
    ss << info;
    char g[256];
    do { ss.getline(g,256); } while(g[0] != '[');
    ss.str(std::string());
    ss << g;
    #ifdef USING_AF
        ss << " -- USING_AF";
    #endif
    ss << '\n';
    #ifdef IS_APPLE
    ss << "Scale," << 3 << '\n';
    #else
    ss << "Scale," << scale << '\n';
    #endif
    for (auto const &data : _times) {
        ss << data.first << ',';
        for (size_t i = 0; i < data.second.size(); ++i) {
            if (i + 1 == data.second.size()) {
                ss << data.second[i] << '\n';
            } else {
                ss << data.second[i] << ',';
            }
        }
    }
    std::ofstream file(directory() + "result.csv", std::ios_base::app);
    file << ss.str();
    file.close();
}


#include <sstream>
#include <string>
#include <algorithm>
#include "include/Logger.h"

Logger& Logger::instance() {
    static Logger instance;
    return instance;
}

void Logger::startTimer(std::string const &name) {
    af::sync();
    instance()._timers[name] = af::timer::start();
}

void Logger::logTime(std::string const &name, bool show) {
    af::sync();
    if (!instance()._timers.count(name)) {
        char buffer[128];
        sprintf(buffer, "Timer %s does not exists, start one first", name.c_str());
        std::cerr << buffer << std::endl;
        return;
    }
    instance()._times[name].emplace_back(af::timer::stop(instance()._timers.at(name)));
    if (show) std::cout << name << ": " << instance()._times[name].back() << std::endl;
}
std::string& Logger::output() {
    static std::string directory;
    return directory;
}

void Logger::sendToCSV() {
    std::stringstream ss;
    auto info = std::string(af::infoString());
    std::replace(info.begin(), info.end(), ',', ' ');
    ss << "Device Info" << ',' << info;
    for (auto const &data : instance()._times) {
        ss << data.first << ',';
        for (size_t i = 0; i < data.second.size(); ++i) {
            if (i + 1 == data.second.size()) {
                ss << data.second[i] << '\n';
            } else {
                ss << data.second[i] << ',';
            }
        }
    }
    std::ofstream file(output() + "result.csv", std::ios_base::app);
    file << ss.str();
    file.close();
}



#include <sstream>
#include <string>
#include <algorithm>
#include "Logger.h"

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
    instance()._times.emplace_back(std::make_pair(name, af::timer::stop(instance()._timers.at(name))));
    if (show) std::cout << instance()._times.back().first << ": " << instance()._times.back().second << std::endl;
}
std::string& Logger::output() {
    static std::string directory;
    return directory;
}

void Logger::sendToCSV() {
    std::stringstream ss;
    auto info = std::string(af::infoString());
    std::replace(info.begin(), info.end(), ',', ' ');
    ss << info;
    for (auto const &data : instance()._times) {
        ss << data.first << ',' << data.second << '\n';
    }
    std::ofstream file(output() + "result.csv", std::ios_base::app);
    file << ss.str();
    file.close();
}



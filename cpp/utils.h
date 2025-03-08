#pragma once

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <filesystem>
#include <sstream>


template<typename T>
class NamedArg {
    const char* name;
    T value;
public:
    NamedArg(const char* n, T v) : name(n), value(v) {}
    
    std::string toString() const {
        std::stringstream ss;
        ss << "--" << name << "=" << value;
        return ss.str();
    }
};

#define NAMED(x) NamedArg(#x, x)

template<typename... Args>
std::string generateCmd(std::string& cmd, Args... args) {
    std::stringstream ss;
    ss << cmd << " ";
    ((ss << NAMED(args).toString() << " "), ...);
    return ss.str();
}

std::pair<std::string, int> executeCmd(const std::string& cmd) {
    std::array<char, 128> buffer;
    std::string result;
    int exitCode;
    
    #ifdef _WIN32
        FILE* pipe = _popen(cmd.c_str(), "r");
    #else
        FILE* pipe = popen(cmd.c_str(), "r");
    #endif
    
    if (!pipe) {
        throw std::runtime_error("popen() failed!");
    }
    
    try {
        while (fgets(buffer.data(), buffer.size(), pipe) != nullptr) {
            result += buffer.data();
        }
    } catch (...) {
        #ifdef _WIN32
            _pclose(pipe);
        #else
            pclose(pipe);
        #endif
        throw;
    }
    
    #ifdef _WIN32
        exitCode = _pclose(pipe);
    #else
        exitCode = pclose(pipe);
    #endif
    
    return {result, exitCode};
}

class SharedLibrary {
private:
    void* handle;

public:
    SharedLibrary(std::string& path) {
        handle = dlopen(path.c_str(), RTLD_LAZY);
        if (!handle) {
            throw std::runtime_error(dlerror());
        }
    }

    ~SharedLibrary() {
        if (handle) {
            dlclose(handle);
        }
    }

    // Get raw function pointer
    void* getRawFunction(const char* funcName) {
        dlerror(); // Clear any existing error
        void* funcPtr = dlsym(handle, funcName);
        const char* error = dlerror();
        if (error) {
            throw std::runtime_error(error);
        }
        return funcPtr;
    }

    // Template to call function with any return type and arguments
    template<typename ReturnType = void, typename... Args>
    ReturnType call(Args... args) {
        auto func = reinterpret_cast<ReturnType(*)(Args...)>(getRawFunction("call"));
        return func(std::forward<Args>(args)...);
    }
};
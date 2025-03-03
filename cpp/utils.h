#pragma once

#include <dlfcn.h>
#include <hip/hip_runtime.h>
#include <iostream>
#include <stdexcept>
#include <filesystem>

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
    SharedLibrary(const char* path) {
        handle = dlopen(path, RTLD_LAZY);
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
    ReturnType call(const char* funcName, Args... args) {
        auto func = reinterpret_cast<ReturnType(*)(Args...)>(getRawFunction(funcName));
        return func(std::forward<Args>(args)...);
    }
};
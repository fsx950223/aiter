import shutil
import os
import subprocess
from jinja2 import Template
import ctypes
import importlib

# AITER_CACHE_DIR=os.environ.get("AITER_CACHE_DIR", "./")

this_dir = os.path.dirname(os.path.abspath(__file__))

AITER_CORE_DIR = os.path.abspath(f"{this_dir}/../")

find_aiter = importlib.util.find_spec("aiter")
if find_aiter is not None:
    if find_aiter.submodule_search_locations:
        package_path = find_aiter.submodule_search_locations[0]
    elif find_aiter.origin:
        package_path = find_aiter.origin
    package_path = os.path.dirname(package_path)
    import site
    site_packages_dirs = site.getsitepackages()
    ### develop mode
    if package_path not in site_packages_dirs:
        AITER_ROOT_DIR = AITER_CORE_DIR
    ### install mode
    else:
        AITER_ROOT_DIR = os.path.abspath(f"{AITER_CORE_DIR}/aiter_meta/")
else:
    print("aiter is not installed.")

BUILD_DIR=os.path.abspath(os.path.join(AITER_ROOT_DIR, "build"))


libs = {}

def init_build_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
    else:
        subprocess.run(f"rm -rf {dir}/*", shell=True)

makefile_template = Template("""
build:
	hipcc -DUSE_ROCM -DENABLE_FP8 -fPIC -shared {{cxxflags | join(" ")}} -I{{includes | join(" ")}} {{sources | join(" ")}} -o lib.so
""")

def compile_lib(src_file, folder, includes=[], sources=[], cxxflags=["-O3", "-std=c++17", "--offload-arch=native"]):
    init_build_dir(os.path.join(BUILD_DIR, folder))
    includes += [f"{AITER_ROOT_DIR}/csrc/include"]
    for include in includes:
        if os.path.isdir(include):
            shutil.copytree(include, BUILD_DIR, dirs_exist_ok=True)
        else:
            shutil.copy(include, BUILD_DIR)
    for source in sources:
        if os.path.isdir(source):
            shutil.copytree(source, os.path.join(BUILD_DIR, folder), dirs_exist_ok=True)
        else:
            shutil.copy(source, os.path.join(BUILD_DIR, folder))
    with open(f"{BUILD_DIR}/{folder}/call_lib.cpp", "w") as f:
        f.write(src_file)
    sources += ["call_lib.cpp"]
    makefile_file = makefile_template.render(includes=includes, sources=sources, cxxflags=cxxflags)
    with open(f"{BUILD_DIR}/{folder}/Makefile", "w") as f:
        f.write(makefile_file)
    subprocess.run(f"cd {BUILD_DIR}/{folder} && make build -j8", shell=True)

def run_lib(folder, *args):
    if folder in libs:
        lib = libs[folder]
    else:
        lib = ctypes.CDLL(f"{BUILD_DIR}/{folder}/lib.so", os.RTLD_LAZY)
        libs[folder] = lib
    lib.call(*args)

def compile_template_op(src_file, folder, includes=[], sources=[], cxxflags=["-O3", "-std=c++17", "--offload-arch=native"]):
    if not os.path.exists(f"{BUILD_DIR}/{folder}/lib.so") or os.environ.get("AITER_FORCE_COMPILE", "0") == "1":
        compile_lib(src_file, folder, includes, sources, cxxflags)
    def wrapper(*args):
        return run_lib(folder, *args)
    return wrapper

project('statphy', 'cpp', default_options: ['cpp_std=c++17', 'warning_level=3'])

name = 'mc_ising'
src = ['main.cc']
dep_openmp = dependency('openmp', language: 'cpp')

executable(name, src, dependencies: dep_openmp)

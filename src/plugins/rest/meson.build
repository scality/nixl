# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Scality AI Connector: HTTP control plane over libcurl, RDMA data plane over
# in-process libibverbs DC.
rest_sources = [
    'cufile_nics.h',
    'dc_descriptor.h',
    'dc_descriptor_provider.cpp',
    'dc_descriptor_provider.h',
    'rdma_descriptor_provider.h',
    'rest_backend.cpp',
    'rest_backend.h',
    'rest_client.cpp',
    'rest_client.h',
    'rest_plugin.cpp',
]

cpp = meson.get_compiler('cpp')
# curl_multi_wakeup() needs libcurl 7.68.
rest_curl_dep = dependency('libcurl', version: '>=7.68.0', required: false)
rest_ibverbs_dep = dependency('libibverbs', required: false)
rest_mlx5_dep = dependency('libmlx5', required: false)
# libibverbs.pc does not give the rdma-core release, so check the flag itself.
rest_ibverbs_relaxed_ordering = rest_ibverbs_dep.found() and cpp.has_header_symbol(
    'infiniband/verbs.h', 'IBV_ACCESS_RELAXED_ORDERING', dependencies: rest_ibverbs_dep)
if not (rest_curl_dep.found() and rest_ibverbs_relaxed_ordering and rest_mlx5_dep.found())
    warning('libcurl >= 7.68, libibverbs with IBV_ACCESS_RELAXED_ORDERING or libmlx5 not found, skipping REST plugin build')
    subdir_done()
endif

rest_deps = [
    nixl_infra,
    nixl_common_dep,
    dependency('asio', required: true),
    rest_curl_dep,
    rest_ibverbs_dep,
    rest_mlx5_dep,
    dl_dep,
]
rest_compile_defs = []

if 'REST' in static_plugins
    rest_backend_lib = static_library(
        'REST',
        rest_sources,
        dependencies: rest_deps,
        cpp_args: rest_compile_defs,
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: false,
        name_prefix: 'libplugin_')  # Custom prefix for plugin libraries
else
    rest_backend_lib = shared_library(
        'REST',
        rest_sources,
        dependencies: rest_deps,
        cpp_args: rest_compile_defs + ['-fPIC'],
        include_directories: [nixl_inc_dirs, utils_inc_dirs],
        install: true,
        name_prefix: 'libplugin_',  # Custom prefix for plugin libraries
        install_dir: plugin_install_dir,
        install_rpath: '$ORIGIN/..')
    if get_option('buildtype') == 'debug'
        run_command('sh', '-c',
            'echo "REST=' + rest_backend_lib.full_path() + '" >> ' + plugin_build_dir + '/pluginlist',
            check: true
        )
    endif
endif

if 'REST' in static_plugins
    rest_backend_interface = declare_dependency(link_whole: rest_backend_lib)
else
    rest_backend_interface = declare_dependency(link_with: rest_backend_lib)
endif

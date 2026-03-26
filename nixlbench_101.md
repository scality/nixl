# NIXL Bench 101

## Components overview

**NIXL (Nvidia's Inference Xfer Library)** [[github](https://github.com/ai-dynamo/nixl)]: A hardware-agnostic transfer library designed to move data between components in a distributed[<sup>*</sup>](#glossary) LLM inference system. It provides :

* An API to move KV cache across different transports: NVLink, RDMA (InfiniBand/RoCE), PCIe, NVMe
* Abstracts the underlying hardware to handle each transport the same way
* Optimized for low-latency, high-bandwidth transfer between GPUs.

[**nixlbench**](#nixl-benchmark): A benchmarking tool that measures the raw transfer performance relevant when reading/writing KV attention cache[<sup>*</sup>](#glossary).

**kvbench** [[tutorial](https://github.com/ai-dynamo/nixl/blob/main/benchmark/kvbench/docs/tutorial-gds.md)]: A frontend tool that automates the calculation of the block size and total size of KV cache based on a model's architecture, then feeds it to nixlbench.

**cuObject** [[doc](https://docs.nvidia.com/gpudirect-storage/cuobject/)]: Nvidia's GPU-Direct Storage library for object stores. It enables RDMA transfer between GPU memory and S3-compatible object storage, bypassing the CPU entirely.

## [NIXL Benchmark](https://github.com/ai-dynamo/nixl/blob/main/benchmark/nixlbench/README.md)

Ideal for evaluating high-performance data transfer in distributed environments. It uses ETCD[<sup>*</sup>](#glossary) for coordination and metadata exchange.

NIXL benchmark can be adapted to different communication backends, storage backends, communication patterns, use CPU or GPU memory support and so on. ETCD is only required for multi-process. For storage backends (ie: POSIX, OBJ), which run as a single process, it is optional.

For our goal, we will focus on a nixlbench with no communication backend, a **POSIX** storage backend and a pairwise communication pattern. We will only use **CPU memory transfers** and thus NIXL workers.

### Building

#### Install system dependencies (Ubuntu/Debian)

```sh
sudo apt-get update && sudo apt-get install -y \
  build-essential cmake ninja-build pkg-config \
  autotools-dev automake libtool libz-dev flex \
  libgtest-dev hwloc libhwloc-dev libgflags-dev \
  libgrpc-dev libgrpc++-dev libprotobuf-dev \
  libaio-dev liburing-dev protobuf-compiler-grpc \
  libcpprest-dev etcd-server etcd-client \
  pybind11-dev libclang-dev libcurl4-openssl-dev \
  libssl-dev uuid-dev libxml2-dev zlib1g-dev python3-dev python3-pip
```

#### etcd-cpp-api Installation

```sh
# Clone and build etcd-cpp-api
git clone --depth 1 https://github.com/etcd-cpp-apiv3/etcd-cpp-apiv3.git
cd etcd-cpp-apiv3

# Remove cpprestsdk dependency from CMake config (already installed via apt)
sed -i '/^find_dependency(cpprestsdk)$/d' etcd-cpp-api-config.in.cmake

# Build and install
mkdir build && cd build
cmake .. \
  -DBUILD_ETCD_CORE_ONLY=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local
make -j$(nproc) && sudo make install
sudo ldconfig
```

#### Building NIXL and NIXLBench

```sh
# You may adjust these paths to your preference
NIXL_SRC=~/nixl-src
NIXL_INSTALL=/usr/local/nixl
NIXLBENCH_INSTALL=/usr/local/nixlbench

git clone https://github.com/ai-dynamo/nixl.git $NIXL_SRC

# Build NIXL first
cd $NIXL_SRC
rm -rf build && mkdir build
meson setup build --prefix=$NIXL_INSTALL --buildtype=release
cd build && ninja && sudo ninja install

# Build NIXLBench
cd $NIXL_SRC/benchmark/nixlbench
rm -rf build && mkdir build
meson setup build \
  -Dnixl_path=$NIXL_INSTALL \
  -Dprefix=$NIXLBENCH_INSTALL \
  --buildtype=release
cd build && ninja && sudo ninja install
```

## Run

```sh
mkdir -p /tmp/nixlbench
cd $NIXL_SRC/benchmark/nixlbench/build
./nixlbench -backend POSIX -posix_api_type POSIXAIO -filepath /tmp/nixlbench
```

This command benchmarks raw POSIX async I/O performance of a local filesystem at `/tmp/nixlbench`, using the POSIX AIO API. It measures throughput and latency across block sizes from 4 KB to 64 MB, writing an 8 GB file from a DRAM staging buffer.

### Understanding the parameters

| Parameter | Value | What it does |
| --- | --- | --- |
| backend | POSIX | Use the filesystem backend - transfers go from CPU memory to a local file. |
| posix_api_type | POSIXAIO | Use the async POSIX I/O interface. |
| filepath | /tmp/nixlbench | Directory where nixlbench creates its test files. |

There are plenty of default parameters that were not explicitly set. The exhaustive list is available in nixlbench README, section [Command Line Options](https://github.com/ai-dynamo/nixl/blob/main/benchmark/nixlbench/README.md#command-line-options).

Parameters can be specified in a .config file (TOML format), specified using the `--config-file` command line parameter. An example can be found [here](https://github.com/ai-dynamo/nixl/blob/main/benchmark/nixlbench/README.md#configuration-file).

### Understanding the process

#### Startup

POSIX being a storage backend with no `--etcd_endpoints` specified (no inter-process communication is needed), nixlbench runs as a standalone single process.

#### NIXL Agent & Backend initialization

A NIXL agent is created, then a POSIX backend engine is created with the given and default parameters.

#### Memory & File allocation

A test file is created at `/tmp/nixlbench/nixlbench_posix_test_file_initiator_0`. A local DRAM buffer of 8 GB is allocated. Both are registered with the NIXL backend.

#### Benchmark loop

The benchmark then runs the same test repeatedly, each time with a different I/O block size, to see how performance changes.

Here, it starts at 4 KB and doubles each iteration.

There is first a warmup phase, then a measurement phase.

#### Clean up

After outputing the results, the memory is deregistered, the DRAM buffer is freed, and the file descriptor is closed.

## Error handling

### ETCD: Clear stale state from previous run

If a previous nixlbench command did not run successfully, you may encounter the following error :

```sh
Rank 1 is greater than or equal to global size 1
Failed to setup ETCD runtime
```

It means the previous run is still in etcd and you need to clear it before being able to start again.

```sh
etcdctl del "" --prefix
```

## Benchmark output

```sh
WARNING: Adjusting num_iter to 1008 to allow equal distribution to 1 threads
WARNING: Adjusting warmup_iter to 112 to allow equal distribution to 1 threads
Using null runtime for storage backend without ETCD
POSIX backend with API type: POSIXAIO
Single instance storage backend - no synchronization needed
Creating file: /tmp/nixlbench/nixlbench_posix_test_file_initiator_0
****************************************************************************************************************************************************************
NIXLBench Configuration
****************************************************************************************************************************************************************
Runtime (--runtime_type=[etcd])                             : ETCD
ETCD Endpoint                                               : disabled (storage backend)
Worker type (--worker_type=[nixl,nvshmem])                  : nixl
Backend (--backend=[UCX,GDS,GDS_MT,POSIX,Mooncake,HF3FS,OBJ,AZURE_BLOB]): POSIX
Enable pt (--enable_pt=[0,1])                               : 0
Progress threads (--progress_threads=N)                     : 0
Device list (--device_list=dev1,dev2,...)                   : all
Enable VMM (--enable_vmm=[0,1])                             : 0
Recreate xfer each iteration (--recreate_xfer=[0,1])        : 0
POSIX API type (--posix_api_type=[AIO,URING,POSIXAIO])      : POSIXAIO
POSIX IO pool size (--posix_ios_pool_size=N)                : 65536
POSIX kernel queue size (--posix_kernel_queue_size=N)       : 256
filepath (--filepath=path)                                  : /tmp/nixlbench
filenames (--filenames=filename1,filename2,...)             : 
Number of files (--num_files=N)                             : 1
Storage enable direct (--storage_enable_direct=[0,1])       : 0
Initiator seg type (--initiator_seg_type=[DRAM,VRAM])       : DRAM
Target seg type (--target_seg_type=[DRAM,VRAM])             : DRAM
Scheme (--scheme=[pairwise,manytoone,onetomany,tp])         : pairwise
Mode (--mode=[SG,MG])                                       : SG
Op type (--op_type=[READ,WRITE])                            : WRITE
Check consistency (--check_consistency=[0,1])               : 0
Total buffer size (--total_buffer_size=N)                   : 8589934592
Num initiator dev (--num_initiator_dev=N)                   : 1
Num target dev (--num_target_dev=N)                         : 1
Start block size (--start_block_size=N)                     : 4096
Max block size (--max_block_size=N)                         : 67108864
Start batch size (--start_batch_size=N)                     : 1
Max batch size (--max_batch_size=N)                         : 1
Num iter (--num_iter=N)                                     : 1008
Warmup iter (--warmup_iter=N)                               : 112
Large block iter factor (--large_blk_iter_ftr=N)            : 16
Num threads (--num_threads=N)                               : 1
----------------------------------------------------------------------------------------------------------------------------------------------------------------

Block Size (B)      Batch Size     B/W (GB/Sec)   Avg Lat. (us)  Avg Prep (us)  P99 Prep (us)  Avg Post (us)  P99 Post (us)  Avg Tx (us)    P99 Tx (us)    
----------------------------------------------------------------------------------------------------------------------------------------------------------------
4096                1              0.291436       14.1           24.0           24.0           2.7            3.0            11.4           18.0           
8192                1              0.718984       11.4           1.0            1.0            1.5            3.0            9.9            16.0           
16384               1              2.233880       7.3            1.0            1.0            0.6            1.0            6.7            10.0           
32768               1              4.183149       7.8            1.0            1.0            0.6            2.0            7.2            11.0           
65536               1              7.041924       9.3            0.0            0.0            0.6            2.0            8.7            14.0           
131072              1              10.258605      12.8           3.0            3.0            0.8            2.0            12.0           28.0           
262144              1              10.493672      25.0           7.0            7.0            1.0            4.0            23.9           45.0           
524288              1              22.275334      23.5           1.0            1.0            0.6            2.0            22.9           40.0           
1048576             1              24.248982      43.2           1.0            1.0            0.7            2.0            42.6           111.0          
2097152             1              25.835075      81.2           2.0            2.0            0.6            2.0            80.5           87.0           
4194304             1              25.946696      161.7          3.0            3.0            0.6            1.0            160.9          184.0          
8388608             1              7.490359       1119.9         4.0            4.0            2.0            10.0           1117.8         1898.0         
16777216            1              4.525394       3707.3         6.0            6.0            2.9            11.0           3704.2         4578.0         
33554432            1              5.225371       6421.4         5.0            5.0            2.0            14.0           6419.2         11103.0        
67108864            1              4.926833       13621.1        24.0           24.0           2.3            12.0           13618.2        20271.0        
```

### Understand the results

#### Categories

The Prep/Post/Tx breakdown map directly to the three phases of the NIXL transfer API. Therefore, the interpretation differs depending on the use of NIXL or NVSHMEM worker (GPU-focused).

| <div style="width:100px">Metric name</div> | Task | NIXL worker description | NVSHMEM worker description |
| --- | --- | --- | --- |
| Avg Prep (µs) | Request creation | Time spent in `agent->createXferReq()`. Builds the transfer descriptor (source/destination segment list, operation type, etc...). No data moves, nothing submitted to hardware or the OS. <br> It reflects how expensive it is to set up a transfer for a given backend. It tends to be cheap and roughly stable for most sizes. | Always 0. |
| Avg Post (µs) | Request submission | Time spent in `agent->postXferReq()`. Hands the request off to the backend. The call returns quickly even if the transfer is not complete yet. <br> It reflects the overhead of initiating the operation (a syscall, a network send, DMA descriptor enqueue...). | Always 0. |
| Avg Tx (µs) | Transfer completion wait | Time spent polling `agent->getXferStatus()` until the transfer finishes. Actual data movement. <br> It reflects the true cost of the I/O itself (storage latency, network round-trip, DMA transfer,...) | Full cost of `nvshmemx_put/getmem_on_stream` + `nvshmemx_quiet_on_stream` per iteration. Waits for all outstanding operations on the stream to complete. |
| Avg Lat (µs) | End-to-end per-operation latency | Measured with a separate outer timer spanning the entire iteration: `avg_latency = total_duration / (num_iter × batch_size)`. <br> Avg Lat ≈ Avg Post + Avg Tx. | `avg_latency = total_duration / (num_iter × batch_size)`. Should match Avg Tx. |

`B/W (GB/s)` measures the total bytes moved divided by the total elapsed time across all iterations.

There is no written definition of the timing categories in the code. The relevant code to interpret the categories can be found in :

* `src/worker/nixl/nixl_worker.cpp`: NIXL worker timers started and stopped
* `src/worker/nvshmem/nvshmem_worker.cpp`: NVSHMEM worker timers started and stopped
* `src/utils/utils.cpp`: timers aggregated and printed

#### P99

P99 is the 99th percentile. It means 99% of all observed values were at or below this value, only 1% were above. It measures tail latency: occasional slow outliers that might be hidden in the average, when the system occasionally stalls.

#### Observations

* **Avg Prep** is a single sample, not a distribution, as the request is created only once. Avg Prep and P99 Prep will always be identical. GUSLI backend, even though not relevant for our use, is an exception.
* **Bandwidth** is reported in GB/s not GiB/s.
* **Bandwidth** and **Avg Lat** are both computed from the same total elapsed time, so a noisy run will impact them both.
* **Avg Post** sometimes also measures the flush of dirty pages if the kernel decides the page cache holds too many dirty pages.
* There are **fewer iterations** at large block size. The `large_blk_iter_ftr` (default 16) divides both `num_iter` and `warmup_iter` for blocks above 1MB. You have less statistical confidence in those rows than in the small block rows.
* **Buffered vs direct I/O**. With `storage_enable_direct=0` (the default), writes go through the OS page cache (a RAM buffer the kernel maintains on behalf of the filesystem). It means the kernel acknowledges the write as soon as the data lands in RAM, without waiting for it to reach the storage device. Therefore, the benchmark measures memory bandwidth for small-to-medium block sizes, and only reaches actual storage bandwidth when the cache fills and the kernel is forced to flush dirty pages to disk.
* On many Linux systems `/tmp` is mounted as `tmpfs`, a filesystem backed entirely by RAM. If that is the case, the benchmark never touches a storage device regardless of the `storage_enable_direct` setting. You can check with `findmnt /tmp` - if the type is `tmpfs`, you are measuring RAM speed. To benchmark actual storage, point `--filepath` at a path on a real filesystem.
* **Page cache** between runs or between block-size rows is not reset.

## Glossary

**ETCD**: A distributed key-value store used as a coordination service between processes. When you run multiple nixlbench processes (ex: one initiator, one target), ETCD acts as the meeting point of those processes. Each registers itself, waits until all participants have checked in, then proceeds.

**Distributed environment**: An environment split across multiple machines (nodes) connected over a network.

**KV attention cache**: During inference, LLMs store the Key and Value vectors for each token and reuse all the previous ones for new token generation. Those two vectors, along with the Query, are used within the attention mechanism to decide which tokens are relevant when predicting the next one.


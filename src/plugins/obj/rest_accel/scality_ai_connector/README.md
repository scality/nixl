<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scality AI Connector

A plugin engine for the OBJ backend of
[NIXL](https://github.com/ai-dynamo/nixl), the NVIDIA Inference Xfer Library:
one API for moving data between HBM, DRAM, NVMe, file, and object storage, with
pluggable backends.

This engine connects NIXL to a Scality endpoint and splits each transfer into
two planes: the **data plane** is RDMA; the **control plane** is a header-only
HTTP request that names the operation and the object id. Object bytes never
travel over HTTP.

## Overview

The connector splits each transfer into two independent channels:

- **Data plane (RDMA):** the object bytes move directly between the local
  buffer and Scality over RDMA, zero-copy and GPUDirect-capable, so data in
  GPU memory never needs a staging copy through host memory.
- **Control plane (HTTP):** a header-only HTTP request tells Scality *what* to do
  (`PUT`/`GET` of an object). Its body is **empty**: no payload travels
  over HTTP. The RDMA "token" (a descriptor identifying the registered buffer)
  is passed to Scality in a custom `x-scal-rdma` header.

```text
  local buffer (DRAM/VRAM) ─RDMA──────────────────────────► Scality endpoint
                                                              ▲
  control: HTTP PUT/GET {endpoint}/{object_id}                │
           header  x-scal-rdma: <token>     ──────────────────┘
           body    (empty)
```

**Terms:**

- **RDMA** (Remote Direct Memory Access): network transfers straight into
  registered memory, bypassing CPU copies.
- **DC** (Dynamically Connected): the RDMA transport mode used here.
- **cuObject**: NVIDIA's GPUDirect Storage client library that provides the
  RDMA path.

## When to use it

| Use this engine when | Use `s3` / `s3_crt` instead when |
|---|---|
| The store is a **Scality AI Connector** (REST) endpoint | The store is **S3-compatible** |
| **RDMA-capable** hardware and the cuObject stack are available | Plain HTTP(S) data transfer is sufficient |
| Transfers are large buffers, often in **GPU memory** | RDMA / GPUDirect is not needed |

The Scality AI Connector speaks Scality's own REST dialect; it is **not** an S3
client and uses no AWS SDK or AWS authentication.

## Prerequisites

- **NVIDIA cuObject / GPUDirect Storage** installed and working. This provides
  the DC RDMA path and is mandatory. The engine uses the
  `CUOBJ_PROTO_RDMA_DC_V1` protocol and is developed against the CUDA 13.x
  cuObject stack.
- An **RDMA-capable fabric/NIC** reachable by the Scality endpoint.
- A reachable **Scality AI Connector REST endpoint**.
- Build dependencies: `cuobjclient` and `libcurl`.

## Building

The engine is part of NIXL's standard Meson build and is compiled in
automatically when both `cuobjclient` and `libcurl` are detected at configure
time:

```sh
meson setup build
ninja -C build
```

If `cuobjclient` is not found, the accelerated OBJ engines (including this one)
are skipped, and the engine reports `Accelerated engine support not available
(not compiled)` at run time.

## Configuration

The engine is selected and configured through the backend's `customParams`:

| Parameter | Required | Default | Meaning |
|---|---|---|---|
| `accelerated` | yes | n/a | Must be `"true"` to request an accelerated engine |
| `type` | yes | n/a | Must be `"scality_ai_connector"` to select this engine |
| `endpoint_override` | yes | n/a | Base URL of the connector, e.g. `http://10.0.0.1:10000` |
| `num_threads` | no | `max(2, cpu_threads / 4)` | Size of the callback worker pool (see [Concurrency model](#concurrency-model)). |
| `rdma_nics` | no | from `cufile.json` | NIC list for DRAM transfers (see [Multi-NIC DRAM](#multi-nic-dram)): comma-separated IPv4 addresses or device names, e.g. `10.10.40.208,10.10.48.208` or `mlx5_1,mlx5_2`. Empty falls back to `cufile.json` `rdma_dev_addr_list`. |
| `rdma_dc_key` | no | from `cufile.json`, else `0xffeeddcc` | DC access key (hex) the server's DCI side must present, for DRAM transfers. Empty falls back to `cufile.json` `rdma_dc_key`. |
| `max_inflight` | no | `min(512, RLIMIT_NOFILE / 4)` | Cap on requests running at once. Each holds a connection and therefore a descriptor, so the default is clamped to a share of `RLIMIT_NOFILE`; excess requests queue and start as slots free. `0` disables the cap. |
| `split_size` | no | `8388608` (8 MiB) | Bytes per object request. A transfer descriptor of any size is cut into requests of at most this, on boundaries aligned in the object's own offset space. `0` disables splitting. |
| `dram_rdma` | no | `true` | Whether DRAM transfers use RDMA. `false` makes DRAM reads plain HTTP with nothing pinned: no MR, and no NIC list needed at all. For callers whose host buffers hold metadata, where an `ibv_reg_mr` per rail costs more than the read itself. Reads only. VRAM is unaffected. |

Requests are issued to `{endpoint_override}/{object_id}`.

Example (parameters as passed to the OBJ backend):

```ini
accelerated       = true
type              = scality_ai_connector
endpoint_override = http://10.0.0.1:10000
num_threads       = 8            # optional: callback pool size
```

## Concurrency model

A single poller thread drives all HTTP requests through libcurl's multi
interface, so the in-flight request count does not depend on `num_threads` (the
bulk data moves over RDMA; curl only carries the header-only command). Each completed
request's callback runs in a worker pool sized by `num_threads`, so a slow
callback cannot stall the poller. Size `num_threads` according to the amount of
work each callback does. The default value is sufficient for lightweight callbacks.

## Multi-NIC DRAM

The RDMA descriptor carries the GID of the NIC the server will RDMA against. NVIDIA cuObject
picks that NIC from the buffer's GPU: VRAM on distinct GPUs spreads across NICs, but **host
(DRAM) memory has no GPU affinity and cuObject pins every host registration to a single
NIC** — so cuObject DRAM transfers cannot use more than one link.

This engine therefore routes **DRAM** through an in-process libibverbs DC client (VRAM and
OBJ stay on cuObject — a pure per-segment-type split, no flag to choose). It opens one DC
target context per NIC and round-robins memory regions across them, building the descriptor
itself with the chosen NIC's GID. The `x-scal-rdma` DC wire format and `dct_access_key`
match cuObject's, so the server is unchanged. To spread across all NICs, register at least
one buffer per NIC (e.g. nixlbench `--mode=MG --num_initiator_dev=<n>`, `n` ≥ NIC count).

The NIC list resolves from `rdma_nics` (IPv4 addresses or device names like `mlx5_1`), or,
when unset, from `cufile.json`'s `rdma_dev_addr_list` (`$CUFILE_ENV_PATH_JSON`, else
`/etc/cufile.json`) — the same file and key cuObject uses. The DC access key resolves the
same way: `rdma_dc_key`, else `cufile.json`'s `rdma_dc_key`, else `0xffeeddcc`. If no NICs
resolve — or any fails to open — a DRAM registration fails with `NIXL_ERR_BACKEND` rather
than silently falling back to single-NIC cuObject.

Bonded (LAG) NICs are detected from sysfs; one DCT QP is created per physical port and
regions round-robin across them (the distinct DCT QP numbers feed the hardware bond hash —
DCT QPs cannot be pinned to a physical port directly). Requires libibverbs + libmlx5 at
build time; otherwise a DRAM transfer fails at startup with a clear error.

## How a transfer works

1. **Register memory.** Local **DRAM or VRAM (GPU)** buffers are registered for
   RDMA; remote **object** segments are mapped to their object ids.
2. **Prepare.** For each piece of the transfer the connector obtains an RDMA
   token for the buffer (no data moves yet).
3. **Post.** It issues the HTTP `PUT`/`GET` for the object id, carrying the
   token in the `x-scal-rdma` header, and returns immediately; the request is
   in progress. Scality performs the actual RDMA transfer against the buffer.
4. **Poll.** The transfer is checked for completion until all parts succeed (or
   one reports an error).

## Limitations

- **DC transport only.** Both the cuObject (VRAM/OBJ) and libibverbs (DRAM) paths use RDMA
  DC; there is no RC fallback.
- **4 GiB** maximum per memory registration (cuObject limit; the DC descriptor SIZE field
  is also 32-bit).
- **DRAM requires libibverbs/libmlx5 and a resolvable NIC list.** VRAM stays on cuObject
  (GPUDirect peer-memory MR registration is not handled by the ibverbs client); there is no
  cuObject DRAM path.
- The **cuObject stack is required** at both build and run time; without it the
  engine does not exist.

## Troubleshooting

| Symptom (in logs) | Potential cause / fix |
|---|---|
| `'endpoint_override' parameter is required` | Set `endpoint_override` in `customParams`. |
| `RDMA token client failed to connect` | cuObject / GPUDirect Storage or the RDMA fabric is not ready on this host. |
| `<op>: failed url=<url> curl_code=<n> http_code=<n>` | The HTTP call reached the endpoint but failed; check the object id, endpoint URL. A non-2xx `http_code` comes from Scality. |
| `Accelerated engine support not available (not compiled)` | cuObject was not detected at build time, so the connector was not compiled in. |

## For contributors

Source layout (see the code for details):

| File | Role |
|---|---|
| `engine_impl.{h,cpp}` | `ScalityObjEngineImpl`: the engine (register / prepare / post / check). |
| `client.{h,cpp}` | `RestClient`: the libcurl HTTP control-plane client, an event-driven `curl_multi` poller with a callback worker pool (behind the `iRestClient` interface, which is also the test-injection seam). See [Concurrency model](#concurrency-model). |
| `rdma_token_client.h` | `iRdmaTokenClient`: the data-plane interface and the `getCtx()` helper. |
| `rdma_ctx.h` | `rdma_ctx_t`: the per-transfer context the token client fills with the descriptor. |
| `cuobj_rdma_token_client.{h,cpp}` | `CuObjRdmaTokenClient`: the DC RDMA implementation, a thin wrapper over NVIDIA `cuObjClient` (`CUOBJ_PROTO_RDMA_DC_V1`). |
| `ibverbs_dc_rdma_token_client.{h,cpp}` | `IbverbsDcRdmaTokenClient`: in-process libibverbs DC client for DRAM multi-NIC spreading (see [Multi-NIC DRAM](#multi-nic-dram)). |
| `dc_descriptor.h` | libibverbs-free DC descriptor formatting + bond-slave parsing (unit-tested). |
| `cufile_nics.h` | libibverbs-free `cufile.json` `rdma_dev_addr_list` extractor (unit-tested). |

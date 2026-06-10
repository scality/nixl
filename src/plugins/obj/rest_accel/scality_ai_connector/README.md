<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Scality AI Connector

A NIXL **OBJ-plugin engine** that moves object data to and from a Scality
endpoint over **RDMA**, using a small HTTP request only to carry the command.

## Overview

The connector splits each transfer into two independent channels:

- **Data plane (RDMA):** the object bytes move directly between the local
  buffer and Scality over RDMA, zero-copy and GPU-direct capable, so data in
  GPU memory never needs a staging copy through host memory.
- **Control plane (HTTP):** a tiny HTTP request tells Scality *what* to do
  (`PUT`/`GET` of an object key). Its body is **empty**: no payload travels
  over HTTP. The RDMA "token" (a descriptor identifying the registered buffer)
  is passed to Scality in a custom `x-scal-rdma` header.

```
  local buffer (DRAM/VRAM) ─RDMA──────────────────────────► Scality endpoint
                                                              ▲
  control: HTTP PUT/GET {endpoint}/{key}                      │
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
  the DC RDMA path and is mandatory.
- An **RDMA-capable fabric/NIC** reachable by the Scality endpoint.
- A reachable **Scality AI Connector REST endpoint**.
- Build dependencies: `cuobjclient` and `libcurl`.

## Configuration

The engine is selected and configured through the backend's `customParams`:

| Parameter | Required | Default | Meaning |
|---|---|---|---|
| `accelerated` | yes | n/a | Must be `"true"` to request an accelerated engine |
| `type` | yes | n/a | Must be `"scality_ai_connector"` to select this engine |
| `endpoint_override` | yes | n/a | Base URL of the connector, e.g. `http://10.0.0.1:81` |
| `num_threads` | no | `max(2, cpu_threads / 4)` | Size of the callback worker pool (see [Concurrency model](#concurrency-model)). |

Requests are issued to `{endpoint_override}/{key}`.

Example (parameters as passed to the OBJ backend):

```
accelerated       = true
type              = scality_ai_connector
endpoint_override = http://10.0.0.1:81
num_threads       = 8            # optional: callback pool size
```

## Concurrency model

A single poller thread drives all HTTP requests through libcurl's multi
interface, so the in-flight request count does not depend on `num_threads` (the
bulk data moves over RDMA; curl only carries the tiny command). Each completed
request's callback runs in a worker pool sized by `num_threads`, so a slow
callback cannot stall the poller. Size `num_threads` to how much work the
callbacks do; the default suits cheap callbacks.

## How a transfer works

1. **Register memory.** Local **DRAM or VRAM (GPU)** buffers are registered for
   RDMA; remote **object** segments are mapped to their object keys.
2. **Prepare.** For each piece of the transfer the connector obtains an RDMA
   token for the buffer (no data moves yet).
3. **Post.** It issues the HTTP `PUT`/`GET` for the object key, carrying the
   token in the `x-scal-rdma` header, and returns immediately; the request is
   in progress. Scality performs the actual RDMA transfer against the buffer.
4. **Poll.** The transfer is checked for completion until all parts succeed (or
   one reports an error).

## Limitations

- **DC transport only.** There is no fallback to other RDMA transports.
- **4 GiB** maximum per memory registration (a cuObject limit).
- Data is **never** sent over HTTP; the HTTP body is always empty by design.
- The **cuObject stack is required** at both build and run time; without it the
  engine does not exist.

## Troubleshooting

| Symptom (in logs) | Likely cause / fix |
|---|---|
| `'endpoint_override' parameter is required` | Set `endpoint_override` in `customParams`. |
| `RDMA token client failed to connect` | cuObject / GPUDirect Storage or the RDMA fabric is not ready on this host. |
| `<op>: failed url=<url> curl_code=<n> http_code=<n>` | The HTTP call reached the endpoint but failed; check the object key, endpoint URL, and permissions. A non-2xx `http_code` comes from Scality. |
| `Accelerated engine support not available (not compiled)` | cuObject was not detected at build time, so the connector was not compiled in. |

## For contributors

Source layout (see the code for details):

| File | Role |
|---|---|
| `engine_impl.{h,cpp}` | `ScalityObjEngineImpl`: the engine (register / prepare / post / check). |
| `client.{h,cpp}` | `RestClient`: the libcurl HTTP control-plane client, an event-driven `curl_multi` poller with a callback worker pool (behind the `iRestClient` interface, which is also the test-injection seam). See [Concurrency model](#concurrency-model). |
| `rdma_token_client.h` | `iRdmaTokenClient`: the data-plane interface and the `getCtx()` helper. |
| `cuobj_rdma_token_client.{h,cpp}` | `CuObjRdmaTokenClient`: the DC RDMA implementation, a thin wrapper over NVIDIA `cuObjClient` (`CUOBJ_PROTO_RDMA_DC_V1`). |

This is a **DC-only** build. The `iRdmaTokenClient` abstraction is the seam
where an RDMA implementation plugs in (and the home of `getCtx()`);
`CuObjRdmaTokenClient` is the only implementation today.

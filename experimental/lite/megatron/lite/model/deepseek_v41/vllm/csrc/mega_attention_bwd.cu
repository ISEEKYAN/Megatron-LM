// Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
// Sparse MLA backward with attention sink for the DS4.1 mega attention core
// (64 heads, K = V, d = 512), on rotated operands. One CTA per query row holds
// all 64 heads and walks the row's index list in blocks of 32 KV slots twice:
//   pass 1: S = Q K^T, dP = dO K^T, P = exp(S - LSE), D = sum_j P dP;
//   pass 2: dS = scale * P (dP - D); dQ += dS K; dK_j += dS^T Q + P^T dO.
// bf16 mma.sync (m16n8k16) with FP32 accumulation; dK is added to the FP32
// dKV buffer with atomics. D is reduced in a fixed order and written out so
// the caller forms dsink = -sum_rows exp(sink - LSE) * D.
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

namespace {

constexpr int H = 64;
constexpr int D = 512;
constexpr int B = 32;  // KV slots per block
constexpr int THREADS = 512;

struct Smem {
  __nv_bfloat16 q[H * D];
  __nv_bfloat16 dout[H * D];
  __nv_bfloat16 k[B * D];
  __nv_bfloat16 p[H * B];
  __nv_bfloat16 ds[H * B];
  float dpart[4][H];
  float delta[H];
  float lse2[H];
  int idx[B];
};

// 1024-byte rows of 64 16-byte chunks, chunk XOR-swizzled by row so that the
// 8 rows an ldmatrix phase reads hit distinct bank groups.
__device__ __forceinline__ int wide(int r, int c) {
  return r * D + ((((c >> 3) ^ (r & 7))) << 3) + (c & 7);
}
// 64-byte rows of 4 chunks.
__device__ __forceinline__ int narrow(int r, int c) {
  return r * B + ((((c >> 3) ^ ((r >> 1) & 3))) << 3) + (c & 7);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ldsm_x4(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t (&r)[4], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void ldsm_x2(uint32_t (&r)[2], const void* p) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
               : "=r"(r[0]), "=r"(r[1])
               : "r"(smem_u32(p)));
}
__device__ __forceinline__ void mma(float (&c)[4], const uint32_t (&a)[4], uint32_t b0,
                                    uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, "
      "{%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(smem_u32(dst)), "l"(src));
}
__device__ __forceinline__ void cp_async_wait_all() {
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
  return *reinterpret_cast<uint32_t*>(&v);
}

// S and dP tiles of this warp: heads [16 mt, +16) x slots [8 nt, +8), K = 512.
__device__ __forceinline__ void score_tiles(const Smem& s, int mt, int nt, int lane,
                                            float (&acc_s)[4], float (&acc_dp)[4]) {
#pragma unroll
  for (int i = 0; i < 4; ++i) acc_s[i] = acc_dp[i] = 0.f;
  const int arow = 16 * mt + (lane & 15);
  const int brow = 8 * nt + (lane & 7);
#pragma unroll 4
  for (int k = 0; k < D; k += 16) {
    uint32_t a[4], g[4], b[2];
    ldsm_x4(a, &s.q[wide(arow, k + (lane >> 4) * 8)]);
    ldsm_x4(g, &s.dout[wide(arow, k + (lane >> 4) * 8)]);
    ldsm_x2(b, &s.k[wide(brow, k + ((lane >> 3) & 1) * 8)]);
    mma(acc_s, a, b[0], b[1]);
    mma(acc_dp, g, b[0], b[1]);
  }
}

__global__ void __launch_bounds__(THREADS, 1)
    mega_attention_bwd_kernel(const __nv_bfloat16* __restrict__ qr,
                              const __nv_bfloat16* __restrict__ dout,
                              const __nv_bfloat16* __restrict__ kv, int64_t kv_rows,
                              const int* __restrict__ indices, int64_t index_stride,
                              const int* __restrict__ lens, const float* __restrict__ lse2,
                              float scale, float scale_log2, float* __restrict__ dq,
                              float* __restrict__ dkv, float* __restrict__ delta_out) {
  extern __shared__ __align__(128) unsigned char smem_raw[];
  Smem& s = *reinterpret_cast<Smem*>(smem_raw);
  const int row = blockIdx.x;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int len = lens[row];
  const int nblocks = (len + B - 1) / B;

  {
    const __nv_bfloat16* qrow = qr + (int64_t)row * H * D;
    const __nv_bfloat16* grow = dout + (int64_t)row * H * D;
    for (int c = tid; c < H * D / 8; c += THREADS) {
      const int r = c / (D / 8), col = (c % (D / 8)) * 8;
      cp_async16(&s.q[wide(r, col)], qrow + r * D + col);
      cp_async16(&s.dout[wide(r, col)], grow + r * D + col);
    }
    if (tid < H) s.lse2[tid] = lse2[(int64_t)row * H + tid];
    cp_async_wait_all();
    __syncthreads();
  }

  auto load_kv = [&](int block) {
    if (tid < B) {
      const int slot = block * B + tid;
      int id = slot < len ? indices[(int64_t)row * index_stride + slot] : -1;
      s.idx[tid] = (id >= 0 && id < kv_rows) ? id : -1;
    }
    __syncthreads();
    for (int c = tid; c < B * D / 8; c += THREADS) {
      const int r = c / (D / 8), col = (c % (D / 8)) * 8;
      const int id = s.idx[r];
      if (id >= 0) {
        cp_async16(&s.k[wide(r, col)], kv + (int64_t)id * D + col);
      } else {
        *reinterpret_cast<uint4*>(&s.k[wide(r, col)]) = make_uint4(0, 0, 0, 0);
      }
    }
    cp_async_wait_all();
    __syncthreads();
  };

  const int mt = warp >> 2, nt = warp & 3;
  const int h0 = 16 * mt + (lane >> 2), h1 = h0 + 8;
  const int c0 = 8 * nt + 2 * (lane & 3);

  // ---- pass 1: D = sum_j P_j dP_j ----
  float dsum0 = 0.f, dsum1 = 0.f;
  for (int block = 0; block < nblocks; ++block) {
    load_kv(block);
    float acc_s[4], acc_dp[4];
    score_tiles(s, mt, nt, lane, acc_s, acc_dp);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int h = i < 2 ? h0 : h1, slot = c0 + (i & 1);
      const float p = s.idx[slot] >= 0 ? exp2f(acc_s[i] * scale_log2 - s.lse2[h]) : 0.f;
      (i < 2 ? dsum0 : dsum1) += p * acc_dp[i];
    }
    __syncthreads();
  }
#pragma unroll
  for (int off = 1; off < 4; off <<= 1) {
    dsum0 += __shfl_xor_sync(0xffffffff, dsum0, off);
    dsum1 += __shfl_xor_sync(0xffffffff, dsum1, off);
  }
  if ((lane & 3) == 0) {
    s.dpart[nt][h0] = dsum0;
    s.dpart[nt][h1] = dsum1;
  }
  __syncthreads();
  if (tid < H) {
    const float d = ((s.dpart[0][tid] + s.dpart[1][tid]) + s.dpart[2][tid]) + s.dpart[3][tid];
    s.delta[tid] = d;
    delta_out[(int64_t)row * H + tid] = d;
  }
  __syncthreads();

  // ---- pass 2 ----
  // dQ: this warp owns dims [32 warp, +32) of all 64 heads.
  float acc_q[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b)
#pragma unroll
      for (int i = 0; i < 4; ++i) acc_q[a][b][i] = 0.f;
  const int dcol = 32 * warp;

  for (int block = 0; block < nblocks; ++block) {
    load_kv(block);
    {
      float acc_s[4], acc_dp[4];
      score_tiles(s, mt, nt, lane, acc_s, acc_dp);
      float pv[4], dsv[4];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int h = i < 2 ? h0 : h1, slot = c0 + (i & 1);
        const bool valid = s.idx[slot] >= 0;
        const float p = valid ? exp2f(acc_s[i] * scale_log2 - s.lse2[h]) : 0.f;
        pv[i] = p;
        dsv[i] = valid ? scale * p * (acc_dp[i] - s.delta[h]) : 0.f;
      }
      *reinterpret_cast<uint32_t*>(&s.p[narrow(h0, c0)]) = pack_bf16(pv[0], pv[1]);
      *reinterpret_cast<uint32_t*>(&s.p[narrow(h1, c0)]) = pack_bf16(pv[2], pv[3]);
      *reinterpret_cast<uint32_t*>(&s.ds[narrow(h0, c0)]) = pack_bf16(dsv[0], dsv[1]);
      *reinterpret_cast<uint32_t*>(&s.ds[narrow(h1, c0)]) = pack_bf16(dsv[2], dsv[3]);
    }
    __syncthreads();

    // dQ[64 x 32 dims] += dS[64 x 32 slots] . K[32 slots x 32 dims]
#pragma unroll
    for (int ks = 0; ks < B; ks += 16) {
      uint32_t bfrag[4][2];
#pragma unroll
      for (int n2 = 0; n2 < 2; ++n2) {
        // x4.trans: matrices (k0-7, n), (k8-15, n), (k0-7, n+8), (k8-15, n+8)
        uint32_t r[4];
        const int krow = ks + (lane & 7) + ((lane >> 3) & 1) * 8;
        const int ncol = dcol + 16 * n2 + (lane >> 4) * 8;
        ldsm_x4_t(r, &s.k[wide(krow, ncol)]);
        bfrag[2 * n2][0] = r[0];
        bfrag[2 * n2][1] = r[1];
        bfrag[2 * n2 + 1][0] = r[2];
        bfrag[2 * n2 + 1][1] = r[3];
      }
#pragma unroll
      for (int m = 0; m < 4; ++m) {
        uint32_t a[4];
        ldsm_x4(a, &s.ds[narrow(16 * m + (lane & 15), ks + (lane >> 4) * 8)]);
#pragma unroll
        for (int n = 0; n < 4; ++n) mma(acc_q[m][n], a, bfrag[n][0], bfrag[n][1]);
      }
    }

    // dK[32 slots x 32 dims] = dS^T . Q + P^T . dO over the 64 heads.
    float acc_k[2][4][4];
#pragma unroll
    for (int a = 0; a < 2; ++a)
#pragma unroll
      for (int b = 0; b < 4; ++b)
#pragma unroll
        for (int i = 0; i < 4; ++i) acc_k[a][b][i] = 0.f;
#pragma unroll
    for (int part = 0; part < 2; ++part) {
      const __nv_bfloat16* left = part == 0 ? s.ds : s.p;
      const __nv_bfloat16* right = part == 0 ? s.q : s.dout;
#pragma unroll
      for (int ks = 0; ks < H; ks += 16) {
        uint32_t bfrag[4][2];
#pragma unroll
        for (int n2 = 0; n2 < 2; ++n2) {
          uint32_t r[4];
          const int krow = ks + (lane & 7) + ((lane >> 3) & 1) * 8;
          const int ncol = dcol + 16 * n2 + (lane >> 4) * 8;
          ldsm_x4_t(r, &right[wide(krow, ncol)]);
          bfrag[2 * n2][0] = r[0];
          bfrag[2 * n2][1] = r[1];
          bfrag[2 * n2 + 1][0] = r[2];
          bfrag[2 * n2 + 1][1] = r[3];
        }
#pragma unroll
        for (int m = 0; m < 2; ++m) {
          // A[m = slot][k = head] from left[head][slot]: x4.trans of
          // (k0-7, m0-7), (k0-7, m8-15), (k8-15, m0-7), (k8-15, m8-15).
          uint32_t a[4];
          const int g = lane >> 3;
          const int hrow = ks + (lane & 7) + (g >> 1) * 8;
          const int scol = 16 * m + (g & 1) * 8;
          ldsm_x4_t(a, &left[narrow(hrow, scol)]);
#pragma unroll
          for (int n = 0; n < 4; ++n) mma(acc_k[m][n], a, bfrag[n][0], bfrag[n][1]);
        }
      }
    }
#pragma unroll
    for (int m = 0; m < 2; ++m) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int slot = 16 * m + (lane >> 2) + 8 * half;
        const int id = s.idx[slot];
        if (id < 0) continue;
        float* dst = dkv + (int64_t)id * D + dcol + 2 * (lane & 3);
#pragma unroll
        for (int n = 0; n < 4; ++n) {
          atomicAdd(reinterpret_cast<float2*>(dst + 8 * n),
                    make_float2(acc_k[m][n][2 * half], acc_k[m][n][2 * half + 1]));
        }
      }
    }
    __syncthreads();
  }

  float* qrow = dq + (int64_t)row * H * D;
#pragma unroll
  for (int m = 0; m < 4; ++m)
#pragma unroll
    for (int n = 0; n < 4; ++n)
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int h = 16 * m + (lane >> 2) + 8 * half;
        const int d = dcol + 8 * n + 2 * (lane & 3);
        *reinterpret_cast<float2*>(qrow + h * D + d) =
            make_float2(acc_q[m][n][2 * half], acc_q[m][n][2 * half + 1]);
      }
}

}  // namespace

void backward(torch::Tensor qr, torch::Tensor dout, torch::Tensor kv, torch::Tensor indices,
              torch::Tensor lens, torch::Tensor lse2, double scale, torch::Tensor dq,
              torch::Tensor dkv, torch::Tensor delta) {
  TORCH_CHECK(qr.is_cuda() && qr.scalar_type() == at::kBFloat16 && qr.is_contiguous());
  TORCH_CHECK(qr.dim() == 3 && qr.size(1) == H && qr.size(2) == D);
  TORCH_CHECK(dout.sizes() == qr.sizes() && dout.scalar_type() == at::kBFloat16 &&
              dout.is_contiguous());
  TORCH_CHECK(kv.dim() == 2 && kv.size(1) == D && kv.scalar_type() == at::kBFloat16 &&
              kv.is_contiguous());
  TORCH_CHECK(indices.dim() == 2 && indices.size(0) == qr.size(0) &&
              indices.scalar_type() == at::kInt && indices.stride(1) == 1);
  TORCH_CHECK(lens.numel() == qr.size(0) && lens.scalar_type() == at::kInt);
  TORCH_CHECK(lse2.sizes() == at::IntArrayRef({qr.size(0), H}) &&
              lse2.scalar_type() == at::kFloat && lse2.is_contiguous());
  TORCH_CHECK(dq.sizes() == qr.sizes() && dq.scalar_type() == at::kFloat && dq.is_contiguous());
  TORCH_CHECK(dkv.sizes() == kv.sizes() && dkv.scalar_type() == at::kFloat &&
              dkv.is_contiguous());
  TORCH_CHECK(delta.sizes() == lse2.sizes() && delta.scalar_type() == at::kFloat &&
              delta.is_contiguous());
  const int rows = qr.size(0);
  if (rows == 0) return;
  const size_t smem = sizeof(Smem);
  cudaFuncSetAttribute(mega_attention_bwd_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       (int)smem);
  const float scale_f = static_cast<float>(scale);
  const float scale_log2 = static_cast<float>(scale * 1.4426950408889634);
  mega_attention_bwd_kernel<<<rows, THREADS, smem, c10::cuda::getCurrentCUDAStream()>>>(
      reinterpret_cast<const __nv_bfloat16*>(qr.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(dout.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(kv.data_ptr()), kv.size(0),
      indices.data_ptr<int>(), indices.stride(0), lens.data_ptr<int>(), lse2.data_ptr<float>(),
      scale_f, scale_log2, dq.data_ptr<float>(), dkv.data_ptr<float>(),
      delta.data_ptr<float>());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("backward", &backward); }

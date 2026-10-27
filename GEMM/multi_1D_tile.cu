#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ============================================================
// 1D Thread Tile Matmul
//
// A: [M, K]
// B: [K, N]
// C: [M, N]
//
// BM = 64, BN = 64, BK = 8, TM = 8
//
// 每个 Block 计算 C 的 64x64 子块
// 每个线程计算同一列中连续的 8 个 C 元素
// ============================================================

template<int BM, int BN, int BK, int TM>
__global__ void matmul_1d_tile(const float* A,const float* B,float* C,int M,int N,int K) {
    // Shared Memory
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];

    // 当前 Block 在 C 中的 Tile 坐标
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // 当前线程的一维编号
    int tid = threadIdx.x;

    // --------------------------------------------------------
    // 1. 计算线程负责的输出位置
    // --------------------------------------------------------

    // 每个线程负责 TM 行、1 列
    int threadRow = tid / BN;
    int threadCol = tid % BN;

    // 当前 Block 对应的 C 子块起点
    int cRow = blockRow * BM;
    int cCol = blockCol * BN;

    // 每个线程使用 TM 个累加器
    float threadResults[TM] = {0.0f};

    // --------------------------------------------------------
    // 2. 计算线程负责加载的 Shared Memory 坐标
    // --------------------------------------------------------

    // As 的大小为 BM * BK = 64 * 8 = 512
    int innerRowA = tid / BK;
    int innerColA = tid % BK;

    // Bs 的大小为 BK * BN = 8 * 64 = 512
    int innerRowB = tid / BN;
    int innerColB = tid % BN;

    // --------------------------------------------------------
    // 3. 沿 K 方向逐 Tile 计算
    // --------------------------------------------------------

    for (int bkIdx = 0; bkIdx < K; bkIdx += BK) {

        // ---------------------------
        // 3.1 加载 A 到 Shared Memory
        // ---------------------------

        int globalRowA = cRow + innerRowA;
        int globalColA = bkIdx + innerColA;

        if (globalRowA < M && globalColA < K) {
            As[innerRowA][innerColA] =
                A[globalRowA * K + globalColA];
        } else {
            As[innerRowA][innerColA] = 0.0f;
        }

        // ---------------------------
        // 3.2 加载 B 到 Shared Memory
        // ---------------------------

        int globalRowB = bkIdx + innerRowB;
        int globalColB = cCol + innerColB;

        if (globalRowB < K && globalColB < N) {
            Bs[innerRowB][innerColB] =
                B[globalRowB * N + globalColB];
        } else {
            Bs[innerRowB][innerColB] = 0.0f;
        }

        // 等待所有线程完成 As、Bs 的加载
        __syncthreads();

        // ---------------------------
        // 3.3 使用 Shared Memory 计算
        // ---------------------------

        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {

            // 当前线程从 Bs 读取一个元素
            float Btmp = Bs[dotIdx][threadCol];

            // 同一个 Btmp 参与 TM 次乘加
            for (int resIdx = 0; resIdx < TM; ++resIdx) {

                int row = threadRow * TM + resIdx;

                float Atmp = As[row][dotIdx];

                threadResults[resIdx] += Atmp * Btmp;
            }
        }

        // 防止下一轮加载覆盖尚未使用完的 As、Bs
        __syncthreads();
    }

    // --------------------------------------------------------
    // 4. 将 TM 个结果写回 Global Memory
    // --------------------------------------------------------

    int globalColC = cCol + threadCol;

    for (int resIdx = 0; resIdx < TM; ++resIdx) {

        int globalRowC =
            cRow + threadRow * TM + resIdx;

        if (globalRowC < M && globalColC < N) {
            C[globalRowC * N + globalColC] =
                threadResults[resIdx];
        }
    }
}


// ============================================================
// CPU 参考实现
// ============================================================

void matmul_cpu(const float* A,const float* B,float* C,int M,int N,int K) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {

            float sum = 0.0f;

            for (int k = 0; k < K; ++k) {
                sum += A[row * K + k] * B[k * N + col];
            }

            C[row * N + col] = sum;
        }
    }
}


// ============================================================
// CUDA 错误检查
// ============================================================

#define CUDA_CHECK(call)                                      \
    do {                                                      \
        cudaError_t err = (call);                             \
        if (err != cudaSuccess) {                             \
            printf("CUDA Error: %s\n",                         \
                   cudaGetErrorString(err));                  \
            std::exit(EXIT_FAILURE);                          \
        }                                                     \
    } while (0)


// ============================================================
// main
// ============================================================

int main() {

    // 故意使用非整除尺寸，验证边界处理
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;

    // 当前实现要求：
    // 线程数 = BM * BN / TM
    // 同时恰好覆盖 As 和 Bs 的加载
    static_assert(BM * BN / TM == BM * BK,
                  "Thread count must equal As element count");
    static_assert(BM * BN / TM == BK * BN,
                  "Thread count must equal Bs element count");
    static_assert(BM % TM == 0,
                  "BM must be divisible by TM");
    static_assert(BM * BN / TM <= 1024,
                  "Too many threads per block");

    size_t sizeA = sizeof(float) * M * K;
    size_t sizeB = sizeof(float) * K * N;
    size_t sizeC = sizeof(float) * M * N;

    float* h_A = new float[M * K];
    float* h_B = new float[K * N];
    float* h_C = new float[M * N];
    float* h_ref = new float[M * N];

    // 初始化数据
    for (int i = 0; i < M * K; ++i) {
        h_A[i] = static_cast<float>(i % 7) * 0.1f;
    }

    for (int i = 0; i < K * N; ++i) {
        h_B[i] = static_cast<float>(i % 5) * 0.1f;
    }

    float *d_A = nullptr;
    float *d_B = nullptr;
    float *d_C = nullptr;

    CUDA_CHECK(cudaMalloc(&d_A, sizeA));
    CUDA_CHECK(cudaMalloc(&d_B, sizeB));
    CUDA_CHECK(cudaMalloc(&d_C, sizeC));

    CUDA_CHECK(cudaMemcpy(
        d_A, h_A, sizeA, cudaMemcpyHostToDevice
    ));

    CUDA_CHECK(cudaMemcpy(
        d_B, h_B, sizeB, cudaMemcpyHostToDevice
    ));

    // 每个 Block 负责 64x64 的 C 子块
    dim3 block(BM * BN / TM);

    // grid.x 对应列，grid.y 对应行
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    matmul_1d_tile<BM, BN, BK, TM>
        <<<grid, block>>>(d_A, d_B, d_C, M, N, K);

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(
        h_C, d_C, sizeC, cudaMemcpyDeviceToHost
    ));

    // CPU 参考结果
    matmul_cpu(h_A, h_B, h_ref, M, N, K);

    // 验证结果
    int errors = 0;
    float maxError = 0.0f;

    for (int i = 0; i < M * N; ++i) {

        float diff = std::fabs(h_C[i] - h_ref[i]);

        float tolerance =
            1e-4f + 1e-4f * std::fabs(h_ref[i]);

        if (diff > maxError) {
            maxError = diff;
        }

        if (diff > tolerance) {
            ++errors;
        }
    }

    printf("M=%d, N=%d, K=%d\n", M, N, K);
    printf("Block threads: %d\n", BM * BN / TM);
    printf("Grid: (%u, %u)\n", grid.x, grid.y);
    printf("Max error: %.8f\n", maxError);

    if (errors == 0) {
        printf("success!\n");
    } else {
        printf("failed! errors=%d\n", errors);
    }

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    delete[] h_ref;

    return errors == 0 ? 0 : 1;
}

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#define LOAD_FLOAT4(pointer) \
    (*reinterpret_cast<const float4 *>(&(pointer)))

#define STORE_FLOAT4(pointer) \
    (*reinterpret_cast<float4 *>(&(pointer)))
// ============================================================
// float4  Matmul
//
// A: [M, K]
// B: [K, N]
// C: [M, N]
//
// BM = 64, BN = 64, BK = 8, TM = 8,TN=8;
//
// 每个 Block 计算 C 的 64x64 子块
// 每个线程计算 8x8 个 C 元素
// ============================================================
template<int BM,int BN,int BK,int TM,int TN>
__global__ void matmul_float4(const float* A,const float* B,float* C,int M,int N,int K){
    static_assert(BM>0&&BN>0&&BK>0);
    static_assert(TM>0&&TN>0);
    static_assert(BM % TM == 0 && BN % TN==0,
                  "BM must be divisible by TM");
    constexpr int threadNums=(BM*BN)/(TM*TN);
    static_assert(BM * BN / (TM*TN) <= 1024,
                  "Too many threads per block");

    // Shared Memory
    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    //每线程处理的子矩阵，存放在寄存器中的数据
    float reg_m[TM]={0.0};
    float reg_n[TN]={0.0};
    // 当前 Block 在 C 中的 Tile 坐标
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    // 当前线程的一维编号
    int tid = threadIdx.x;
    // --------------------------------------------------------
    // 1. 计算线程负责的输出位置
    // --------------------------------------------------------

    // 每个线程负责的子块位置
    int threadRow = tid / (BN/TN);
    int threadCol = tid % (BN/TN);

    // 当前 Block 对应的 C 子块起点
    int cRow = blockRow * BM;
    int cCol = blockCol * BN;

    // 每个线程使用 TM*TN 个累加器
    float threadResults[TM*TN] = {0.0f};
    
    //每个线程一次搬四个，要搬多少轮
    //const int ldg_a_num=BM*BK/(4*threadNums);
    //const int ldg_b_num=BM*BK/(4*threadNums);
    //每轮每个线程要搬运的行数和列数
    //const int stride_a=BM/ldg_a_num;
    //const int stride_b=BK/ldg_b_num;
   // --------------------------------------------------------
    // 2. 沿 K 方向逐 Tile 计算
    // --------------------------------------------------------
    
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK){
           // ---------------------------
        // 2.1 协作加载 A 到 Shared Memory
        // ---------------------------
        for(int idx=tid*4;idx<BM*BK;idx+=threadNums*4){
           int innerRowA = idx / BK;
           int innerColA = idx % BK;

           int globalRowA = cRow + innerRowA;
           int globalColA = bkIdx + innerColA;
         
           if (globalRowA < M && globalColA+3 < K) {
            float4 temp;
            temp=LOAD_FLOAT4(A[globalRowA * K + globalColA]);
            As[innerColA][innerRowA]=temp.x;
            As[innerColA+1][innerRowA]=temp.y;
            As[innerColA+2][innerRowA]=temp.z;
            As[innerColA+3][innerRowA]=temp.w;
           } else {
            As[innerColA][innerRowA] = 0.0f;
           }
        }
        // ---------------------------
        // 2.2 协作加载 B 到 Shared Memory
        // ---------------------------
        for(int idx=tid*4;idx<BN*BK;idx+=threadNums*4){
           int innerRowB = idx / BN;
           int innerColB = idx % BN;

           int globalRowB = bkIdx + innerRowB;
           int globalColB = cCol + innerColB;
         
           if (globalRowB < K && globalColB+3 < N) {
            STORE_FLOAT4(Bs[innerRowB][innerColB])=LOAD_FLOAT4(B[globalRowB * N + globalColB]);
           } else {
            Bs[innerRowB][innerColB] = 0.0f;
           }
        }
        __syncthreads();
        // ---------------------------
        // 2.3 沿K方向计算
        // ---------------------------
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {
            //从As中读取数据到red_m中
            for(int i=0;i<TM;i+=4){
               STORE_FLOAT4(reg_m[i])=LOAD_FLOAT4(As[dotIdx][threadRow*TM+i]);
            }
            for(int i=0;i<TN;i+=4){
               STORE_FLOAT4(reg_n[i])=LOAD_FLOAT4(Bs[dotIdx][threadCol*TN+i]);
            }
            for(int m=0;m<TM;m++){
                for(int n=0;n<TN;n++){
                    threadResults[m*TN+n]+=reg_m[m]*reg_n[n];
                }
            }
        }
        __syncthreads();
   }
   // --------------------------------------------------------
    // 3. 将 TM*TN 个结果写回 Global Memory
    // ------------------------------------------------------
    for(int m=0;m<TM;m++){
        for(int n=0;n<TN;n+=4){
            STORE_FLOAT4(C[(cRow+threadRow*TM+m)*N+cCol+threadCol*TN+n])=LOAD_FLOAT4(threadResults[m*TN+n]);
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

    
    constexpr int M = 256;
    constexpr int N = 256;
    constexpr int K = 256;

    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 8;
    constexpr int TM = 8;
    constexpr int TN = 8;

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
    dim3 block(BM * BN / (TM*TN));

    // grid.x 对应列，grid.y 对应行
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    matmul_float4<BM, BN, BK, TM, TN>
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
    printf("Block threads: %d\n", BM * BN / (TM*TN));
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

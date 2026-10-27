#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
//#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
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
// BM = 64, BN = 64, BK = 8, TM = 4,TN=4;
//
// 每个 Block 计算 C 的 64x64 子块
// 每个线程计算 4x4 个 C 元素
// ============================================================
template<int BM,int BN,int BK,int WM,int WN,int WMITER,int WNITER,int TM,int TN>
__global__ void matmul_banksm_doubleBuffer(const float* A,const float* B,float* C,int M,int N,int K){
    static_assert(BM>0&&BN>0&&BK>0);
    static_assert(WM>0&&WN>0);
    static_assert(TM>0&&TN>0);
    static_assert(BM % WM == 0 && BN % WN==0,
                  "BM must be divisible by WM");
    constexpr int WARPSIZE=32;
    //Warp中每轮要处理的行数和列数
    constexpr int WSUBM=WM/WMITER;
    constexpr int WSUBN=WN/WNITER;
    //计算一个block中需要有多少个Warp
    constexpr int warp_rows=BM/WM;
    constexpr int warp_cols=BN/WN;
    constexpr int warp_nums=warp_rows*warp_cols;
    constexpr int threadNums=warp_nums*WARPSIZE;
    static_assert(threadNums <= 1024,
                  "Too many threads per block");
    static_assert(WM % WMITER == 0 && WN % WNITER == 0);
    static_assert(WSUBM % TM == 0 && WSUBN % TN == 0);
    static_assert((WSUBM / TM) * (WSUBN / TN) == 32);
    static_assert(BK % 4 == 0 && BN % 4 == 0);
    // Shared Memory
    __shared__ float As[2][BK][BM+4];
    __shared__ float Bs[2][BK][BN];

    //每线程处理的子矩阵，存放在寄存器中的数据
    float reg_m[2][TM]={0.0};
    float reg_n[2][TN]={0.0};
    
    // --------------------------------------------------------
    // 1. 计算线程负责的输出位置
    // --------------------------------------------------------
    // 当前 Block 在 C 中的 Tile 坐标
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // 当前 Block 对应的 C 子块起点
    int cRow = blockRow * BM;
    int cCol = blockCol * BN;
      const int tid = threadIdx.x;
    // 当前线程所在Warp在block中的位置
    const int warp_idx = tid / WARPSIZE;
    const int warp_col = warp_idx % (BN / WN);
    const int warp_row = warp_idx / (BN / WN);
    
    // 当前线程在warp内的索引
    const int thread_idx_in_warp = tid % WARPSIZE; // [0, 31]
    const int thread_col_in_warp = thread_idx_in_warp % (WSUBN / TN);
    const int thread_row_in_warp = thread_idx_in_warp / (WSUBN / TN);

    // 每个线程使用 TM*WMITER*TN*WNITER 个累加器
    float threadResults[TM*WMITER*TN*WNITER] = {0.0f};
    
    // --------------------------------------------------------
    // 2. 沿 K 方向逐 Tile 计算
    // --------------------------------------------------------
    //先把第一块加载到As和Bs中
    // As---------------------------
        for(int idx=tid*4;idx<BM*BK;idx+=threadNums*4){
           int innerRowA = idx / BK;
           int innerColA = idx % BK;

           int globalRowA = cRow + innerRowA;
           int globalColA = 0 + innerColA;
         
           if (globalRowA < M && globalColA+3 < K) {
            float4 temp;
            temp=LOAD_FLOAT4(A[globalRowA * K + globalColA]);
            As[0][innerColA][innerRowA]=temp.x;
            As[0][innerColA+1][innerRowA]=temp.y;
            As[0][innerColA+2][innerRowA]=temp.z;
            As[0][innerColA+3][innerRowA]=temp.w;
           } else {
            As[0][innerColA][innerRowA] = 0.0f;
           }
        }
         // Bs---------------------------
        for(int idx=tid*4;idx<BN*BK;idx+=threadNums*4){
           int innerRowB = idx / BN;
           int innerColB = idx % BN;

           int globalRowB = 0 + innerRowB;
           int globalColB = cCol + innerColB;
         
           if (globalRowB < K && globalColB+3 < N) {
            STORE_FLOAT4(Bs[0][innerRowB][innerColB])=LOAD_FLOAT4(B[globalRowB * N + globalColB]);
           } else {
            Bs[0][innerRowB][innerColB] = 0.0f;
           }
        }
        __syncthreads();
        int reader=0;
    for (int bkIdx = 0; bkIdx < K; bkIdx += BK){
           // ---------------------------
        // 2.1 协作加载 A 到 Shared Memory
        // ---------------------------
     if(bkIdx+BK<K){
        for(int idx=tid*4;idx<BM*BK;idx+=threadNums*4){
           int innerRowA = idx / BK;
           int innerColA = idx % BK;

           int globalRowA = cRow + innerRowA;
           int globalColA = bkIdx + BK + innerColA;
         
           if (globalRowA < M && globalColA+3 < K) {
            float4 temp;
            temp=LOAD_FLOAT4(A[globalRowA * K + globalColA]);
            As[1-reader][innerColA][innerRowA]=temp.x;
            As[1-reader][innerColA+1][innerRowA]=temp.y;
            As[1-reader][innerColA+2][innerRowA]=temp.z;
            As[1-reader][innerColA+3][innerRowA]=temp.w;
           } else {
            As[1-reader][innerColA][innerRowA] = 0.0f;
           }
        }
        // ---------------------------
        // 2.2 协作加载 B 到 Shared Memory
        // ---------------------------
        for(int idx=tid*4;idx<BN*BK;idx+=threadNums*4){
           int innerRowB = idx / BN;
           int innerColB = idx % BN;

           int globalRowB = bkIdx + BK + innerRowB;
           int globalColB = cCol + innerColB;
         
           if (globalRowB < K && globalColB+3 < N) {
            STORE_FLOAT4(Bs[1-reader][innerRowB][innerColB])=LOAD_FLOAT4(B[globalRowB * N + globalColB]);
           } else {
            Bs[1-reader][innerRowB][innerColB] = 0.0f;
           }
        }
    }
        //__syncthreads();
    
    // ---------------------------
    // 2.3 沿K方向计算
    // ---------------------------
    for (uint warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; ++warp_sub_row_idx){
      for (uint warp_sub_col_idx = 0; warp_sub_col_idx < WNITER; ++warp_sub_col_idx){
        for(int i=0;i<TM;i+=4){
                int as_row=warp_row*WM+thread_row_in_warp*TM+warp_sub_row_idx*WSUBM+i;
               STORE_FLOAT4(reg_m[0][i])=LOAD_FLOAT4(As[reader][0][as_row]);
            }
            for(int i=0;i<TN;i+=4){
                int bs_col=warp_col*WN+thread_col_in_warp*TN+warp_sub_col_idx*WSUBN+i;
               STORE_FLOAT4(reg_n[0][i])=LOAD_FLOAT4(Bs[reader][0][bs_col]);
            }
            int writer=0;
        for (int dotIdx = 0; dotIdx < BK; ++dotIdx) {    
            //从As中读取数据到red_m中
           if(dotIdx+1<BK){
             for(int i=0;i<TM;i+=4){
                int as_row=warp_row*WM+thread_row_in_warp*TM+warp_sub_row_idx*WSUBM+i;
               STORE_FLOAT4(reg_m[1-writer][i])=LOAD_FLOAT4(As[reader][dotIdx+1][as_row]);
             }
             for(int i=0;i<TN;i+=4){
                int bs_col=warp_col*WN+thread_col_in_warp*TN+warp_sub_col_idx*WSUBN+i;
               STORE_FLOAT4(reg_n[1-writer][i])=LOAD_FLOAT4(Bs[reader][dotIdx+1][bs_col]);
             }
           }
            for(int m=0;m<TM;m++){
                for(int n=0;n<TN;n++){
                    threadResults[(warp_sub_row_idx*TM+m)*(WNITER*TN)+warp_sub_col_idx*TN+n]+=
                    reg_m[writer][m]*reg_n[writer][n];
                }
            }
            writer=1-writer;
        }
        
      }
   }
        __syncthreads();
        reader=1-reader; 
   }
   // --------------------------------------------------------
    // 3. 将 threadResults 个结果写回 Global Memory
    // ------------------------------------------------------
    for (uint warp_sub_row_idx = 0; warp_sub_row_idx < WMITER; ++warp_sub_row_idx){
      for (uint warp_sub_col_idx = 0; warp_sub_col_idx < WNITER; ++warp_sub_col_idx){
        int rowBase=cRow+warp_row*WM+thread_row_in_warp*TM+warp_sub_row_idx*WSUBM;
        int colBase=cCol+warp_col*WN+thread_col_in_warp*TN+warp_sub_col_idx*WSUBN;
        for(int m=0;m<TM;m++){
           for(int n=0;n<TN;n+=4){
             STORE_FLOAT4(C[(rowBase+m)*N+colBase+n])=
             STORE_FLOAT4(threadResults[(warp_sub_row_idx*TM+m)*(WNITER*TN)+warp_sub_col_idx*TN+n]);
           }
        }
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
    constexpr int WM = 32;
    constexpr int WN = 32;
    constexpr int WMITER=2;
    constexpr int WNITER=1;
    constexpr int TM = 4;
    constexpr int TN = 4;

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
    constexpr int warpNums = (BM / WM) * (BN / WN);
    dim3 block(warpNums * 32);

    // grid.x 对应列，grid.y 对应行
    dim3 grid(
        (N + BN - 1) / BN,
        (M + BM - 1) / BM
    );

    matmul_banksm_doubleBuffer<BM, BN, BK,WM, WN,WMITER,WNITER, TM, TN>
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
    printf("Block warps: %d\n", BM * BN / (WM*WN));
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

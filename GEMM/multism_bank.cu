// Ampere (sm_80+) GEMM: register double buffering + two shared-memory stages.
// Aligned FP32 row-major GEMM: M%64==0, N%64==0, K%8==0.
// For comparison with the user's existing warp/register-double-buffer kernels.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

__device__ __forceinline__ void cp_async_16(void* smem, const void* gmem) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    unsigned int shared_addr = static_cast<unsigned int>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" ::
                 "r"(shared_addr), "l"(gmem) : "memory");
#else
    *reinterpret_cast<float4*>(smem) = *reinterpret_cast<const float4*>(gmem);
#endif
}

__device__ __forceinline__ void cp_async_commit() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

template<int BM,int BN,int BK,int WM,int WN,int WMITER,int WNITER,int TM,int TN>
__global__ void matmul_bankasync_doubleBuffer(const float* __restrict__ A,
                                       const float* __restrict__ B,
                                       float* __restrict__ C,
                                       int M,int N,int K) {
    constexpr int WS = 32;
    constexpr int SUBM=WM/WMITER, SUBN=WN/WNITER;
    constexpr int NWARPS=(BM/WM)*(BN/WN);
    constexpr int NT=NWARPS*WS;
    static_assert(BM%WM==0 && BN%WN==0 && WM%WMITER==0 && WN%WNITER==0);
    static_assert(SUBM%TM==0 && SUBN%TN==0 && (SUBM/TM)*(SUBN/TN)==32);
    static_assert(BK%4==0 && BN%4==0 && TN%4==0);
    static_assert(NT==128, "This experiment assumes 128 threads/block");
    // A is row-major in shared to permit aligned 16-byte cp.async from global.
    // (The original A[BK][BM] transposed layout cannot be directly cp.async copied.)
    __shared__ __align__(16) float As[2][BM][BK+4];
    __shared__ __align__(16) float Bs[2][BK][BN];
    const int tid=threadIdx.x;
    const int warp=tid/WS;
    const int warp_row=warp/(BN/WN), warp_col=warp%(BN/WN);
    const int lane=tid%WS;
    const int lane_col=lane%(SUBN/TN), lane_row=lane/(SUBN/TN);
    const int cRow=blockIdx.y*BM, cCol=blockIdx.x*BN;
    float accum[WMITER*TM*WNITER*TN]={};

    // Stage first tile into buffer 0 (one 16-byte global->shared copy per chunk).
    for(int idx=tid*4;idx<BM*BK;idx+=NT*4) {
        int r=idx/BK, k=idx%BK;
        cp_async_16(&As[0][r][k], &A[(cRow+r)*K+k]);
    }
    for(int idx=tid*4;idx<BK*BN;idx+=NT*4) {
        int k=idx/BN, c=idx%BN;
        cp_async_16(&Bs[0][k][c], &B[k*N+cCol+c]);
    }
    cp_async_commit();
    cp_async_wait();
    __syncthreads();

    int reader=0;
    for(int kb=0;kb<K;kb+=BK) {
        const bool has_next=(kb+BK<K);
        if(has_next) {
            // Issue async copy for next K tile to the alternate buffer.
            const int next=1-reader;
            for(int idx=tid*4;idx<BM*BK;idx+=NT*4) {
                int r=idx/BK, k=idx%BK;
                cp_async_16(&As[next][r][k], &A[(cRow+r)*K+kb+BK+k]);
            }
            for(int idx=tid*4;idx<BK*BN;idx+=NT*4) {
                int k=idx/BN, c=idx%BN;
                cp_async_16(&Bs[next][k][c], &B[(kb+BK+k)*N+cCol+c]);
            }
            cp_async_commit();
        }

        // Current tile computation. Keep the user's register ping-pong scheme.
        for(int wr=0;wr<WMITER;++wr) for(int wc=0;wc<WNITER;++wc) {
            const int row0=warp_row*WM+wr*SUBM+lane_row*TM;
            const int col0=warp_col*WN+wc*SUBN+lane_col*TN;
            float regA[2][TM];
            float regB[2][TN];
            for(int m=0;m<TM;m++) regA[0][m]=As[reader][row0+m][0];
            for(int n=0;n<TN;n++) regB[0][n]=Bs[reader][0][col0+n];
            int current=0;
            for(int dot=0;dot<BK;dot++) {
                if(dot+1<BK) {
                    for(int m=0;m<TM;m++) regA[1-current][m]=As[reader][row0+m][dot+1];
                    for(int n=0;n<TN;n++) regB[1-current][n]=Bs[reader][dot+1][col0+n];
                }
                for(int m=0;m<TM;m++) for(int n=0;n<TN;n++) {
                    accum[(wr*TM+m)*(WNITER*TN)+wc*TN+n] +=
                        regA[current][m]*regB[current][n];
                }
                current^=1;
            }
        }
        // All threads must finish reading current buffer before it can be
        // recycled two tiles later, and next tile must have arrived.
        if(has_next) {
            cp_async_wait();
            __syncthreads();
            reader^=1;
        }
    }
    for(int wr=0;wr<WMITER;wr++) for(int wc=0;wc<WNITER;wc++) {
        const int r0=cRow+warp_row*WM+wr*SUBM+lane_row*TM;
        const int c0=cCol+warp_col*WN+wc*SUBN+lane_col*TN;
        for(int m=0;m<TM;m++) for(int n=0;n<TN;n+=4) {
            const float* src=&accum[(wr*TM+m)*(WNITER*TN)+wc*TN+n];
            // Scalar stores avoid assuming the register array is 16B aligned.
            #pragma unroll
            for(int q=0;q<4;q++) C[(r0+m)*N+c0+n+q]=src[q];
        }
    }
}

#ifndef GEMM_NO_STANDALONE_MAIN
int main() {
    constexpr int M=256,N=256,K=256;
    std::vector<float> A(M*K), B(K*N), out(M*N), ref(M*N,0.f);
    for(size_t i=0;i<A.size();i++) A[i]=float(int(i%7))*0.1f;
    for(size_t i=0;i<B.size();i++) B[i]=float(int(i%5))*0.1f;
    float *dA=nullptr,*dB=nullptr,*dC=nullptr;
    auto check=[](cudaError_t e){if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e));exit(1);}};
    check(cudaMalloc(&dA,A.size()*sizeof(float)));
    check(cudaMalloc(&dB,B.size()*sizeof(float)));
    check(cudaMalloc(&dC,out.size()*sizeof(float)));
    check(cudaMemcpy(dA,A.data(),A.size()*sizeof(float),cudaMemcpyHostToDevice));
    check(cudaMemcpy(dB,B.data(),B.size()*sizeof(float),cudaMemcpyHostToDevice));
    dim3 grid(N/64,M/64);
    matmul_bankasync_doubleBuffer<64,64,8,32,32,2,1,4,4><<<grid,128>>>(dA,dB,dC,M,N,K);
    check(cudaGetLastError()); check(cudaDeviceSynchronize());
    check(cudaMemcpy(out.data(),dC,out.size()*sizeof(float),cudaMemcpyDeviceToHost));
    for(int i=0;i<M;i++) for(int j=0;j<N;j++)
        for(int k=0;k<K;k++) ref[i*N+j]+=A[i*K+k]*B[k*N+j];
    double maxerr=0; int bad=0;
    for(size_t i=0;i<out.size();i++) {
        double diff=fabs(double(out[i])-ref[i]);
        maxerr=fmax(maxerr,diff);
        if(!std::isfinite(out[i])||diff>1e-4+1e-4*fabs(ref[i])) bad++;
    }
    printf("M=%d N=%d K=%d bad=%d max_abs=%g\n",M,N,K,bad,maxerr);
    check(cudaFree(dA));check(cudaFree(dB));check(cudaFree(dC));
    return bad?1:0;
}
#endif

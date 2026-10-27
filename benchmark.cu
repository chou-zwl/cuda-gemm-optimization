// Benchmark for the original GEMM demos. Compile once per VARIANT (1..4).
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <cstdlib>
#include <string>

// Keep the user's kernel unchanged; rename its own main to avoid conflicts.
#define main demo_original_main
#if VARIANT == 1
#include "GEMM/multi_1D_tile.cu"
#elif VARIANT == 2
#include "GEMM/multi_2D_tile.cu"
#elif VARIANT == 3
#include "GEMM/multi_float4.cu"
#elif VARIANT == 4
#include "GEMM/warpTiling.cu"
#elif VARIANT == 5
#include "GEMM/doubleBuffer.cu"
#elif VARIANT == 6
#include "GEMM/sm_doubleBuffer.cu"
#elif VARIANT == 7
#include "GEMM/multi_sm_doubleBuffer.cu"
#elif VARIANT == 8
#include "GEMM/multism_bank.cu"
#elif VARIANT == 9
#include "GEMM/banksm_doubleBuffer.cu"
#else
#error "Define VARIANT=1,2,3,4,5,6,7"
#endif
#undef main

#define CK(call) do { auto e=(call); if(e!=cudaSuccess){std::cerr<<"CUDA: "<<cudaGetErrorString(e)<<" line "<<__LINE__<<"\n";std::exit(1);} }while(0)
#define CB(call) do { auto s=(call); if(s!=CUBLAS_STATUS_SUCCESS){std::cerr<<"cuBLAS error "<<int(s)<<" line "<<__LINE__<<"\n";std::exit(1);} }while(0)

void launch(float* a,float* b,float* c,int M,int N,int K){
 const int BM=64,BN=64,BK=8;
 dim3 grid((N+BN-1)/BN,(M+BM-1)/BM);
#if VARIANT == 1
 matmul_1d_tile<BM,BN,BK,8><<<grid,512>>>(a,b,c,M,N,K);
#elif VARIANT == 2
 matmul_2d_tile<BM,BN,BK,8,8><<<grid,64>>>(a,b,c,M,N,K);
#elif VARIANT == 3
 matmul_float4<BM,BN,BK,8,8><<<grid,64>>>(a,b,c,M,N,K);
#elif VARIANT == 4
 matmul_warpTiling<BM,BN,BK,32,32,2,1,4,4><<<grid,128>>>(a,b,c,M,N,K);
 #elif VARIANT == 5
    // Register Double Buffer
 matmul_doubleBuffer<BM, BN, BK, 32, 32, 2, 1, 4, 4><<<grid, 128>>>(a, b, c, M, N, K);
 #elif VARIANT == 6
    // Register Double Buffer
 matmul_sm_doubleBuffer<BM, BN, BK, 32, 32, 2, 1, 4, 4><<<grid, 128>>>(a, b, c, M, N, K);
 #elif VARIANT == 7
    // Register Double Buffer
 matmul_async_doubleBuffer<BM, BN, BK, 32, 32, 2, 1, 4, 4><<<grid, 128>>>(a, b, c, M, N, K);
 #elif VARIANT == 8
    // Register Double Buffer
 matmul_bankasync_doubleBuffer<BM, BN, BK, 32, 32, 2, 1, 4, 4><<<grid, 128>>>(a, b, c, M, N, K);
 #elif VARIANT == 9
    // Register Double Buffer
 matmul_banksm_doubleBuffer<BM, BN, BK, 32, 32, 2, 1, 4, 4><<<grid, 128>>>(a, b, c, M, N, K);
#endif
 CK(cudaGetLastError());
}

template<typename F> float time_cuda_events(F fn,int warmup=15,int repeat=100){
 for(int i=0;i<warmup;i++)fn();
 CK(cudaDeviceSynchronize());
 cudaEvent_t start,stop; CK(cudaEventCreate(&start)); CK(cudaEventCreate(&stop));
 CK(cudaEventRecord(start));
 for(int i=0;i<repeat;i++)fn();
 CK(cudaEventRecord(stop)); CK(cudaEventSynchronize(stop));
 float total=0; CK(cudaEventElapsedTime(&total,start,stop));
 CK(cudaEventDestroy(start));CK(cudaEventDestroy(stop));
 return total/repeat;
}

int main(int argc,char**argv){
 // Initial test only: square, multiples of 64; float4 also requires alignment.
 int M=argc>1?std::atoi(argv[1]):256;
 int N=argc>2?std::atoi(argv[2]):256;
 int K=argc>3?std::atoi(argv[3]):256;
 if(M<=0||N<=0||K<=0||M%64||N%64||K%8||N%4||K%4){
  std::cerr<<"For first benchmark: M,N multiples of 64; K multiple of 8.\n";return 1;
 }
 size_t nA=size_t(M)*K,nB=size_t(K)*N,nC=size_t(M)*N;
 std::vector<float> A(nA),B(nB),custom(nC),reference(nC);
 for(size_t i=0;i<nA;i++)A[i]=(float(int((i*17)%31)-15))/31.f;
 for(size_t i=0;i<nB;i++)B[i]=(float(int((i*13)%29)-14))/29.f;
 float *a,*b,*c,*ref;CK(cudaMalloc(&a,nA*4));CK(cudaMalloc(&b,nB*4));CK(cudaMalloc(&c,nC*4));CK(cudaMalloc(&ref,nC*4));
 CK(cudaMemcpy(a,A.data(),nA*4,cudaMemcpyHostToDevice));
 CK(cudaMemcpy(b,B.data(),nB*4,cudaMemcpyHostToDevice));
 cublasHandle_t h;CB(cublasCreate(&h));
 // Row-major C=A*B corresponds to column-major C^T=B^T*A^T.
 const float alpha=1.f,beta=0.f;
 auto run_cublas=[&](){ CB(cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,M,K,&alpha,b,N,a,K,&beta,ref,N)); };
 auto run_custom=[&](){launch(a,b,c,M,N,K);};
 CK(cudaMemset(c,0xFF,nC*4));run_custom();run_cublas();CK(cudaDeviceSynchronize());
 CK(cudaMemcpy(custom.data(),c,nC*4,cudaMemcpyDeviceToHost));
 CK(cudaMemcpy(reference.data(),ref,nC*4,cudaMemcpyDeviceToHost));
 int bad=0;double max_abs=0,max_rel=0;
 for(size_t i=0;i<nC;i++){
  double x=custom[i], y=reference[i];
  double diff=std::abs(x-y),tol=1e-3+1e-3*std::abs(y);
  if(!std::isfinite(x)||diff>tol)bad++;
  max_abs=std::max(max_abs,diff);
  max_rel=std::max(max_rel,diff/(std::abs(y)+1e-6));
 }
 std::cout<<"VARIANT="<<VARIANT<<" M="<<M<<" N="<<N<<" K="<<K
          <<" bad="<<bad<<" max_abs="<<max_abs<<" max_rel="<<max_rel<<"\n";
 if(bad){std::cerr<<"Correctness FAILED; skip speed test.\n";return 2;}
 // Time device compute only: no H2D, D2H, allocation or handle construction.
 float mine=time_cuda_events(run_custom);
 float blas=time_cuda_events(run_cublas);
 double flops=2.0*M*N*K;
 std::cout<<std::fixed<<std::setprecision(4)
          <<"custom_ms="<<mine<<" cublas_ms="<<blas
          <<" custom_GFLOPS="<<flops/(mine*1e6)
          <<" cublas_GFLOPS="<<flops/(blas*1e6)
          <<" ratio_custom_over_cublas="<<blas/mine<<"\n";
 CB(cublasDestroy(h)); CK(cudaFree(a));CK(cudaFree(b));CK(cudaFree(c));CK(cudaFree(ref));
 return 0;
}

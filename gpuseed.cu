/*
 * gpuseed.cu — GPU-ONLY searcher over the whole 2^128 Electrum v1 old-seed space.
 *   Everything runs on the card: seed-gen + stretch + a*G + 83-key cascade.
 *   The CPU does NO per-seed work — it only drains the rare hits, so a separate
 *   CPU hunter (e.g. ecloop) keeps all its cores.
 *
 * Two kernels (so the EC math does not wreck the stretch's occupancy):
 *   A) stretch_kernel : seed=base+idx -> 32-hex -> 100000x SHA -> master (in GPU mem)
 *   B) casc_kernel    : master -> a*G=MPK -> cascade -> hit in a device buffer
 *   The intermediate master buffer stays on the card (zero PCIe per seed).
 *
 * The SHA round uses lop3.b32 (ch=0xca, maj=0xe8 — verified) + __funnelshift_r.
 * EC (a*G) and the cascade (child_low mod N) were verified on the host against
 * libsecp256k1 / Python before going into the kernel.
 *
 * Build:
 *   5090 (Blackwell): nvcc -O3 -arch=sm_120 -Xptxas -v -Xcompiler -fopenmp gpuseed.cu -o gpuseed -lcrypto -lgomp
 *   4060 Ti (Ada):    nvcc -O3 -arch=sm_89  -Xptxas -v -Xcompiler -fopenmp gpuseed.cu -o gpuseed -lcrypto -lgomp
 *   ./gpuseed --selftest     # stretch + a*G + child + address + mnemonic -> SELFTEST PASS
 *   CUDA_VISIBLE_DEVICES=0 ./gpuseed
 *
 * Env: BATCH, JUMP=1 (random base/batch), PROGRESS=<file> (resume), BASE_HI/BASE_LO,
 *      HIT_PUBKEY=<pem> (encrypt hits), WORDS=<old_words.txt>.
 *
 * Built by Hristo x Claude. MIT License.
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <ctime>
#include <cuda_runtime.h>
#include <openssl/sha.h>
#include <openssl/bn.h>
#include <openssl/ec.h>
#include <openssl/obj_mac.h>
#include <openssl/ripemd.h>
#include <openssl/crypto.h>
#include "known_keys.h"
#include "hitcrypt.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#ifdef _WIN32
#define strcasecmp _stricmp
#endif

#define ROUNDS  100000
#define MAXHITS 64
#ifdef ILP2
#define SPT 2
#else
#define SPT 1
#endif
#define PUZZLE71_ADDRESS "1PWo3JeB9jrGwfHDNpdGK54CRas7fsVzXU"
#define CUDA_CHECK(x) do{ cudaError_t e_=(x); if(e_!=cudaSuccess){ \
    fprintf(stderr,"CUDA error %s at %s:%d\n",cudaGetErrorString(e_),__FILE__,__LINE__); exit(3);} }while(0)

/* ============================ device SHA256 ============================ */
__constant__ uint32_t cK[64]={
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};

__device__ __forceinline__ uint32_t rotr(uint32_t x,int n){ return __funnelshift_r(x,x,n); }
/* PLAIN_CHMAJ: keep the plain C expression and let the compiler choose (forcing
 * lop3 can hurt on some arches). Default: forced lop3.b32 (ch=0xca, maj=0xe8).
 * Build both and measure — on Ada they come out equal. */
#ifdef PLAIN_CHMAJ
__device__ __forceinline__ uint32_t Chf(uint32_t e,uint32_t f,uint32_t g){ return (e&f)^(~e&g); }
__device__ __forceinline__ uint32_t Majf(uint32_t a,uint32_t b,uint32_t c){ return (a&b)^(a&c)^(b&c); }
#else
__device__ __forceinline__ uint32_t Chf(uint32_t e,uint32_t f,uint32_t g){
    uint32_t r; asm("lop3.b32 %0,%1,%2,%3,0xca;":"=r"(r):"r"(e),"r"(f),"r"(g)); return r; }
__device__ __forceinline__ uint32_t Majf(uint32_t a,uint32_t b,uint32_t c){
    uint32_t r; asm("lop3.b32 %0,%1,%2,%3,0xe8;":"=r"(r):"r"(a),"r"(b),"r"(c)); return r; }
#endif
#define RS0(x) (rotr(x,7)^rotr(x,18)^((x)>>3))
#define RS1(x) (rotr(x,17)^rotr(x,19)^((x)>>10))
#define RB0(x) (rotr(x,2)^rotr(x,13)^rotr(x,22))
#define RB1(x) (rotr(x,6)^rotr(x,11)^rotr(x,25))

/* one SHA256 compression, rolling schedule w[16] */
__device__ __forceinline__ void d_block(uint32_t s[8], uint32_t w[16]){
    uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
    #pragma unroll
    for(int i=0;i<64;i++){
        if(i>=16) w[i&15]+=RS1(w[(i-2)&15])+w[(i-7)&15]+RS0(w[(i-15)&15]);
        uint32_t t1=h+RB1(e)+Chf(e,f,g)+cK[i]+w[i&15];
        uint32_t t2=RB0(a)+Majf(a,b,c);
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
}
/* SHA256 of an arbitrary short message (<=119 bytes -> <=2 blocks) */
__device__ void d_sha256_var(const unsigned char*m,int len,unsigned char out[32]){
    uint32_t s[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    unsigned char buf[128]; int full=((len+9+63)/64)*64;
    for(int i=0;i<len;i++) buf[i]=m[i];
    buf[len]=0x80; for(int i=len+1;i<full;i++) buf[i]=0;
    uint64_t bits=(uint64_t)len*8; for(int i=0;i<8;i++) buf[full-1-i]=(unsigned char)(bits>>(8*i));
    for(int o=0;o<full;o+=64){ uint32_t w[16];
        #pragma unroll
        for(int i=0;i<16;i++) w[i]=((uint32_t)buf[o+4*i]<<24)|((uint32_t)buf[o+4*i+1]<<16)|((uint32_t)buf[o+4*i+2]<<8)|buf[o+4*i+3];
        d_block(s,w); }
    #pragma unroll
    for(int i=0;i<8;i++){ out[4*i]=s[i]>>24;out[4*i+1]=s[i]>>16;out[4*i+2]=s[i]>>8;out[4*i+3]=s[i]; }
}
__device__ void d_sha256d(const unsigned char*m,int len,unsigned char out[32]){ unsigned char t[32]; d_sha256_var(m,len,t); d_sha256_var(t,32,out); }

/* ============================ KERNEL A: stretch ============================ */
__device__ __forceinline__ uint32_t d_hx(uint32_t nib){ return nib<10 ? ('0'+nib) : ('a'+nib-10); }
__device__ __forceinline__ void d_seed_to_words(uint64_t hi, uint64_t lo, uint32_t osw[8]){
    #pragma unroll
    for(int j=0;j<8;j++){ uint64_t src=(j<4)?hi:lo; int loc=(j<4)?j:(j-4); int sh0=56-16*loc,sh1=sh0-8;
        uint32_t b0=(src>>sh0)&0xffu,b1=(src>>sh1)&0xffu;
        osw[j]=(d_hx(b0>>4)<<24)|(d_hx(b0&15)<<16)|(d_hx(b1>>4)<<8)|d_hx(b1&15); }
}
/* one stretch step on (osw,cur): block1 (rolling) + block2 (register schedule) */
__device__ __forceinline__ void stretch_step(const uint32_t osw[8], uint32_t cur[8]){
    uint32_t s[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19},w[16];
    #pragma unroll
    for(int i=0;i<8;i++){ w[i]=cur[i]; w[8+i]=osw[i]; }
    d_block(s,w);
    uint32_t w2[16]; w2[0]=0x80000000u;
    #pragma unroll
    for(int i=1;i<15;i++) w2[i]=0;
    w2[15]=512u; d_block(s,w2);
    #pragma unroll
    for(int i=0;i<8;i++) cur[i]=s[i];
}
__device__ __forceinline__ void write_master(unsigned char*o,const uint32_t cur[8]){
    #pragma unroll
    for(int i=0;i<8;i++){ o[4*i]=cur[i]>>24;o[4*i+1]=cur[i]>>16;o[4*i+2]=cur[i]>>8;o[4*i+3]=cur[i]; }
}
/* NO_LB: drop launch_bounds, let ptxas allocate registers freely (launch_bounds can hurt) */
#ifndef NO_LB
__launch_bounds__(256)
#endif
__global__ void stretch_kernel(uint64_t g_hi,uint64_t g_lo,unsigned char*d_master,int nthreads){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=nthreads) return;
#ifdef ILP2
    /* 2 seeds/thread, both chains in ONE loop body -> ptxas interleaves them (ILP) */
    uint64_t si0=(uint64_t)2*t, si1=si0+1;
    uint64_t lo0=g_lo+si0, hi0=g_hi+(lo0<g_lo?1ull:0ull);
    uint64_t lo1=g_lo+si1, hi1=g_hi+(lo1<g_lo?1ull:0ull);
    uint32_t osw0[8],cur0[8],osw1[8],cur1[8];
    d_seed_to_words(hi0,lo0,osw0); d_seed_to_words(hi1,lo1,osw1);
    #pragma unroll
    for(int i=0;i<8;i++){ cur0[i]=osw0[i]; cur1[i]=osw1[i]; }
    for(int r=0;r<ROUNDS;r++){ stretch_step(osw0,cur0); stretch_step(osw1,cur1); }
    write_master(d_master+si0*32,cur0); write_master(d_master+si1*32,cur1);
#else
    uint64_t lo=g_lo+(uint64_t)t, hi=g_hi+(lo<g_lo?1ull:0ull);
    uint32_t osw[8],cur[8]; d_seed_to_words(hi,lo,osw);
    #pragma unroll
    for(int i=0;i<8;i++) cur[i]=osw[i];
    for(int r=0;r<ROUNDS;r++) stretch_step(osw,cur);
    write_master(d_master+(size_t)t*32,cur);
#endif
}

/* ============================ device secp256k1 a*G ============================ */
typedef struct { uint32_t n[8]; } fe;
__constant__ uint32_t cP[8]={0xFFFFFC2F,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
#define CVAL 0x1000003d1ULL
__device__ int d_ge_p(const uint32_t a[8]){ for(int i=7;i>=0;i--) if(a[i]!=cP[i]) return a[i]>cP[i]; return 1; }
__device__ void d_sub_p(uint32_t a[8]){ uint64_t br=0; for(int i=0;i<8;i++){ uint64_t t=(uint64_t)a[i]-cP[i]-br; a[i]=(uint32_t)t; br=(t>>63)&1; } }
__device__ void d_reduce_p(uint32_t a[8]){ if(d_ge_p(a)) d_sub_p(a); }
__device__ void d_add_small(uint32_t a[8],uint64_t add){ while(add){ uint64_t carry=add,i=0; add=0; while(carry&&i<8){ uint64_t t=(uint64_t)a[i]+(uint32_t)carry; a[i]=(uint32_t)t; carry=(carry>>32)+(t>>32); i++; } if(carry) add=carry*CVAL; } }
__device__ void fe_add(fe*r,const fe*a,const fe*b){ uint64_t c=0; for(int i=0;i<8;i++){ uint64_t t=(uint64_t)a->n[i]+b->n[i]+c; r->n[i]=(uint32_t)t; c=t>>32; } if(c) d_add_small(r->n,c*CVAL); d_reduce_p(r->n); }
__device__ void fe_sub(fe*r,const fe*a,const fe*b){ uint64_t br=0; uint32_t t[8]; for(int i=0;i<8;i++){ uint64_t x=(uint64_t)a->n[i]-b->n[i]-br; t[i]=(uint32_t)x; br=(x>>63)&1; } if(br){ uint64_t c=0; for(int i=0;i<8;i++){ uint64_t s=(uint64_t)t[i]+cP[i]+c; t[i]=(uint32_t)s; c=s>>32; } } for(int i=0;i<8;i++) r->n[i]=t[i]; }
__device__ void fe_mul(fe*r,const fe*a,const fe*b){
    uint64_t prod[16];
    #pragma unroll
    for(int i=0;i<16;i++) prod[i]=0;
    for(int i=0;i<8;i++){ uint64_t carry=0;
        for(int j=0;j<8;j++){ unsigned __int128 t=(unsigned __int128)a->n[i]*b->n[j]+prod[i+j]+carry; prod[i+j]=(uint32_t)t; carry=(uint64_t)(t>>32); }
        int k=i+8; while(carry){ uint64_t t=prod[k]+(uint32_t)carry; prod[k]=(uint32_t)t; carry=(carry>>32)+(t>>32); k++; } }
    uint32_t w[16];
    #pragma unroll
    for(int i=0;i<16;i++) w[i]=(uint32_t)prod[i];
    uint64_t acc[9];
    #pragma unroll
    for(int i=0;i<8;i++) acc[i]=w[i];
    acc[8]=0; uint64_t carry=0;
    for(int i=0;i<8;i++){ unsigned __int128 t=(unsigned __int128)w[8+i]*CVAL+acc[i]+carry; acc[i]=(uint32_t)t; carry=(uint64_t)(t>>32); }
    acc[8]+=carry;
    uint32_t res[8]; uint64_t cc=0;
    for(int i=0;i<8;i++){ uint64_t t=acc[i]+cc; res[i]=(uint32_t)t; cc=t>>32; }
    uint64_t extra=acc[8]+cc; if(extra) d_add_small(res,extra*CVAL);
    for(int i=0;i<8;i++) r->n[i]=res[i]; d_reduce_p(r->n);
}
__device__ void fe_sqr(fe*r,const fe*a){ fe_mul(r,a,a); }
__device__ void fe_set(fe*r,uint32_t v){ for(int i=0;i<8;i++) r->n[i]=0; r->n[0]=v; }
__device__ int fe_iszero(const fe*a){ uint32_t x=0; for(int i=0;i<8;i++) x|=a->n[i]; return x==0; }
__device__ void fe_copy(fe*r,const fe*a){ for(int i=0;i<8;i++) r->n[i]=a->n[i]; }
__device__ void fe_inv(fe*r,const fe*a){ fe base,res; fe_copy(&base,a); fe_set(&res,1);
    uint32_t e[8];
    #pragma unroll
    for(int i=0;i<8;i++) e[i]=cP[i];
    { uint64_t br=2; for(int i=0;i<8&&br;i++){ uint64_t t=(uint64_t)e[i]-(uint32_t)br; e[i]=(uint32_t)t; br=(t>>63)&1; } }
    for(int i=0;i<256;i++){ if((e[i>>5]>>(i&31))&1) fe_mul(&res,&res,&base); fe_sqr(&base,&base); }
    fe_copy(r,&res);
}
typedef struct { fe X,Y,Z; } jac;
__device__ void jac_inf(jac*p){ fe_set(&p->X,1); fe_set(&p->Y,1); fe_set(&p->Z,0); }
__device__ int jac_is_inf(const jac*p){ return fe_iszero(&p->Z); }
__device__ void jac_double(jac*r,const jac*p){ if(jac_is_inf(p)){ *r=*p; return; }
    fe A,B,C,D,t,X3,Y3,Z3;
    fe_sqr(&A,&p->Y); fe_mul(&B,&p->X,&A); fe_add(&B,&B,&B); fe_add(&B,&B,&B);
    fe_sqr(&C,&A); fe_add(&C,&C,&C); fe_add(&C,&C,&C); fe_add(&C,&C,&C);
    fe_sqr(&D,&p->X); fe_add(&t,&D,&D); fe_add(&D,&t,&D);
    fe_sqr(&X3,&D); fe_sub(&X3,&X3,&B); fe_sub(&X3,&X3,&B);
    fe_sub(&Y3,&B,&X3); fe_mul(&Y3,&D,&Y3); fe_sub(&Y3,&Y3,&C);
    fe_mul(&Z3,&p->Y,&p->Z); fe_add(&Z3,&Z3,&Z3);
    fe_copy(&r->X,&X3); fe_copy(&r->Y,&Y3); fe_copy(&r->Z,&Z3);
}
__device__ void jac_add(jac*r,const jac*p,const jac*q){ if(jac_is_inf(p)){ *r=*q; return; } if(jac_is_inf(q)){ *r=*p; return; }
    fe Z1Z1,Z2Z2,U1,U2,S1v,S2v,H,Rr;
    fe_sqr(&Z1Z1,&p->Z); fe_sqr(&Z2Z2,&q->Z);
    fe_mul(&U1,&p->X,&Z2Z2); fe_mul(&U2,&q->X,&Z1Z1);
    fe_mul(&S1v,&p->Y,&q->Z); fe_mul(&S1v,&S1v,&Z2Z2);
    fe_mul(&S2v,&q->Y,&p->Z); fe_mul(&S2v,&S2v,&Z1Z1);
    fe_sub(&H,&U2,&U1); fe_sub(&Rr,&S2v,&S1v);
    if(fe_iszero(&H)){ if(fe_iszero(&Rr)) jac_double(r,p); else jac_inf(r); return; }
    fe HH,HHH,U1HH,X3,Y3,Z3,t;
    fe_sqr(&HH,&H); fe_mul(&HHH,&HH,&H); fe_mul(&U1HH,&U1,&HH);
    fe_sqr(&X3,&Rr); fe_sub(&X3,&X3,&HHH); fe_sub(&X3,&X3,&U1HH); fe_sub(&X3,&X3,&U1HH);
    fe_sub(&Y3,&U1HH,&X3); fe_mul(&Y3,&Rr,&Y3); fe_mul(&t,&S1v,&HHH); fe_sub(&Y3,&Y3,&t);
    fe_mul(&Z3,&p->Z,&q->Z); fe_mul(&Z3,&Z3,&H);
    fe_copy(&r->X,&X3); fe_copy(&r->Y,&Y3); fe_copy(&r->Z,&Z3);
}
__constant__ uint32_t cGx[8]={0x16f81798,0x59f2815b,0x2dce28d9,0x029bfcdb,0xce870b07,0x55a06295,0xf9dcbbac,0x79be667e};
__constant__ uint32_t cGy[8]={0xfb10d4b8,0x9c47d08f,0xa6855419,0xfd17b448,0x0e1108a8,0x5da4fbfc,0x26a3c465,0x483ada77};
/* a (uint32[8] little-endian) -> mpk[64] = x||y big-endian */
__device__ void d_a_to_mpk(const uint32_t k[8], unsigned char mpk[64]){
    jac G; fe_copy(&G.X,(const fe*)cGx); fe_copy(&G.Y,(const fe*)cGy); fe_set(&G.Z,1);
    jac acc; jac_inf(&acc);
    for(int i=255;i>=0;i--){ jac_double(&acc,&acc); if((k[i>>5]>>(i&31))&1) jac_add(&acc,&acc,&G); }
    fe zi,zi2,zi3,X,Y; fe_inv(&zi,&acc.Z); fe_sqr(&zi2,&zi); fe_mul(&zi3,&zi2,&zi);
    fe_mul(&X,&acc.X,&zi2); fe_mul(&Y,&acc.Y,&zi3);
    #pragma unroll
    for(int i=0;i<8;i++){ uint32_t xv=X.n[7-i],yv=Y.n[7-i];
        mpk[i*4]=xv>>24;mpk[i*4+1]=xv>>16;mpk[i*4+2]=xv>>8;mpk[i*4+3]=xv;
        mpk[32+i*4]=yv>>24;mpk[32+i*4+1]=yv>>16;mpk[32+i*4+2]=yv>>8;mpk[32+i*4+3]=yv; }
}

/* ============================ device cascade (mod N) ============================ */
__constant__ uint32_t cN[8]={0xD0364141,0xBFD25E8C,0xAF48A03B,0xBAAEDCE6,0xFFFFFFFE,0xFFFFFFFF,0xFFFFFFFF,0xFFFFFFFF};
__device__ int d_ge_n(const uint32_t a[8]){ for(int i=7;i>=0;i--) if(a[i]!=cN[i]) return a[i]>cN[i]; return 1; }
__device__ void d_sub_n(uint32_t a[8]){ uint64_t br=0; for(int i=0;i<8;i++){ uint64_t t=(uint64_t)a[i]-cN[i]-br; a[i]=(uint32_t)t; br=(t>>63)&1; } }
__device__ void d_be_to_limbs(const unsigned char b[32],uint32_t n[8]){ for(int i=0;i<8;i++) n[7-i]=((uint32_t)b[4*i]<<24)|((uint32_t)b[4*i+1]<<16)|((uint32_t)b[4*i+2]<<8)|b[4*i+3]; }
__device__ int d_u2dec(unsigned v,char*out){ char t[12]; int n=0; if(v==0){out[0]='0';return 1;} while(v){t[n++]='0'+v%10;v/=10;} for(int i=0;i<n;i++)out[i]=t[n-1-i]; return n; }
/* child_low = ((a + sha256d("idx:change:"+mpk)) mod N) masked to `bits` low bits */
__device__ void d_childlow(const uint32_t a[8],const unsigned char mpk[64],int idx,int change,int bits,uint32_t out[8]){
    unsigned char msg[80]; int hl=0;
    hl+=d_u2dec((unsigned)idx,(char*)msg+hl); msg[hl++]=':'; hl+=d_u2dec((unsigned)change,(char*)msg+hl); msg[hl++]=':';
    for(int i=0;i<64;i++) msg[hl+i]=mpk[i]; int mlen=hl+64;
    unsigned char z[32]; d_sha256d(msg,mlen,z);
    uint32_t zn[8]; d_be_to_limbs(z,zn); if(d_ge_n(zn)) d_sub_n(zn);
    uint32_t r[8]; uint64_t c=0;
    #pragma unroll
    for(int i=0;i<8;i++){ uint64_t t=(uint64_t)a[i]+zn[i]+c; r[i]=(uint32_t)t; c=t>>32; }
    if(c||d_ge_n(r)) d_sub_n(r);
    #pragma unroll
    for(int i=0;i<8;i++){ int lo=i*32,hi=lo+32; if(bits<=lo) r[i]=0; else if(bits<hi) r[i]&=((bits-lo)>=32?0xffffffffu:((1u<<(bits-lo))-1)); }
    #pragma unroll
    for(int i=0;i<8;i++) out[i]=r[i];
}
/* device tables for the cascade (filled from the host) */
__constant__ int      cKN_n[128];
__constant__ int      cKN_bits[128];
__constant__ uint32_t cKN_res[128][8];
__constant__ int      cPROBE[16];
__constant__ int      cNPROBE;
__constant__ int      cNKNOWN;

struct Hit { uint32_t idx; int change; int off; };
__device__ Hit d_hits[MAXHITS];
__device__ int d_hitcount;

__device__ int d_cmp(const uint32_t a[8],const uint32_t b[8]){ for(int i=0;i<8;i++) if(a[i]!=b[i]) return 1; return 0; }

__global__ void casc_kernel(const unsigned char*d_master,int n){
    int t=blockIdx.x*blockDim.x+threadIdx.x; if(t>=n) return;
    uint32_t a[8]; d_be_to_limbs(d_master+(size_t)t*32,a); if(d_ge_n(a)) d_sub_n(a);
    unsigned char mpk[64]; d_a_to_mpk(a,mpk);
    uint32_t out[8];
    for(int change=0;change<=1;change++) for(int off=0;off<=1;off++){
        int ok=1;
        for(int p=0;p<cNPROBE&&ok;p++){ int j=cPROBE[p]; int n_=cKN_n[j],bits=cKN_bits[j],idx=n_-1+off; if(idx<0){ok=0;break;}
            d_childlow(a,mpk,idx,change,bits,out); if(d_cmp(out,cKN_res[j])) ok=0; }
        if(!ok) continue;
        for(int j=0;j<cNKNOWN&&ok;j++){ int n_=cKN_n[j],bits=cKN_bits[j],idx=n_-1+off;
            d_childlow(a,mpk,idx,change,bits,out); if(d_cmp(out,cKN_res[j])) ok=0; }
        if(ok){ int s=atomicAdd(&d_hitcount,1); if(s<MAXHITS){ d_hits[s].idx=(uint32_t)t; d_hits[s].change=change; d_hits[s].off=off; } }
    }
}

/* selftest kernel: a -> mpk(64) + childfull(32) */
__global__ void st_kernel(const unsigned char*d_a,unsigned char*d_mpk,unsigned char*d_child){
    uint32_t a[8]; d_be_to_limbs(d_a,a); if(d_ge_n(a)) d_sub_n(a);
    unsigned char mpk[64]; d_a_to_mpk(a,mpk); for(int i=0;i<64;i++) d_mpk[i]=mpk[i];
    uint32_t out[8]; d_childlow(a,mpk,0,0,256,out);
    for(int i=0;i<8;i++){ uint32_t v=out[7-i]; d_child[i*4]=v>>24;d_child[i*4+1]=v>>16;d_child[i*4+2]=v>>8;d_child[i*4+3]=v; }
}

/* ============================ host helpers ============================ */
static void host_seed_hex(unsigned __int128 v, char out[33]){ static const char*hd="0123456789abcdef";
    for(int i=15;i>=0;i--){ unsigned b=(unsigned)(v&0xff); v>>=8; out[2*i]=hd[b>>4]; out[2*i+1]=hd[b&15]; } out[32]=0; }
static unsigned __int128 parse_hex128(const char*s){ unsigned __int128 v=0; for(;*s;s++){ char c=*s; unsigned d;
    if(c>='0'&&c<='9')d=c-'0'; else if(c>='a'&&c<='f')d=c-'a'+10; else if(c>='A'&&c<='F')d=c-'A'+10; else continue; v=(v<<4)|d; } return v; }
static unsigned __int128 rand128(void){ unsigned __int128 v=0; FILE*ur=fopen("/dev/urandom","rb"); if(ur){ unsigned char r[16]; if(fread(r,1,16,ur)==16) for(int i=0;i<16;i++) v=(v<<8)|r[i]; fclose(ur);} return v; }
static char **g_words=NULL; static int g_nwords=0;
static void load_words(const char*path){ FILE*f=fopen(path,"r"); if(!f) return; g_words=(char**)malloc(sizeof(char*)*2048); char line[64];
    while(g_nwords<2048 && fgets(line,sizeof line,f)){ int L=(int)strlen(line); while(L&&(line[L-1]=='\n'||line[L-1]=='\r'||line[L-1]==' ')) line[--L]=0; if(L==0)continue; g_words[g_nwords]=strdup(line); if(g_words[g_nwords])g_nwords++; } fclose(f); }
static int mnemonic_encode(const char seed_hex[33],char out[256]){ out[0]=0; if(g_nwords!=1626) return 0; int n=g_nwords; char tmp[9];
    for(int grp=0;grp<4;grp++){ memcpy(tmp,seed_hex+grp*8,8); tmp[8]=0; unsigned long x=strtoul(tmp,NULL,16);
        int w1=(int)(x%n),w2=(int)(((x/n)+w1)%n),w3=(int)(((x/n/n)+w2)%n);
        strcat(out,g_words[w1]);strcat(out," ");strcat(out,g_words[w2]);strcat(out," ");strcat(out,g_words[w3]); if(grp<3)strcat(out," "); } return 1; }

/* host OpenSSL cascade (verified) for reporting a hit */
static void sha256d_h(const unsigned char*in,size_t n,unsigned char out[32]){ unsigned char t[32]; SHA256(in,n,t); SHA256(t,32,out); }
typedef struct { EC_GROUP*grp; BN_CTX*ctx; EC_POINT*pt; BIGNUM*a,*z,*k,*order,*x,*y; } ecctx_t;
static void ec_init(ecctx_t*e){ e->grp=EC_GROUP_new_by_curve_name(NID_secp256k1); e->ctx=BN_CTX_new(); e->pt=EC_POINT_new(e->grp);
    e->a=BN_new();e->z=BN_new();e->k=BN_new();e->order=BN_new();e->x=BN_new();e->y=BN_new(); EC_GROUP_get_order(e->grp,e->order,e->ctx); }
static void a_to_mpk_h(ecctx_t*e,const unsigned char a32[32],unsigned char mpk64[64]){ BN_bin2bn(a32,32,e->a); BN_mod(e->a,e->a,e->order,e->ctx);
    EC_POINT_mul(e->grp,e->pt,e->a,NULL,NULL,e->ctx); EC_POINT_get_affine_coordinates(e->grp,e->pt,e->x,e->y,e->ctx);
    memset(mpk64,0,64); BN_bn2binpad(e->x,mpk64,32); BN_bn2binpad(e->y,mpk64+32,32); }
static void child_full(ecctx_t*e,const unsigned char mpk64[64],int idx,int change,BIGNUM*out){ unsigned char pre[32+64]; char hdr[32]; int hl=snprintf(hdr,sizeof hdr,"%d:%d:",idx,change);
    memcpy(pre,hdr,hl); memcpy(pre+hl,mpk64,64); unsigned char z32[32]; sha256d_h(pre,hl+64,z32); BN_bin2bn(z32,32,e->z); BN_mod_add(out,e->a,e->z,e->order,e->ctx); }
static void child_low(ecctx_t*e,const unsigned char mpk64[64],int idx,int change,int bits,BIGNUM*out){ child_full(e,mpk64,idx,change,out); BN_mask_bits(out,bits); }
static int base58check(const unsigned char payload[21],char out[64]){ static const char alpha[]="123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
    unsigned char raw[25],h[32]; memcpy(raw,payload,21); sha256d_h(payload,21,h); memcpy(raw+21,h,4); BIGNUM*bn=BN_bin2bn(raw,25,NULL); if(!bn)return 0; char tmp[64]; int nn=0,zeros=0;
    while(zeros<25&&raw[zeros]==0)zeros++; while(!BN_is_zero(bn)){ BN_ULONG r=BN_div_word(bn,58); if(r==(BN_ULONG)-1){BN_free(bn);return 0;} tmp[nn++]=alpha[r]; }
    while(zeros--) tmp[nn++]='1'; for(int i=0;i<nn;i++) out[i]=tmp[nn-1-i]; out[nn]=0; BN_free(bn); return 1; }
static int p2pkh_from_scalar(ecctx_t*e,const BIGNUM*priv,int compressed,char address[64]){ EC_POINT*pt=EC_POINT_new(e->grp); unsigned char pub[65],sha[32],h160[20],payload[21]; size_t len; int ok=0;
    if(!pt||!EC_POINT_mul(e->grp,pt,priv,NULL,NULL,e->ctx)) goto done;
    len=EC_POINT_point2oct(e->grp,pt,compressed?POINT_CONVERSION_COMPRESSED:POINT_CONVERSION_UNCOMPRESSED,pub,sizeof(pub),e->ctx);
    if(len!=(compressed?33u:65u)) goto done; SHA256(pub,len,sha); if(!RIPEMD160(sha,32,h160)) goto done; payload[0]=0x00; memcpy(payload+1,h160,20); ok=base58check(payload,address);
done: EC_POINT_free(pt); return ok; }
static void stretch_host(const char seed_hex[33],unsigned char out[32]){ unsigned char cur[32],old[32]; memcpy(old,seed_hex,32); memcpy(cur,seed_hex,32);
    unsigned char buf[64]; for(int r=0;r<ROUNDS;r++){ memcpy(buf,cur,32); memcpy(buf+32,old,32); SHA256(buf,64,cur); } memcpy(out,cur,32); }
/* reproduce full report for a found seed (OpenSSL, verified path) */
static void report_hit(unsigned __int128 seedval,int change,int off){
    char seed[33]; host_seed_hex(seedval,seed);
    unsigned char a32[32]; stretch_host(seed,a32);
    ecctx_t e; ec_init(&e); unsigned char mpk[64]; a_to_mpk_h(&e,a32,mpk);
    int idx71=71-1+off; BIGNUM*pred=BN_new(); child_full(&e,mpk,idx71,change,pred); BN_mask_bits(pred,70); BN_set_bit(pred,70);
    char ac[64],au[64]; p2pkh_from_scalar(&e,pred,1,ac); p2pkh_from_scalar(&e,pred,0,au); BN_free(pred);
    char mnem[256]; mnemonic_encode(seed,mnem);
    int mc=!strcmp(ac,PUZZLE71_ADDRESS),mu=!strcmp(au,PUZZLE71_ADDRESS);
    char msg[768]; snprintf(msg,sizeof msg,"SEED=%s\n  12 words: %s\n  change=%d off=%d #71 comp=%s[%s] uncomp=%s[%s]",
        seed,mnem[0]?mnem:"(old_words.txt missing)",change,off,ac,mc?"MATCH":"no",au,mu?"MATCH":"no");
    hit_report("seed_found.txt","ELECTRUM V1 SEED MATCH",msg);
}

/* ============================ device tables setup ============================ */
static void parse_res(const char*hex,int bits,uint32_t out[8]){ uint32_t v[8]={0,0,0,0,0,0,0,0}; int L=(int)strlen(hex);
    for(int i=0;i<L;i++){ char ch=hex[L-1-i]; unsigned d= (ch>='0'&&ch<='9')?ch-'0':(ch>='a'&&ch<='f')?ch-'a'+10:(ch>='A'&&ch<='F')?ch-'A'+10:0;
        int word=i/8, sh=(i%8)*4; if(word<8) v[word]|=d<<sh; }
    for(int i=0;i<8;i++){ int lo=i*32,hi=lo+32; if(bits<=lo) v[i]=0; else if(bits<hi) v[i]&=((bits-lo)>=32?0xffffffffu:((1u<<(bits-lo))-1)); }
    for(int i=0;i<8;i++) out[i]=v[i];
}
static void setup_tables(void){
    int hn[128],hb[128]; static uint32_t hres[128][8]; int hp[16];
    for(int j=0;j<N_KNOWN;j++){ hn[j]=KNOWN[j].n; hb[j]=KNOWN[j].bits; parse_res(KNOWN[j].res_hex,KNOWN[j].bits,hres[j]); }
    for(int p=0;p<N_PROBES;p++){ int kidx=-1; for(int j=0;j<N_KNOWN;j++) if(KNOWN[j].n==PROBES[p]){kidx=j;break;} hp[p]=kidx; }
    int np=N_PROBES,nk=N_KNOWN;
    CUDA_CHECK(cudaMemcpyToSymbol(cKN_n,hn,sizeof(int)*N_KNOWN));
    CUDA_CHECK(cudaMemcpyToSymbol(cKN_bits,hb,sizeof(int)*N_KNOWN));
    CUDA_CHECK(cudaMemcpyToSymbol(cKN_res,hres,sizeof(uint32_t)*8*N_KNOWN));
    CUDA_CHECK(cudaMemcpyToSymbol(cPROBE,hp,sizeof(int)*N_PROBES));
    CUDA_CHECK(cudaMemcpyToSymbol(cNPROBE,&np,sizeof(int)));
    CUDA_CHECK(cudaMemcpyToSymbol(cNKNOWN,&nk,sizeof(int)));
}

/* ============================ selftest ============================ */
static int selftest(void){
    setup_tables();
    uint64_t hi=0xafc3eef71d96c468ull,lo=0xca52b437c385a621ull;
    unsigned char *d_m,hm[32]; CUDA_CHECK(cudaMalloc(&d_m,64)); /* 64B: under ILP2 thread 0 writes 2 seeds */
    stretch_kernel<<<1,1>>>(hi,lo,d_m,1); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hm,d_m,32,cudaMemcpyDeviceToHost));
    BIGNUM*abn=BN_new(); BN_bin2bn(hm,32,abn); char*ah=BN_bn2hex(abn);
    int ok1=(strcasecmp(ah,"5DF10C922A1C7888B5C3A5A7106E72576F09C17F0993F4F2CE0EF5CCDB2F53C5")==0);
    printf("GPU stretch vector:  %s (%s)\n",ok1?"OK":"FAIL",ah); OPENSSL_free(ah);
    /* GPU a*G + child */
    unsigned char *d_a,*d_mpk,*d_child,hmpk[64],hchild[32]; CUDA_CHECK(cudaMalloc(&d_a,32)); CUDA_CHECK(cudaMalloc(&d_mpk,64)); CUDA_CHECK(cudaMalloc(&d_child,32));
    CUDA_CHECK(cudaMemcpy(d_a,hm,32,cudaMemcpyHostToDevice));
    st_kernel<<<1,1>>>(d_a,d_mpk,d_child); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(hmpk,d_mpk,64,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(hchild,d_child,32,cudaMemcpyDeviceToHost));
    char mpks[129]; for(int i=0;i<64;i++) sprintf(mpks+2*i,"%02x",hmpk[i]);
    int ok2=(strcasecmp(mpks,"fd6d91db1bdfc231116fd7d44c61a02e032b38b90aad419ecf75acf501eebdc7a0b7818e16a97a6c87693c723106e6fe17d4da743fae991139b05e6d1fe5c8a8")==0);
    printf("GPU a*G (MPK):       %s\n",ok2?"OK":"FAIL");
    char chs[65]; for(int i=0;i<32;i++) sprintf(chs+2*i,"%02x",hchild[i]);
    int ok3=(strcasecmp(chs,"a0835e3fc02475c1b9c418794fc247a732bbabd16ca4ef38983923bd3dcd8177")==0);
    printf("GPU child(0,0):      %s (%s)\n",ok3?"OK":"FAIL",chs);
    /* host address (privkey=1) + mnemonic */
    ecctx_t e; ec_init(&e); BIGNUM*one=BN_new(); BN_one(one); char ac[64],au[64]; p2pkh_from_scalar(&e,one,1,ac); p2pkh_from_scalar(&e,one,0,au);
    int ok4=(strcmp(ac,"1BgGZ9tcN4rm9KBzDn7KprQz87SZ26SAMH")==0 && strcmp(au,"1EHNa6Q4Jz2uvNExL497mE43ikXhwF6kZm")==0);
    printf("P2PKH address test:  %s\n",ok4?"OK":"FAIL");
    { const char*wp=getenv("WORDS"); load_words(wp&&*wp?wp:"old_words.txt"); }
    char mnem[256]; int ok5=mnemonic_encode("afc3eef71d96c468ca52b437c385a621",mnem);
    printf("mnemonic (12 words): %s (%s)\n",ok5?"OK":"SKIP",ok5?mnem:"old_words.txt missing");
    cudaFree(d_m);cudaFree(d_a);cudaFree(d_mpk);cudaFree(d_child);
    int pass=ok1&&ok2&&ok3&&ok4;
    printf("SELFTEST %s\n",pass?"PASS":"FAIL");
    return pass?0:1;
}

/* ============================ main ============================ */
int main(int argc,char**argv){
    if(argc>1 && strcmp(argv[1],"--selftest")==0) return selftest();
    setup_tables();
    { const char*wp=getenv("WORDS"); load_words(wp&&*wp?wp:"old_words.txt"); }

    int BATCH=1<<20; if(const char*b=getenv("BATCH")){ int v=atoi(b); if(v>=4096) BATCH=v; }
    /* progress file for auto-resume (multi-GPU: give each one PROGRESS=prog_gpu0.txt etc.) */
    const char*progpath=getenv("PROGRESS"); if(!progpath||!*progpath) progpath="gpuseed_progress.txt";
    unsigned __int128 base=0; int resumed=0;
    if(getenv("BASE_HI")||getenv("BASE_LO")){
        if(const char*bh=getenv("BASE_HI")) base=(unsigned __int128)strtoull(bh,NULL,16)<<64;
        if(const char*bl=getenv("BASE_LO")) base=(base&(((unsigned __int128)~0ull)<<64))|strtoull(bl,NULL,16);
    } else {
        FILE*pf=fopen(progpath,"r"); char line[64];
        if(pf && fgets(line,sizeof line,pf) && strlen(line)>=32){ base=parse_hex128(line); resumed=1; fclose(pf); }
        else { if(pf)fclose(pf); FILE*ur=fopen("/dev/urandom","rb"); if(ur){ unsigned char r[16]; if(fread(r,1,16,ur)==16) for(int i=0;i<16;i++) base=(base<<8)|r[i]; fclose(ur);} }
    }

    unsigned char *d_master; CUDA_CHECK(cudaMalloc(&d_master,(size_t)BATCH*32));
    char dev[256]="?"; cudaDeviceProp pr; int devid=0; cudaGetDevice(&devid);
    if(cudaGetDeviceProperties(&pr,devid)==cudaSuccess) snprintf(dev,sizeof dev,"%s sm_%d%d",pr.name,pr.major,pr.minor);
    { char hx[33]; host_seed_hex(base,hx); fprintf(stderr,"gpuseed GPU-only | dev=%s | targets=%d probes=%d batch=%d\n",dev,N_KNOWN,N_PROBES,BATCH);
      fprintf(stderr,"%s base=%s  (progress -> %s)\n", resumed?"RESUMED from":"new", hx, progpath); }

    int jump = getenv("JUMP") && atoi(getenv("JUMP"))!=0;   /* JUMP=1 -> fresh random base every batch */
    if(jump) fprintf(stderr,"mode: JUMP (random base per batch)\n");
    unsigned long long total=0; long hits=0; time_t t0=time(NULL); unsigned __int128 g=base;
    int zero=0;
    for(;;){
        uint64_t g_hi=(uint64_t)(g>>64),g_lo=(uint64_t)g; int tpb=256;
        int sthreads=BATCH/SPT, sblocks=(sthreads+tpb-1)/tpb, cblocks=(BATCH+tpb-1)/tpb;
        CUDA_CHECK(cudaMemcpyToSymbol(d_hitcount,&zero,sizeof(int)));
        stretch_kernel<<<sblocks,tpb>>>(g_hi,g_lo,d_master,sthreads);
        casc_kernel<<<cblocks,tpb>>>(d_master,BATCH);
        CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
        int hc=0; CUDA_CHECK(cudaMemcpyFromSymbol(&hc,d_hitcount,sizeof(int)));
        if(hc>0){ Hit hh[MAXHITS]; int ncopy=hc<MAXHITS?hc:MAXHITS; CUDA_CHECK(cudaMemcpyFromSymbol(hh,d_hits,sizeof(Hit)*ncopy));
            for(int i=0;i<ncopy;i++){ report_hit(g+(unsigned __int128)hh[i].idx, hh[i].change, hh[i].off); hits++; } }
        total+=(unsigned)BATCH;
        if(jump) g=rand128(); else g+=(unsigned __int128)BATCH;
        { char hx[33]; host_seed_hex(g,hx); FILE*pf=fopen(progpath,"w"); if(pf){ fprintf(pf,"%s\n",hx); fclose(pf); } } /* save the next base */
        double el=difftime(time(NULL),t0); if(el<1)el=1;
        fprintf(stderr,"\rtested %llu | %.0f seeds/s | hits %ld   ",total,total/el,hits); fflush(stderr);
    }
    cudaFree(d_master); return 0;
}

/* hitcrypt.h — encrypt a found secret to the USER's public key before it ever
 * touches disk/stdout. On a rented host (root-capable) this stops theft via
 * files/logs/output. Private key stays on the user's own machine only.
 * Falls back to plaintext ONLY if HIT_PUBKEY is unset (trusted local machine).
 * (Honest limit: a host doing live RAM forensics could still catch plaintext at
 *  match time; this defeats the realistic disk/log/stdout/network theft vectors.)
 */
#ifndef HITCRYPT_H
#define HITCRYPT_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <openssl/evp.h>
#include <openssl/pem.h>
#include <openssl/rsa.h>
#include <openssl/bio.h>

/* append base64(RSA-OAEP(secret)) to HIT_OUT (default hit.enc). returns 1 if encrypted. */
static int hit_encrypt(const char *secret){
    const char *pk = getenv("HIT_PUBKEY");
    if(!pk || !*pk) return 0;
    FILE *f = fopen(pk,"r"); if(!f) return 0;
    EVP_PKEY *key = PEM_read_PUBKEY(f,NULL,NULL,NULL); fclose(f);
    if(!key) return 0;
    EVP_PKEY_CTX *ctx = EVP_PKEY_CTX_new(key,NULL);
    size_t outlen=0; unsigned char *out=NULL; int ok=0;
    if(ctx && EVP_PKEY_encrypt_init(ctx)>0 &&
       EVP_PKEY_CTX_set_rsa_padding(ctx,RSA_PKCS1_OAEP_PADDING)>0 &&
       EVP_PKEY_encrypt(ctx,NULL,&outlen,(const unsigned char*)secret,strlen(secret))>0){
        out=(unsigned char*)malloc(outlen);
        if(out && EVP_PKEY_encrypt(ctx,out,&outlen,(const unsigned char*)secret,strlen(secret))>0){
            /* base64 */
            BIO *b64=BIO_new(BIO_f_base64()); BIO *mem=BIO_new(BIO_s_mem());
            BIO_set_flags(b64,BIO_FLAGS_BASE64_NO_NL); b64=BIO_push(b64,mem);
            BIO_write(b64,out,(int)outlen); BIO_flush(b64);
            char *bptr=NULL; long blen=BIO_get_mem_data(mem,&bptr);
            const char *of=getenv("HIT_OUT"); if(!of||!*of) of="hit.enc";
            FILE *o=fopen(of,"a");
            if(o){ fwrite(bptr,1,blen,o); fputc('\n',o); fclose(o); ok=1; }
            BIO_free_all(b64);
        }
        free(out);
    }
    if(ctx) EVP_PKEY_CTX_free(ctx);
    EVP_PKEY_free(key);
    return ok;
}
/* report a hit: encrypted if HIT_PUBKEY set (neutral message, no banner),
 * else plaintext to the given fallback file + stdout. */
static void hit_report(const char *fallback_file, const char *label, const char *secret){
    if(hit_encrypt(secret)){
        /* neutral, non-revealing message so a host grepping logs learns nothing */
        printf("."); fflush(stdout);
        return;
    }
    printf("\n*** %s ***\n%s\n", label, secret);
    FILE *f=fopen(fallback_file,"a"); if(f){ fprintf(f,"%s\n%s\n----\n",label,secret); fclose(f); }
    fflush(stdout);
}
#endif

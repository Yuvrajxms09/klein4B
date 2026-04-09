#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../../klein-cuda-c/flux.h"

typedef struct flux_tokenizer flux_tokenizer;
typedef struct flux_vae flux_vae_t;
typedef struct flux_transformer flux_transformer_t;
typedef struct qwen3_encoder qwen3_encoder_t;

typedef struct flux_ctx_internal {
    flux_tokenizer *tokenizer;
    qwen3_encoder_t *qwen3_encoder;
    flux_vae_t *vae;
    flux_transformer_t *transformer;
    int max_width;
    int max_height;
    int default_steps;
    float default_guidance;
    int is_distilled;
    char model_name[64];
    char model_version[32];
    char model_dir[512];
    int use_mmap;
} flux_ctx_internal_t;

static int ensure_transformer_loaded(
    flux_ctx_internal_t *ctx,
    const float *z, int latent_h, int latent_w,
    const float *text_emb, int text_len,
    float timestep) {
    if (ctx->transformer) return 1;
    float *tmp = flux_denoise_step(
        (flux_ctx *)ctx,
        z,
        timestep,
        text_emb,
        text_len,
        latent_h,
        latent_w);
    if (!tmp) return 0;
    free(tmp);
    return ctx->transformer != NULL;
}

extern float *flux_transformer_forward_with_refs(
    flux_transformer_t *tf,
    const float *img_latent, int img_h, int img_w,
    const float *ref_latent, int ref_h, int ref_w,
    int t_offset,
    const float *txt_emb, int txt_seq,
    float timestep);

typedef struct {
    const float *latent;
    int h, w;
    int t_offset;
} flux_ref_t;

extern float *flux_transformer_forward_with_multi_refs(
    flux_transformer_t *tf,
    const float *img_latent, int img_h, int img_w,
    const flux_ref_t *refs, int num_refs,
    const float *txt_emb, int txt_seq,
    float timestep);

void *klein_bridge_load(const char *model_dir, int use_mmap) {
    flux_ctx *ctx = flux_load_dir(model_dir);
    if (!ctx) return NULL;
    flux_set_mmap(ctx, use_mmap ? 1 : 0);
    return (void *)ctx;
}

void klein_bridge_free(void *ctx) {
    if (ctx) flux_free((flux_ctx *)ctx);
}

int klein_bridge_denoise(
    void *ctx_ptr,
    const float *z, int latent_h, int latent_w,
    const float *text_emb, int text_len,
    float timestep,
    float *out) {
    if (!ctx_ptr || !z || !text_emb || !out) return 0;
    float *v = flux_denoise_step(
        (flux_ctx *)ctx_ptr,
        z,
        timestep,
        text_emb,
        text_len,
        latent_h,
        latent_w);
    if (!v) return 0;
    size_t n = (size_t)128 * (size_t)latent_h * (size_t)latent_w;
    memcpy(out, v, n * sizeof(float));
    free(v);
    return 1;
}

int klein_bridge_denoise_with_refs(
    void *ctx_ptr,
    const float *z, int latent_h, int latent_w,
    const float *ref_latent, int ref_h, int ref_w, int t_offset,
    const float *text_emb, int text_len,
    float timestep,
    float *out) {
    if (!ctx_ptr || !z || !text_emb || !out) return 0;
    flux_ctx_internal_t *ctx = (flux_ctx_internal_t *)ctx_ptr;
    if (!ensure_transformer_loaded(ctx, z, latent_h, latent_w, text_emb, text_len, timestep)) return 0;

    float *v = flux_transformer_forward_with_refs(
        ctx->transformer,
        z, latent_h, latent_w,
        ref_latent, ref_h, ref_w,
        t_offset,
        text_emb, text_len,
        timestep);
    if (!v) return 0;

    size_t n = (size_t)128 * (size_t)latent_h * (size_t)latent_w;
    memcpy(out, v, n * sizeof(float));
    free(v);
    return 1;
}

int klein_bridge_denoise_with_multi_refs(
    void *ctx_ptr,
    const float *z, int latent_h, int latent_w,
    const float **ref_latents, const int *ref_h, const int *ref_w, const int *ref_t_offsets, int num_refs,
    const float *text_emb, int text_len,
    float timestep,
    float *out) {
    if (!ctx_ptr || !z || !text_emb || !out) return 0;
    flux_ctx_internal_t *ctx = (flux_ctx_internal_t *)ctx_ptr;
    if (!ensure_transformer_loaded(ctx, z, latent_h, latent_w, text_emb, text_len, timestep)) return 0;
    if (num_refs < 0) return 0;
    if (num_refs > 0 && (!ref_latents || !ref_h || !ref_w || !ref_t_offsets)) return 0;

    flux_ref_t *refs = NULL;
    if (num_refs > 0) {
        refs = (flux_ref_t *)malloc((size_t)num_refs * sizeof(flux_ref_t));
        if (!refs) return 0;
        for (int i = 0; i < num_refs; i++) {
            refs[i].latent = ref_latents[i];
            refs[i].h = ref_h[i];
            refs[i].w = ref_w[i];
            refs[i].t_offset = ref_t_offsets[i];
        }
    }

    float *v = flux_transformer_forward_with_multi_refs(
        ctx->transformer,
        z, latent_h, latent_w,
        refs, num_refs,
        text_emb, text_len,
        timestep);

    if (refs) free(refs);
    if (!v) return 0;

    size_t n = (size_t)128 * (size_t)latent_h * (size_t)latent_w;
    memcpy(out, v, n * sizeof(float));
    free(v);
    return 1;
}

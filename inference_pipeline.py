get_ipython().system('pip install -q dalle-mini')
get_ipython().system('pip install -q git+https://github.com/patil-suraj/vqgan-jax.git')




DALLE_MODEL = "dalle-mini/dalle-mini/mega-1-fp16:latest"  
DALLE_COMMIT_ID = None

VQGAN_REPO = "dalle-mini/vqgan_imagenet_f16_16384"
VQGAN_COMMIT_ID = "e93a26e7707683d349bf5d5c41c5b0ef69b677a9"





import jax
import jax.numpy as jnp
import streamlit as st


jax.local_device_count()


from dalle_mini import DalleBart, DalleBartProcessor
from vqgan_jax.modeling_flax_vqgan import VQModel
from transformers import CLIPProcessor, FlaxCLIPModel


model, params = DalleBart.from_pretrained(
    DALLE_MODEL, revision=DALLE_COMMIT_ID, dtype=jnp.float16, _do_init=False
)


vqgan, vqgan_params = VQModel.from_pretrained(
    VQGAN_REPO, revision=VQGAN_COMMIT_ID, _do_init=False
)




from flax.jax_utils import replicate

params = replicate(params)
vqgan_params = replicate(vqgan_params)



from functools import partial


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(3, 4, 5, 6))
def p_generate(
    tokenized_prompt, key, params, top_k, top_p, temperature, condition_scale
):
    return model.generate(
        **tokenized_prompt,
        prng_key=key,
        params=params,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        condition_scale=condition_scale,
    )



@partial(jax.pmap, axis_name="batch")
def p_decode(indices, params):
    return vqgan.decode_code(indices, params=params)


import random


seed = random.randint(0, 2**32 - 1)
key = jax.random.PRNGKey(seed)



from dalle_mini import DalleBartProcessor

processor = DalleBartProcessor.from_pretrained(DALLE_MODEL, revision=DALLE_COMMIT_ID)

title = st.text_input("𝗪𝗛𝗔𝗧'𝗦 𝗜𝗡 𝗬𝗢𝗨𝗥 𝗠𝗜𝗡𝗗 ❓" ,)
prompts = [titles]



tokenized_prompts = processor(prompts)




tokenized_prompt = replicate(tokenized_prompts)


n_predictions = 8
gen_top_k = None
gen_top_p = None
temperature = None
cond_scale = 10.0





from flax.training.common_utils import shard_prng_key
import numpy as np
from PIL import Image
from tqdm.notebook import trange

print(f"Prompts: {prompts}\n")

images = []
for i in trange(max(n_predictions // jax.device_count(), 1)):

    key, subkey = jax.random.split(key)

    encoded_images = p_generate(
        tokenized_prompt,
        shard_prng_key(subkey),
        params,
        gen_top_k,
        gen_top_p,
        temperature,
        cond_scale,
    )
    # remove BOS
    encoded_images = encoded_images.sequences[..., 1:]
    # decode images
    decoded_images = p_decode(encoded_images, vqgan_params)
    decoded_images = decoded_images.clip(0.0, 1.0).reshape((-1, 256, 256, 3))
    for decoded_img in decoded_images:
        img = Image.fromarray(np.asarray(decoded_img * 255, dtype=np.uint8))
        images.append(img)
        display(img)
        print()




CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_COMMIT_ID = None


clip, clip_params = FlaxCLIPModel.from_pretrained(
    CLIP_REPO, revision=CLIP_COMMIT_ID, dtype=jnp.float16, _do_init=False
)
clip_processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_COMMIT_ID)
clip_params = replicate(clip_params)


@partial(jax.pmap, axis_name="batch")
def p_clip(inputs, params):
    logits = clip(params=params, **inputs).logits_per_image
    return logits




from flax.training.common_utils import shard


clip_inputs = clip_processor(
    text=prompts * jax.device_count(),
    images=images,
    return_tensors="np",
    padding="max_length",
    max_length=77,
    truncation=True,
).data
logits = p_clip(shard(clip_inputs), clip_params)


p = len(prompts)
logits = np.asarray([logits[:, i::p, i] for i in range(p)]).squeeze()




for i, prompt in enumerate(prompts):
    print(f"Prompt: {prompt}\n")
    for idx in logits[i].argsort()[::-1]:
        display(images[idx * p + i])
        print(f"Score: {jnp.asarray(logits[i][idx], dtype=jnp.float32):.2f}\n")
    print()



import wandb


project = 'dalle-mini-tables-colab'
run = wandb.init(project=project)


columns = ["captions"] + [f"image_{i+1}" for i in range(n_predictions)]
gen_table = wandb.Table(columns=columns)


for i, prompt in enumerate(prompts):

    if logits is not None:
        idxs = logits[i].argsort()[::-1]
        tmp_imgs = images[i::len(prompts)]
        tmp_imgs = [tmp_imgs[idx] for idx in idxs]
    else:
        tmp_imgs = images[i::len(prompts)]


    gen_table.add_data(prompt, *[wandb.Image(img) for img in tmp_imgs])


wandb.log({"Generated Images": gen_table})


run.finish()


import io
import os

import modal

app = modal.App()


@app.function(
    image=modal.Image.debian_slim().pip_install("torch", "diffusers[torch]", "transformers", "ftfy"),
    secrets=[modal.Secret.from_name("huggingface")],
    gpu="any",
)
async def run_stable_diffusion(prompt: str):
    from diffusers import StableDiffusionPipeline

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        use_auth_token=os.environ["HF_TOKEN"],
    ).to("cuda")

    image = pipe(prompt, num_inference_steps=10).images[0]

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    return img_bytes


@app.local_entrypoint()
def main():
    img_bytes = run_stable_diffusion.remote("")
    with open("/tmp/parisian-beagle.png", "wb") as f:
        f.write(img_bytes)
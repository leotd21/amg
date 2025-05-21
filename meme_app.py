import torch
import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler

DEFAULT_MODEL_NAME = "stabilityai/sd-turbo"
GPU_MODEL_NAME = "CompVis/stable-diffusion-v1-4"

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
MODEL_NAME = GPU_MODEL_NAME if device == "cuda" else DEFAULT_MODEL_NAME

# Style dictionary
style_dict = {
    "none": "",
    "anime": "cartoon, animated, Studio Ghibli style, cute, Japanese animation",
    "photo": "photograph, film, 35 mm camera",
    "video game": "rendered in unreal engine, hyper-realistic, volumetric lighting",
    "watercolor": "painting, watercolors, pastel, composition",
}


@st.cache_resource
def load_model():
    scheduler = EulerDiscreteScheduler.from_pretrained(
        MODEL_NAME, subfolder="scheduler"
    )
    pipeline = StableDiffusionPipeline.from_pretrained(
        MODEL_NAME,
        scheduler=scheduler,
        torch_dtype=dtype
    )
    pipeline.to(device)
    return pipeline


def generate_image(prompt, pipeline, guidance=7.5, steps=10, style="none"):
    styled_prompt = f"{prompt}, {style_dict.get(style, '')}".strip(", ")
    result = pipeline(
        styled_prompt,
        guidance_scale=guidance,
        num_inference_steps=steps
    )
    if not result.images:
        raise RuntimeError("Image generation failed.")
    return result.images[0]


def add_text_to_image(image, text, text_color="white", outline_color="black",
                      font_size=50, border_width=2, font_path="arial.ttf"):
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()

    width, height = image.size
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    x = (width - text_width) / 2
    y = (height - text_height) / 2

    draw.text((x, y), text, font=font, fill=text_color,
              stroke_width=border_width, stroke_fill=outline_color)
    return image


def sidebar_controls():
    st.sidebar.title("Meme Controls")
    prompt = st.sidebar.text_area("Text-to-Image Prompt", placeholder="study with corgi")
    caption = st.sidebar.text_area("Meme Caption", placeholder="Go!")
    guidance = st.sidebar.slider("Guidance Scale", 2.0, 10.0, 5.0)
    steps = st.sidebar.slider("Inference Steps", 4, 50, 4)
    style = st.sidebar.selectbox("Style", options=style_dict.keys())
    generate = st.sidebar.button("Generate Meme")
    return prompt, caption, guidance, steps, style, generate


def main():
    st.set_page_config(page_title="Meme Generator", layout="centered")
    st.title("🎈 Awesome Meme For Life")

    prompt, caption, guidance, steps, style, generate = sidebar_controls()

    if generate:
        if not prompt:
            st.error("Please enter a prompt.")
            return
        if not caption:
            st.error("Please enter meme caption text.")
            return

        with st.spinner("Generating meme..."):
            try:
                pipeline = load_model()
                image = generate_image(prompt, pipeline, guidance, steps, style)
                print("image generated! Adding text..")
                image = add_text_to_image(image, caption)
                st.image(image, caption="Your Meme", use_column_width=True)

                # Download button
                from io import BytesIO
                buf = BytesIO()
                image.save(buf, format="PNG")
                st.download_button("Download Meme", data=buf.getvalue(),
                                   file_name="meme.png", mime="image/png")

                st.success("Meme generated successfully!")
            except Exception as e:
                st.error(f"Error: {e}")


if __name__ == "__main__":
    main()

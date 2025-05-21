import diffusers
import streamlit as st
import torch
from PIL import ImageDraw, ImageFont

if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using {device} device with {dtype} data type.")

# The dictionary mapping style names to style strings
style_dict = {
    "none": "",
    "anime": "cartoon, animated, Studio Ghibli style, cute, Japanese animation",
    # A photograph on film suggests an artistic approach
    "photo": "photograph, film, 35 mm camera",
    "video game": "rendered in unreal engine, hyper-realistic, volumetric lighting, --ar 9:16 --hd --q 2",
    "watercolor": "painting, watercolors, pastel, composition",
}

@st.cache_resource
def load_model():
    pipeline = diffusers.AutoPipelineForText2Image.from_pretrained(
        "CompVis/stable-diffusion-v1-4", torch_dtype=dtype
    )
    pipeline.to(device)
    return pipeline


def generate_images(prompt, pipeline, n, guidance=7.5, steps=50, style="none"):
    styled_prompts = f"{prompt} {style}"
    return pipeline(
        [styled_prompts]*n, guidance_scale=guidance, num_inference_steps=steps
    ).images


def add_text_to_image(image, text, text_color="white", outline_color="black",
                      font_size=50, border_width=2, font_path="arial.ttf"):
    # Initialization
    font = ImageFont.truetype(font_path, size=font_size)
    draw = ImageDraw.Draw(image)
    width, height = image.size

    # Calculate the size of the text
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Calculate the position at which to draw the text to center it
    x = (width - text_width) / 2
    y = (height - text_height) / 2

    # Draw text
    draw.text((x, y), text, font=font, fill=text_color,
              stroke_width=border_width, stroke_fill=outline_color)


def generate_memes(prompt, text, pipeline, n, guidance, steps, style):
    # images = generate_images(prompt, pipeline, n)
    images = generate_images(
        prompt, pipeline, n, guidance, steps, style
    )
    for img in images:
        add_text_to_image(img, text)

    return images


def main():
    st.title("Your awesome meme generator")
    
    with st.sidebar:
        num_images = st.sidebar.number_input("Number of Images", min_value=1, max_value=2)
        prompt = st.sidebar.text_area("Text-to-Image Prompt")

        guidance_help = "Lower values follow the prompt less strictly. Higher values risk distored images."
        guidance = st.sidebar.slider("Guidance", 2.0, 15.0, 7.5, help=guidance_help)

        steps_help = "More steps produces better images but takes longer."
        steps = st.sidebar.slider("Steps", 10, 150, 50, help=steps_help)

        style = st.sidebar.selectbox("Style", options=style_dict.keys())

        generate = st.sidebar.button("Generate Meme")

    if generate:
        if not prompt:
            st.error("Please enter a prompt")
        elif not text:
            st.error("Please enter the text")
        else:
            with st.spinner("Generating images..."):
                pipeline = load_model()
                images = generate_memes(prompt, text, pipeline, num_images, guidance, steps, style)
                st.subheader("Generated images")
                for img in images:
                    st.image(img)
                # st.text_area(f"{num_images} images with prompt {prompt} / {text}")


if __name__ == '__main__':
    main()
    
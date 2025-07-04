from ollama import chat
from PIL import Image
import base64
import io

def encode_image_to_base64(img_path: str) -> str:
    img = Image.open(img_path).convert("RGB")
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode()

def query_gemma(model: str, prompt: str, image_path: str) -> str:
    img_b64 = encode_image_to_base64(image_path)
    response = chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt, "images": [img_b64]}
        ]
    )
    return response.message.content

if __name__ == "__main__":
    model = "gemma3n:e2b"  # or gemma3:12b / gemma3:27b
    prompt = "Describe what's in this image, including details and background."
    image_path = r"C:\Users\Yatharth\Desktop\desktop1\AI\Gemma Hack\Screenshot 2025-04-12 235058.jpg"
    output = query_gemma(model, prompt, image_path)
    print("Gemma response:\n", output)

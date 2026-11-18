import base64
import ollama


VLM_MODEL = "gemma4:e2b"

SYSTEM_PROMPT = (
    "You are an expert tennis analyst assistant integrated into a real-time AI sports analysis system. "
    "You have access to computer vision outputs: player bounding boxes, court keypoint landmarks, "
    "and ball trajectory predictions. "
    "When given an image, analyse the tennis scene precisely: describe player positions, "
    "court area visible, shot type if inferable, and any tactical observations. "
    "Keep answers concise and factual. If no image is provided, answer from context."
)


def encode_image(uploaded_file):
    uploaded_file.seek(0)
    return base64.b64encode(uploaded_file.read()).decode("utf-8")


def test_vlm_connection():
    try:
        response = ollama.chat(
            model=VLM_MODEL,
            messages=[{"role": "user", "content": "Reply with only: VLM connected"}],
        )
        return True, response["message"]["content"]
    except Exception as e:
        return False, str(e)


def ask_vlm(prompt, image_file=None, detection_context: str | None = None):
    """
    Args:
        prompt          : user question
        image_file      : optional uploaded file object (Streamlit UploadedFile)
        detection_context : optional string summarising current detections to inject
    """
    full_prompt = prompt
    if detection_context:
        full_prompt = f"[Detection context: {detection_context}]\n\n{prompt}"

    user_message: dict = {"role": "user", "content": full_prompt}

    if image_file is not None:
        image_base64 = encode_image(image_file)
        user_message["images"] = [image_base64]

    response = ollama.chat(
        model=VLM_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            user_message,
        ],
    )
    return response["message"]["content"]
from openai import OpenAI
import base64
import cv2

class ObjectDetection:
    def __init__(self):
        self.API_KEY = "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"
        self.client = OpenAI(api_key=self.API_KEY)

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def infer_image(self, image_path):
        base64_image = self.encode_image(image_path)

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Determine the material of the object in focus either as metal, plastic, paper or glass. Give a one word answer."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpg;base64,{base64_image}",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        return response.choices[0].message.content.strip().lower()

    def infer_image_from_cv2(self, image):
        cv2.imwrite("object.jpg", image)
        return self.infer_image("object.jpg")
    
if __name__ == "__main__":
    object_detector = ObjectDetection()
    
    # Capture image from camera
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()

    # Run inference on captured image
    camera.release()

    material = object_detector.infer_image_from_cv2(frame)
    # Release camera

    # Print the result
    print(material)
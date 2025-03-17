import cv2
import os
import torch
import argparse
from torchvision import transforms
from PIL import Image
from generator import Generator
from train import denormalize

def extract_frames(video_path, output_folder, frame_rate):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)

    frame_count = 0
    saved_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_path = os.path.join(output_folder, f"frame_{saved_count:05d}.png")
            cv2.imwrite(frame_path, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames at {frame_rate} FPS.")

def run_model_on_frames(input_folder, output_folder, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(output_folder, exist_ok=True)

    model = Generator()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((256, 256)),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for filename in sorted(os.listdir(input_folder)):
        if filename.endswith(".png"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)

            image = Image.open(input_path).convert("RGB")
            image = transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(image)

            output = denormalize(output, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            output_image = transforms.ToPILImage()(output.squeeze(0).cpu())
            output_image.save(output_path)

            print(f"Processed: {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames and run pixel model")
    parser.add_argument("--video", required=True, help="Path to input MP4 file")
    parser.add_argument("--frame_rate", type=int, default=10, help="FPS for extraction")
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--output_frames", default="frames", help="Folder for extracted frames")
    parser.add_argument("--output_pixels", default="pixel_outputs", help="Folder for processed images")
    args = parser.parse_args()

    extract_frames(args.video, args.output_frames, args.frame_rate)
    run_model_on_frames(args.output_frames, args.output_pixels, args.model)

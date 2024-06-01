import os
import cv2
from ultralytics import YOLO
import math
import cvzone
import easyocr

classNames = ["Billboard"]
reader = easyocr.Reader(['en'])


def extract_frames(video_path, output_folder, interval=1):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    interval_frames = int(frame_rate * interval)

    frame_count = 0
    saved_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval_frames == 0:
            frame_filename = f"{output_folder}/frame_{saved_frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def detect_billboards(frame, model):
    results = model(frame, stream=True)
    detected_results = []

    frame_height, frame_width, _ = frame.shape
    print(f'Frame dimensions: width={frame_width} pixels, height={frame_height} pixels')

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cls1 = int(box.cls[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            width = x2 - x1
            height = y2 - y1
            cvzone.putTextRect(frame, f'{conf}{"Billboard"}',
                               (max(0, x1), max(10, y1)),
                               thickness=3, offset=2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 4)
            frame = cv2.resize(frame, (400, 600))
            cv2.imshow("frame", frame)
            cv2.waitKey(1)
            print(conf)
            cls = classNames[cls1]

            # Calculate the center of the billboard
            billboard_center_x = (x1 + x2) // 2
            billboard_center_y = (y1 + y2) // 2

            result = x1, y1, x2, y2, conf, cls, width, height, billboard_center_x, billboard_center_y
            if conf >= 0.70:
                detected_results.append(result)
    return detected_results


def crop_billboards(frame, result, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cropped_images = []
    for i, (x1, y1, x2, y2, conf, cls, width, height, _, _) in enumerate(result):
        cropped_image = frame[int(y1):int(y2), int(x1):int(x2)]
        cropped_image_path = f"{output_folder}/cropped_{i}.jpg"
        cv2.imwrite(cropped_image_path, cropped_image)
        cropped_images.append(cropped_image_path)
    return cropped_images


def extract_text_from_image(image_path):
    image = cv2.imread(image_path)
    result = reader.readtext(image)
    text = " ".join([res[1] for res in result])
    return text


def calculate_distance(frame_width, frame_height, billboard_center_x, billboard_center_y):
    frame_center_x = frame_width // 2
    frame_bottom_y = frame_height

    distance = math.sqrt((billboard_center_x - frame_center_x) ** 2 + (billboard_center_y - frame_bottom_y) ** 2)
    return distance


def process_video(video_path, model_path, output_folder, interval=1):
    frames_folder = os.path.join(output_folder, 'frames')
    cropped_folder = os.path.join(output_folder, 'cropped')

    extract_frames(video_path, frames_folder, interval)
    model = YOLO(model_path)

    texts = []
    for frame_filename in sorted(os.listdir(frames_folder)):
        frame_path = os.path.join(frames_folder, frame_filename)
        frame = cv2.imread(frame_path)
        results = detect_billboards(frame, model)
        cropped_images = crop_billboards(frame, results, cropped_folder)
        for image_path in cropped_images:
            text = extract_text_from_image(image_path)
            texts.append(text)
        for result in results:
            x1, y1, x2, y2, conf, cls, width, height, billboard_center_x, billboard_center_y = result
            print(f'Billboard dimensions: width={width} pixels, height={height} pixels')

            # Calculate and print the distance from the middle bottom edge
            print()
            distance = calculate_distance(frame.shape[1], frame.shape[0], billboard_center_x, billboard_center_y)
            print(f'Distance from middle bottom edge: {distance:.2f} pixels')

    return texts


# Example usage
video_path = "Videos/Billboard.mp4"
model_path = "weights/best.pt"
output_folder = "Output"
texts = process_video(video_path, model_path, output_folder, interval=1)
for text in texts:
    print(text)

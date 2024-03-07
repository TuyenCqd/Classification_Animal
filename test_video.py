import torch.nn as nn
import numpy as np
import torch
from model import CNN
import argparse
import cv2


def get_args():
    parser = argparse.ArgumentParser(description="Train an CNN model")
    parser.add_argument("--video_path", "-i", type=str, default="test_2.mp4")
    parser.add_argument("--frame_size", "-s", type=int, default=224)
    parser.add_argument("--checkpoint_path", "-m", type=str, default="animal_checkpoints/best.pt")
    parser.add_argument("--output_path", "-o", type=str, default="demo.mp4")
    args = parser.parse_args()
    return args


def test(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = ["butterfly", "cat", "chicken", "cow", "dog", "elephant", "horse", "sheep", "spider", "squirrel"]
    model = CNN(num_classes=len(classes)).to(device)
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    cap = cv2.VideoCapture(args.video_path)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    out = cv2.VideoWriter(args.output_path, cv2.VideoWriter_fourcc(*"MJPG"), int(cap.get(cv2.CAP_PROP_FPS)),
                          (width, height))
    while cap.isOpened():
        flag, ori_frame = cap.read()
        if not flag:
            break
        # Preprocess frame
        frame = cv2.cvtColor(ori_frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (args.frame_size, args.frame_size))
        frame = np.transpose(frame, (2, 0, 1)) / 255.
        # frame = np.expand_dims(frame, axis=0)
        frame = frame[None, :, :, :]
        frame = torch.from_numpy(frame).float()
        frame = frame.to(device)
        softmax = nn.Softmax()
        with torch.no_grad():
            output = model(frame)
            prob = softmax(output)
        predicted_class = classes[torch.argmax(output)]
        predicted_prob = prob[0, torch.argmax(output)]
        # print("The frame is about {} with probability of {}".format(predicted_class, predicted_prob))
        cv2.putText(ori_frame, predicted_class, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        out.write(ori_frame)
    cap.release()
    out.release()


if __name__ == '__main__':
    args = get_args()
    test(args)
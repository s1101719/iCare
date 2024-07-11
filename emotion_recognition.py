import cv2
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image
import os

# 定義該情緒的中文字
text_obj = {
    'angry': '生氣',
    'disgust': '噁心',
    'fear': '害怕',
    'happy': '開心',
    'sad': '難過',
    'surprise': '驚訝',
    'neutral': '正常'
}

# 定義加入文字函式
def putText(img, x, y, text, size=50, color=(255, 255, 255)):
    fontpath = 'NotoSansTC-Regular.otf'  # 字型
    if not os.path.exists(fontpath):
        fontpath = None
    try:
        if fontpath:
            font = ImageFont.truetype(fontpath, size)  # 定義字型與文字大小
        else:
            font = ImageFont.load_default()  # 使用預設字型
    except IOError:
        font = ImageFont.load_default()  # 使用預設字型
    imgPil = Image.fromarray(img)  # 轉換成 PIL 影像物件
    draw = ImageDraw.Draw(imgPil)  # 定義繪圖物件
    draw.text((x, y), text, fill=color)  # 加入文字
    return np.array(imgPil)  # 轉換成 np.array

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot receive frame")
        break
    img = cv2.resize(frame, (384, 240))
    try:
        analyze = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
        if isinstance(analyze, list):
            analyze = analyze[0]  # 確保返回值為字典而不是列表
        emotion = analyze['dominant_emotion']  # 取得情緒文字
        emotion_chinese = text_obj.get(emotion, '未知')  # 確保情緒文字存在於字典中
        img = putText(img, 0, 40, emotion_chinese)  # 放入文字
        print(emotion_chinese)
    except Exception as e:
        print(f"Error: {e}")
        pass
    cv2.imshow('oxxostudio', img)
    if cv2.waitKey(5) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

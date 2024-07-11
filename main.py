import os
import time
import threading
import queue
import azure.cognitiveservices.speech as speechsdk
from flask import Flask, render_template, request, jsonify, url_for, send_from_directory
import openai
import uuid
import requests
import json
import cv2
import numpy as np
from deepface import DeepFace
from PIL import ImageFont, ImageDraw, Image

# 變數定義
subscription_key = 'your_subscription_key'
region = 'eastus2'
openai_api_key = "your_openai_api_key"

app = Flask(__name__)
app.secret_key = 'JQDxzgL0hjIHBO-xcCwGMw' 

# 線程安全的佇列用來存儲結果 URL
result_queue = queue.Queue()
#result_queue.put("https://studio.d-id.com/share?id=401404189a5049f2d9546e8a48e32fa1&utm_source=copy")
# 設定 GPT API
def call_gpt(text, emotion):
    openai.api_key = openai_api_key
    print(emotion)
    response = openai.ChatCompletion.create(
        model ="gpt-4o",
        messages=[
            {"role": "system", "content": f"你的名字是小妤，使用者為年長者，你須扮演一個陪伴年長者的角色，依據 user 內容及{emotion}給予適當回應，請以較口語化的方式回應，並請勿反駁使用者所說的內容，結果請以繁體中文"},
            {"role": "user", "content": text}
        ]
    )
    return response.choices[0].message.content

# 語音辨識類別
class ContinuousRecognizer:
    def __init__(self):
        self.speech_config = speechsdk.SpeechConfig(subscription=subscription_key, region=region)
        self.speech_config.speech_recognition_language="zh-TW"
        self.audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
        self.recognizer = speechsdk.SpeechRecognizer(speech_config=self.speech_config, audio_config=self.audio_config)
        self.last_speech_time = time.time()
        self.timer = None
        self.audio_path = None
        self.current_emotion = "未知"

    def start(self):
        self.recognizer.recognized.connect(self.recognized)
        self.recognizer.start_continuous_recognition()
        threading.Thread(target=self.detect_emotion).start()

    def recognized(self, args):
        if args.result.reason == speechsdk.ResultReason.RecognizedSpeech:
            self.last_speech_time = time.time()
            print("Recognized: {}".format(args.result.text))
            if self.timer:
                self.timer.cancel()
            self.timer = threading.Timer(3.0, self.respond, [args.result.text])
            self.timer.start()

    def respond(self, text):
        try:
            if text.strip():  # 確認文本不為空
                print(f"Responding to text: {text}")  # 確認此函數被調用
                response_text = call_gpt(text, self.current_emotion)
                print("GPT Response: {}".format(response_text))  # 檢查 GPT 回應
                result_url = create_did(response_text)
                print("Result URL: {}".format(result_url))  # 確認 URL
                result_queue.put(result_url)  # 將 URL 放入佇列中
        except Exception as e:
            print(f"Error in respond: {e}")  # 打印任何錯誤
    
    def detect_emotion(self):
        text_obj = {
            'angry': '生氣',
            'disgust': '噁心',
            'fear': '害怕',
            'happy': '開心',
            'sad': '難過',
            'surprise': '驚訝',
            'neutral': '正常'
        }
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Cannot open camera")
            return

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
                self.current_emotion = text_obj.get(analyze['dominant_emotion'], '未知')  # 取得情緒文字
                #print(f"Detected Emotion: {self.current_emotion}")
            except Exception as e:
                print(f"Error: {e}")
                pass
            cv2.imshow('Emotion Recognition', img)
            if cv2.waitKey(10) == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# 建立 D-ID URL
def create_did(text):
    
    url = "https://api.d-id.com/talks"

    payload = {
    "script": {
        "type": "text",
        "subtitles": "false",
        "provider": {
            "type": "microsoft",
            "voice_id": "zh-CN-XiaoxiaoNeural"
        },
        "ssml": "false",
        "input": text
    },
    "config": {
        "fluent": "false",
        "pad_audio": "0.0"
    },
    "source_url": "../test.png"
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": "Bearer key"
    }

    response = requests.post(url, json=payload, headers=headers)
    response_json = json.loads(response.text)
    print(response_json)

    talk_id = response_json['id']
    url = "https://api.d-id.com/talks/"+talk_id

    headers = {
        "accept": "application/json",
        "authorization": "Bearer key"
    }
    while True:
        response = requests.get(url, headers=headers)
        response_json = json.loads(response.text)
        status = response_json['status']
        if status == 'done':
            break
    
    result_url = response_json['result_url']
    
    print(result_url)
    return result_url

# Flask 路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route("/start_recording", methods=["POST"])
def recognize_from_microphone():
    recognizer = ContinuousRecognizer()
    recognizer.start()
    return jsonify({"status": "recognition started"})

@app.route('/get_result_url', methods=['GET'])
def get_result_url():
    try:
        result_url = result_queue.get_nowait() 
    except queue.Empty:
        return jsonify({"error": "影片尚未準備好"})
    else:
        return jsonify({"result_url": result_url})

if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', use_reloader=False)





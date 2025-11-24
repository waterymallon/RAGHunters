# RAGHunters 캡스톤 앱 프로젝트

- 원본은 [tensorflow lite 안드로이드 앱 데모](https://github.com/tensorflow/examples/tree/master/lite/examples/object_detection/android)
- 이를 yolov11 모델로도 돌릴 수 있게 한 [깃허브 링크](https://github.com/estebanuri/pub-yolo-android). 이를 베이스로 시작했습니다.

사용한 모델은 best_int8.tflite (양자화된 YOLOv11 모델)입니다.
### 1. 첫번째 탭 카메라
- 실시간으로 객체 탐지 결과를 확인할 수 있습니다. 모델에서 YOLOv11을 골라야 저희의 모델을 선택합니다. 다른 모델은 COCO 기반 기본 탑재 tflite입니다.
  
![KakaoTalk_20251123_222947305](https://github.com/user-attachments/assets/7ea3e615-d0f5-465c-b4c3-9b448334c8e1)

### 2. 두번째 탭 챗봇
- 카메라 탭에서 캡처 버큰을 누르면 해당 이미지의 객체 탐지 정보를 챗봇에 넘깁니다. 이는 추후 RAG를 사용해 HSCode 조회에 사용할 예정입니다.
  
![KakaoTalk_20251123_235844133](https://github.com/user-attachments/assets/cb6efb66-5a15-4fdb-8f30-b2f9a0d0d7d5)

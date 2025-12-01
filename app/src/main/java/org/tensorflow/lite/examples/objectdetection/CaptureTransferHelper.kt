package org.tensorflow.lite.examples.objectdetection

import android.app.Activity
import android.content.Context
import android.graphics.*
import android.os.Handler
import android.os.Looper
import android.os.SystemClock
import android.util.Log
import android.widget.Toast
import androidx.navigation.NavController
import com.google.android.material.tabs.TabLayout
import org.tensorflow.lite.examples.objectdetection.detectors.ObjectDetection
import org.tensorflow.lite.examples.objectdetection.detectors.YoloDetector
import org.tensorflow.lite.support.image.TensorImage

class CaptureTransferHelper(private val context: Context) {

    private val TAG = "CaptureTransferHelper"

    // [중요] strings.xml의 모델 순서에 맞춰 YOLO의 인덱스를 지정하세요.
    // 기존 0~3번이 구글 모델이고, 4번째에 Yolo v11을 추가했다고 가정합니다.
    private val MODEL_YOLO_INDEX = 4

    fun processAndNavigateToChatbot(
        bitmapBuffer: Bitmap,
        imageRotation: Int,
        currentModel: Int,
        currentDelegate: Int,
        threshold: Float,
        numThreads: Int,
        maxResults: Int,
        sharedViewModel: SharedViewModel,
        navController: NavController,
        activity: Activity?
    ) {
        Toast.makeText(context, "이미지 정밀 분석 중...", Toast.LENGTH_SHORT).show()

        // 1. 이미지 물리적 회전 (정방향 만들기)
        val uprightBitmap = rotateBitmap(bitmapBuffer, imageRotation.toFloat())

        // 2. 모델 선택에 따른 분기 처리
        if (currentModel == MODEL_YOLO_INDEX) {
            // === [A] YOLO 모델 사용 시 ===
            runYoloAnalysis(
                bitmap = uprightBitmap,
                currentModel = currentModel,
                currentDelegate = currentDelegate,
                threshold = threshold,
                numThreads = numThreads,
                maxResults = maxResults,
                sharedViewModel = sharedViewModel,
                navController = navController,
                activity = activity
            )
        } else {
            // === [B] 기존 Google 예제 모델 사용 시 ===
            runStandardAnalysis(
                bitmap = uprightBitmap,
                currentModel = currentModel,
                currentDelegate = currentDelegate,
                threshold = threshold,
                numThreads = numThreads,
                maxResults = maxResults,
                sharedViewModel = sharedViewModel,
                navController = navController,
                activity = activity
            )
        }
    }

    // --- [A] 커스텀 YOLO 분석 로직 ---
    private fun runYoloAnalysis(
        bitmap: Bitmap,
        currentModel: Int,
        currentDelegate: Int,
        threshold: Float,
        numThreads: Int,
        maxResults: Int,
        sharedViewModel: SharedViewModel,
        navController: NavController,
        activity: Activity?
    ) {
        // 백그라운드 스레드에서 실행 (UI 블로킹 방지)
        Thread {
            try {
                val startTime = SystemClock.uptimeMillis()

                // 1. YoloDetector 인스턴스 생성
                // (사용자가 제공한 YoloDetector 클래스 생성자 시그니처에 맞춤)
                val detector = YoloDetector(
                    confidenceThreshold = threshold,
                    // iouThreshold는 기본값 혹은 필요시 파라미터화
                    iouThreshold = 0.5f,
                    numThreads = numThreads,
                    maxResults = maxResults,
                    currentDelegate = currentDelegate,
                    currentModel = currentModel,
                    context = context
                )

                // 2. TensorImage 변환
                val tensorImage = TensorImage.fromBitmap(bitmap)

                // 3. 추론 실행 (이미 회전된 비트맵이므로 rotation=0)
                val result = detector.detect(tensorImage, 0)

                val inferenceTime = SystemClock.uptimeMillis() - startTime
                val detections = result.detections // DetectionResult 내부의 List<ObjectDetection>

                // 4. 결과 처리 (UI 스레드로 전달)
                handleAnalysisResult(
                    bitmap = bitmap,
                    results = detections,
                    inferenceTime = inferenceTime,
                    currentModel = currentModel,
                    currentDelegate = currentDelegate,
                    sharedViewModel = sharedViewModel,
                    navController = navController,
                    activity = activity
                )

            } catch (e: Exception) {
                Log.e(TAG, "YOLO Analysis failed", e)
                Handler(Looper.getMainLooper()).post {
                    Toast.makeText(context, "YOLO 분석 실패: ${e.message}", Toast.LENGTH_SHORT).show()
                }
            }
        }.start()
    }

    // --- [B] 기존 Standard (ObjectDetectorHelper) 분석 로직 ---
    private fun runStandardAnalysis(
        bitmap: Bitmap,
        currentModel: Int,
        currentDelegate: Int,
        threshold: Float,
        numThreads: Int,
        maxResults: Int,
        sharedViewModel: SharedViewModel,
        navController: NavController,
        activity: Activity?
    ) {
        val staticDetector = ObjectDetectorHelper(
            context = context,
            objectDetectorListener = object : ObjectDetectorHelper.DetectorListener {
                override fun onError(error: String) {
                    Log.e(TAG, "Standard Analysis failed: $error")
                }

                override fun onResults(
                    results: List<ObjectDetection>,
                    inferenceTime: Long,
                    imageHeight: Int,
                    imageWidth: Int
                ) {
                    handleAnalysisResult(
                        bitmap = bitmap,
                        results = results ?: emptyList(),
                        inferenceTime = inferenceTime,
                        currentModel = currentModel,
                        currentDelegate = currentDelegate,
                        sharedViewModel = sharedViewModel,
                        navController = navController,
                        activity = activity
                    )
                }
            }
        )

        // 설정 적용
        staticDetector.currentModel = currentModel
        staticDetector.currentDelegate = currentDelegate
        staticDetector.threshold = threshold
        staticDetector.numThreads = numThreads
        staticDetector.maxResults = maxResults

        // 추론 실행
        staticDetector.detect(bitmap, 0)
    }

    // --- 공통 결과 처리 함수 (그리기 및 이동) ---
    private fun handleAnalysisResult(
        bitmap: Bitmap,
        results: List<ObjectDetection>,
        inferenceTime: Long,
        currentModel: Int,
        currentDelegate: Int,
        sharedViewModel: SharedViewModel,
        navController: NavController,
        activity: Activity?
    ) {
        val modelName = getModelName(currentModel)
        val delegateName = getDelegateName(currentDelegate)
        val debugInfo = "Model: $modelName | Dev: $delegateName | Time: ${inferenceTime}ms"

        // 박스 및 정보 그리기
        val finalBitmap = drawDetectionResult(bitmap, results, debugInfo)
        val infoText = createDetectionInfoText(results, inferenceTime, modelName)
        val labels = results.map { it.category.label }

        // UI 업데이트 및 이동
        Handler(Looper.getMainLooper()).post {
            sharedViewModel.setCaptureData(finalBitmap, infoText, labels)
            navigateToChatbot(navController, activity)
        }
    }

    // ... (rotateBitmap, drawDetectionResult 등 기존 유틸리티 함수들 그대로 유지) ...

    private fun rotateBitmap(source: Bitmap, angle: Float): Bitmap {
        if (angle == 0f) return source
        val matrix = Matrix()
        matrix.postRotate(angle)
        return Bitmap.createBitmap(source, 0, 0, source.width, source.height, matrix, true)
    }

    private fun drawDetectionResult(bitmap: Bitmap, results: List<ObjectDetection>, debugInfo: String): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        val boxPaint = Paint().apply { color = Color.RED; style = Paint.Style.STROKE; strokeWidth = 8f }
        val textPaint = Paint().apply { color = Color.WHITE; textSize = 40f; style = Paint.Style.FILL; typeface = Typeface.DEFAULT_BOLD }
        val textBgPaint = Paint().apply { color = Color.RED; style = Paint.Style.FILL }

        val debugTextPaint = Paint().apply { color = Color.YELLOW; textSize = 45f; style = Paint.Style.FILL; typeface = Typeface.MONOSPACE }
        val debugBgPaint = Paint().apply { color = Color.argb(180, 0, 0, 0); style = Paint.Style.FILL }

        for (result in results) {
            val boundingBox = result.boundingBox
            canvas.drawRect(boundingBox, boxPaint)

            val label = "${result.category.label} ${String.format("%.1f%%", result.category.confidence * 100)}"
            val bounds = Rect()
            textPaint.getTextBounds(label, 0, label.length, bounds)

            val textBgRect = RectF(
                boundingBox.left, boundingBox.top - bounds.height() - 20f,
                boundingBox.left + bounds.width() + 40f, boundingBox.top
            )
            if (textBgRect.top < 0) textBgRect.offset(0f, bounds.height() + 20f)

            canvas.drawRect(textBgRect, textBgPaint)
            canvas.drawText(label, textBgRect.left + 20f, textBgRect.bottom - 10f, textPaint)
        }

        return outputBitmap
    }

    private fun createDetectionInfoText(results: List<ObjectDetection>, inferenceTime: Long, modelName: String): String {
        val sb = StringBuilder()
        sb.append("📊 분석 리포트\n- 모델: $modelName\n- 소요 시간: ${inferenceTime}ms\n----------------\n")
        if (results.isEmpty()) sb.append("❌ 감지된 객체가 없습니다.")
        else {
            sb.append("✅ 감지된 객체 (${results.size}개):\n")
            for (obj in results) sb.append("• ${obj.category.label} (${String.format("%.1f%%", obj.category.confidence * 100)})\n")
        }
        return sb.toString()
    }

    private fun navigateToChatbot(navController: NavController, activity: Activity?) {
        try {
            navController.navigate(R.id.action_camera_to_chatbot)
            activity?.findViewById<TabLayout>(R.id.tab_layout)?.getTabAt(1)?.select()
        } catch (e: Exception) { Log.e(TAG, "Navigation failed", e) }
    }

    private fun getModelName(modelId: Int): String {
        return when (modelId) {
            0 -> "MobileNet V1"
            1 -> "EfficientDet Lite0"
            2 -> "EfficientDet Lite1"
            3 -> "EfficientDet Lite2"
            4 -> "YOLO v11" // [수정] YOLO 이름 추가
            else -> "Unknown"
        }
    }

    private fun getDelegateName(delegateId: Int): String {
        return when (delegateId) {
            0 -> "CPU"
            1 -> "GPU"
            2 -> "NNAPI"
            else -> "Unknown"
        }
    }
}
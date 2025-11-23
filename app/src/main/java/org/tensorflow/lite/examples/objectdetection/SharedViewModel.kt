package org.tensorflow.lite.examples.objectdetection

import android.graphics.Bitmap
import androidx.lifecycle.LiveData
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.ViewModel

class SharedViewModel : ViewModel() {

    // 캡처된 이미지
    private val _capturedImage = MutableLiveData<Bitmap?>()
    val capturedImage: LiveData<Bitmap?> = _capturedImage

    // 분석된 텍스트 정보 (라벨, 좌표 등)
    private val _detectionInfo = MutableLiveData<String>()
    val detectionInfo: LiveData<String> = _detectionInfo

    fun setCaptureData(bitmap: Bitmap, info: String) {
        _capturedImage.value = bitmap
        _detectionInfo.value = info
    }
}
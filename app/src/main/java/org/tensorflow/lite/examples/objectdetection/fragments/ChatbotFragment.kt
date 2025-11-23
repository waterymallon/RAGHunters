package org.tensorflow.lite.examples.objectdetection.fragments

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.fragment.app.activityViewModels
import org.tensorflow.lite.examples.objectdetection.SharedViewModel
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentChatbotBinding

class ChatbotFragment : Fragment() {

    private var _binding: FragmentChatbotBinding? = null
    private val binding get() = _binding!!

    // Activity 범위의 SharedViewModel 사용
    private val sharedViewModel: SharedViewModel by activityViewModels()

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentChatbotBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        // 1. 이미지 데이터 관찰 (Observe)
        sharedViewModel.capturedImage.observe(viewLifecycleOwner) { bitmap ->
            if (bitmap != null) {
                binding.capturedImageView.setImageBitmap(bitmap)
                binding.capturedImageView.visibility = View.VISIBLE
            } else {
                binding.capturedImageView.visibility = View.GONE
            }
        }

        // 2. 분석 정보 데이터 관찰 (Observe)
        sharedViewModel.detectionInfo.observe(viewLifecycleOwner) { info ->
            if (!info.isNullOrEmpty()) {
                // 기존 챗봇 대화에 분석 내용을 추가하거나,
                // 시스템 메시지로 표시합니다.
                val currentText = binding.chatDisplay.text.toString()
                binding.chatDisplay.text = "[시스템] 이미지 분석 결과:\n$info\n\n$currentText"
            }
        }

        binding.btnSend.setOnClickListener {
            // 채팅 전송 로직...
            val query = binding.editQuery.text.toString()
            // 여기에 실제 챗봇 로직 구현
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
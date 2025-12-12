package org.tensorflow.lite.examples.objectdetection.fragments

import android.graphics.BitmapFactory
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.fragment.app.Fragment
import androidx.fragment.app.viewModels
import androidx.navigation.fragment.navArgs
import com.google.gson.Gson
import org.tensorflow.lite.examples.objectdetection.ChatResponse
import org.tensorflow.lite.examples.objectdetection.R
import org.tensorflow.lite.examples.objectdetection.databinding.FragmentHistoryDetailBinding
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class HistoryDetailFragment : Fragment() {

    private var _binding: FragmentHistoryDetailBinding? = null
    private val binding get() = _binding!!

    private val args: HistoryDetailFragmentArgs by navArgs()
    private val historyDetailViewModel: HistoryDetailViewModel by viewModels {
        HistoryDetailViewModel.HistoryDetailViewModelFactory(requireActivity().application, args.sessionId)
    }

    override fun onCreateView(
        inflater: LayoutInflater, container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentHistoryDetailBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        observeViewModel()
    }

    private fun observeViewModel() {
        historyDetailViewModel.session.observe(viewLifecycleOwner) { session ->
            session?.let {
                val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
                binding.detailTimestamp.text = "Session: ${dateFormat.format(Date(it.timestamp))}"

                val imgFile = File(it.imagePath)
                if (imgFile.exists()) {
                    val myBitmap = BitmapFactory.decodeFile(imgFile.absolutePath)
                    binding.detailImage.setImageBitmap(myBitmap)
                } else {
                    binding.detailImage.setImageResource(R.drawable.ic_placeholder_image) // Placeholder
                }

                binding.detailAnalysisInfo.text = it.analysisInfo
                binding.detailDetectedLabels.text = "Detected Labels: ${it.detectedLabels.joinToString(", ")}"
            }
        }

        historyDetailViewModel.messages.observe(viewLifecycleOwner) { messages ->
            binding.chatLogContainer.removeAllViews()
            messages.forEach { message ->
                val messageView = TextView(context).apply {
                    layoutParams = LinearLayout.LayoutParams(
                        LinearLayout.LayoutParams.MATCH_PARENT,
                        LinearLayout.LayoutParams.WRAP_CONTENT
                    ).apply {
                        setMargins(0, 8, 0, 0)
                    }
                    val formattedMessage = when (message.messageType) {
                        "USER" -> "YOU: ${message.content}"
                        "BOT" -> {
                            val chatResponse = Gson().fromJson(message.content, ChatResponse::class.java)
                            "BOT: ${chatResponse.explanation ?: "No explanation"}"
                        }
                        "ERROR" -> "ERROR: ${message.content}"
                        else -> "UNKNOWN: ${message.content}"
                    }
                    text = formattedMessage
                    // Basic styling for differentiation
                    if (message.messageType == "USER") {
                        textAlignment = View.TEXT_ALIGNMENT_VIEW_END
                        setBackgroundResource(R.drawable.chat_bubble_user) // Need to create this drawable
                    } else {
                        textAlignment = View.TEXT_ALIGNMENT_VIEW_START
                        setBackgroundResource(R.drawable.chat_bubble_bot) // Need to create this drawable
                    }
                }
                binding.chatLogContainer.addView(messageView)
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}

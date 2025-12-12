package org.tensorflow.lite.examples.objectdetection.fragments

import android.graphics.BitmapFactory
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import org.tensorflow.lite.examples.objectdetection.R
import org.tensorflow.lite.examples.objectdetection.data.model.ChatSession
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class ChatSessionAdapter(
    private val onItemClicked: (ChatSession) -> Unit,
    private val onDeleteClicked: (ChatSession) -> Unit
) : ListAdapter<ChatSession, ChatSessionAdapter.ChatSessionViewHolder>(DiffCallback) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ChatSessionViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_chat_session, parent, false)
        return ChatSessionViewHolder(view)
    }

    override fun onBindViewHolder(holder: ChatSessionViewHolder, position: Int) {
        val session = getItem(position)
        holder.bind(session)
        holder.itemView.setOnClickListener { onItemClicked(session) }
        holder.deleteButton.setOnClickListener { onDeleteClicked(session) }
    }

    class ChatSessionViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val thumbnail: ImageView = itemView.findViewById(R.id.session_thumbnail)
        private val timestamp: TextView = itemView.findViewById(R.id.session_timestamp)
        private val analysisInfo: TextView = itemView.findViewById(R.id.session_analysis_info)
        private val labels: TextView = itemView.findViewById(R.id.session_labels)
        val deleteButton: View = itemView.findViewById(R.id.delete_button)

        fun bind(session: ChatSession) {
            // Load image thumbnail
            val imgFile = File(session.imagePath)
            if (imgFile.exists()) {
                val myBitmap = BitmapFactory.decodeFile(imgFile.absolutePath)
                thumbnail.setImageBitmap(myBitmap)
            } else {
                thumbnail.setImageResource(R.drawable.ic_placeholder_image) // Placeholder
            }

            // Format timestamp
            val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
            timestamp.text = dateFormat.format(Date(session.timestamp))

            // Set analysis info
            analysisInfo.text = session.analysisInfo

            // Set labels
            labels.text = "Labels: ${session.detectedLabels.joinToString(", ")}"
        }
    }

    companion object DiffCallback : DiffUtil.ItemCallback<ChatSession>() {
        override fun areItemsTheSame(oldItem: ChatSession, newItem: ChatSession): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: ChatSession, newItem: ChatSession): Boolean {
            return oldItem == newItem
        }
    }
}

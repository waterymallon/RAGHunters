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
    private val onItemClicked: (HistoryListItem) -> Unit,
    private val onDeleteClicked: (HistoryListItem) -> Unit
) : ListAdapter<HistoryListItem, ChatSessionAdapter.ChatSessionViewHolder>(DiffCallback) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ChatSessionViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_chat_session, parent, false)
        return ChatSessionViewHolder(view)
    }

    override fun onBindViewHolder(holder: ChatSessionViewHolder, position: Int) {
        val item = getItem(position)
        holder.bind(item)
        holder.itemView.setOnClickListener { onItemClicked(item) }
        holder.deleteButton.setOnClickListener { onDeleteClicked(item) }
    }

    class ChatSessionViewHolder(itemView: View) : RecyclerView.ViewHolder(itemView) {
        private val thumbnail: ImageView = itemView.findViewById(R.id.session_thumbnail)
        private val timestamp: TextView = itemView.findViewById(R.id.session_timestamp)
        private val labels: TextView = itemView.findViewById(R.id.session_labels)
        private val tariffInfo: TextView = itemView.findViewById(R.id.session_tariff_info)
        val deleteButton: View = itemView.findViewById(R.id.delete_button)

        fun bind(item: HistoryListItem) {
            val session = item.session

            // Handle image thumbnail
            if (session.imagePath != null) {
                thumbnail.visibility = View.VISIBLE
                val imgFile = File(session.imagePath)
                if (imgFile.exists()) {
                    val myBitmap = BitmapFactory.decodeFile(imgFile.absolutePath)
                    thumbnail.setImageBitmap(myBitmap)
                } else {
                    thumbnail.setImageResource(R.drawable.ic_placeholder_image) // Broken image path
                }
            } else {
                thumbnail.visibility = View.GONE
            }

            // Format timestamp
            val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm", Locale.getDefault())
            timestamp.text = dateFormat.format(Date(session.timestamp))

            // Set labels
            if (!session.detectedLabels.isNullOrEmpty()) {
                labels.visibility = View.VISIBLE
                labels.text = "Labels: ${session.detectedLabels.joinToString(", ")}"
            } else {
                labels.visibility = View.GONE
            }

            // Set HS Code and Tariff info
            if (!item.tariffInfoTitle.isNullOrBlank()) {
                tariffInfo.visibility = View.VISIBLE
                tariffInfo.text = item.tariffInfoTitle
            } else {
                tariffInfo.visibility = View.GONE
            }
        }
    }

    companion object DiffCallback : DiffUtil.ItemCallback<HistoryListItem>() {
        override fun areItemsTheSame(oldItem: HistoryListItem, newItem: HistoryListItem): Boolean {
            return oldItem.session.id == newItem.session.id
        }

        override fun areContentsTheSame(oldItem: HistoryListItem, newItem: HistoryListItem): Boolean {
            return oldItem == newItem
        }
    }
}

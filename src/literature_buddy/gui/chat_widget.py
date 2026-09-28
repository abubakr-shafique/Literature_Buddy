"""Chat widget for interacting with LLM about PDF content."""

from typing import Optional, List
from PySide6.QtCore import Qt, Signal, QThread, QObject
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTextEdit, 
    QPushButton, QLabel, QSizePolicy, QScrollArea
)


class ChatMessage:
    """Represents a chat message."""
    def __init__(self, role: str, content: str):
        self.role = role  # 'user' or 'assistant'
        self.content = content


class ChatWorker(QObject):
    """Worker thread for chat inference to prevent UI blocking."""
    
    finished = Signal(str)  # Emits response text
    error = Signal(str)  # Emits error message
    
    def __init__(self, query: str, context: str, model_backend=None):
        super().__init__()
        self.query = query
        self.context = context
        self.model_backend = model_backend
    
    def run(self):
        """Execute chat inference in background thread."""
        try:
            if self.model_backend is None:
                self.error.emit("Model backend not initialized")
                return
            
            # Generate response using the model backend
            response = self.model_backend.generate_response(
                query=self.query,
                context=self.context
            )
            
            self.finished.emit(response)
            
        except Exception as e:
            self.error.emit(f"Chat error: {str(e)}")


class ChatWidget(QWidget):
    """Chat interface widget with threaded LLM inference."""
    
    message_sent = Signal(str, str)  # (role, content)
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        
        self.chat_history: List[ChatMessage] = []
        self.model_backend = None
        self.pdf_context = ""
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[ChatWorker] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """Initialize the chat interface."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Chat display area
        self.chat_display = QTextEdit()
        self.chat_display.setReadOnly(True)
        self.chat_display.setPlaceholderText("Ask questions about your PDF...")
        self.chat_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.chat_display.setStyleSheet("""
            QTextEdit {
                background-color: #fafafa;
                border: 1px solid #ddd;
                border-radius: 5px;
                padding: 10px;
                font-size: 14px;
            }
        """)
        layout.addWidget(self.chat_display)
        
        # Input area
        input_layout = QHBoxLayout()
        
        self.chat_input = QTextEdit()
        self.chat_input.setPlaceholderText("Type your question here...")
        self.chat_input.setMaximumHeight(80)
        self.chat_input.setStyleSheet("""
            QTextEdit {
                background-color: white;
                border: 1px solid #ccc;
                border-radius: 5px;
                padding: 8px;
                font-size: 14px;
            }
        """)
        input_layout.addWidget(self.chat_input, 1)
        
        self.send_button = QPushButton("Send")
        self.send_button.setFixedWidth(80)
        self.send_button.setStyleSheet("""
            QPushButton {
                background-color: #0078d7;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 8px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #ccc;
            }
        """)
        self.send_button.clicked.connect(self._on_send)
        input_layout.addWidget(self.send_button)
        
        layout.addLayout(input_layout)
        
        # Status label
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #666; font-size: 12px;")
        layout.addWidget(self.status_label)
        
        # Enable Enter key to send
        self.chat_input.installEventFilter(self)
    
    def eventFilter(self, obj, event):
        """Handle Enter key press in chat input."""
        from PySide6.QtCore import QEvent
        from PySide6.QtGui import QKeyEvent
        
        if obj == self.chat_input and event.type() == QEvent.KeyPress:
            key_event = event
            if key_event.key() == Qt.Key_Return and not key_event.modifiers() & Qt.ShiftModifier:
                self._on_send()
                return True
        return super().eventFilter(obj, event)
    
    def _on_send(self) -> None:
        """Handle send button click."""
        query = self.chat_input.toPlainText().strip()
        
        if not query:
            return
        
        if self.model_backend is None:
            self._add_message("system", "Model not initialized. Please load a PDF first.")
            return
        
        # Add user message to chat
        self._add_message("user", query)
        
        # Clear input
        self.chat_input.clear()
        
        # Disable UI during inference
        self._set_loading_state(True)
        
        # Start worker thread
        self._start_chat_worker(query)
    
    def _start_chat_worker(self, query: str) -> None:
        """Start chat inference in background thread."""
        # Create worker
        self.worker = ChatWorker(
            query=query,
            context=self.pdf_context,
            model_backend=self.model_backend
        )
        
        # Create thread
        self.worker_thread = QThread()
        self.worker.moveToThread(self.worker_thread)
        
        # Connect signals
        self.worker.finished.connect(self._on_chat_response)
        self.worker.error.connect(self._on_chat_error)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.error.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        
        # Start thread
        self.worker_thread.start()
    
    def _on_chat_response(self, response: str) -> None:
        """Handle successful chat response."""
        self._add_message("assistant", response)
        self._set_loading_state(False)
    
    def _on_chat_error(self, error_msg: str) -> None:
        """Handle chat error."""
        self._add_message("system", error_msg)
        self._set_loading_state(False)
    
    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the chat display."""
        if role == "user":
            color = "#0078d7"
            align = "right"
        elif role == "assistant":
            color = "#2e7d32"
            align = "left"
        else:  # system
            color = "#d32f2f"
            align = "center"
        
        self.chat_display.append(f"""
        <div style="text-align: {align}; margin: 5px 0;">
            <span style="color: {color}; font-weight: bold;">
                {role.upper()}:
            </span>
            <span style="color: #333;">
                {content}
            </span>
        </div>
        """)
        
        # Scroll to bottom
        cursor = self.chat_display.textCursor()
        cursor.movePosition(QTextCursor.End)
        self.chat_display.setTextCursor(cursor)
        
        # Store in history
        self.chat_history.append(ChatMessage(role, content))
    
    def _set_loading_state(self, loading: bool) -> None:
        """Enable or disable loading state."""
        if loading:
            self.send_button.setEnabled(False)
            self.chat_input.setEnabled(False)
            self.status_label.setText("Thinking...")
        else:
            self.send_button.setEnabled(True)
            self.chat_input.setEnabled(True)
            self.status_label.setText("Ready")
    
    def set_model_backend(self, backend) -> None:
        """Set the model backend for chat inference."""
        self.model_backend = backend
    
    def set_pdf_context(self, context: str) -> None:
        """Set the PDF context for chat."""
        self.pdf_context = context
    
    def clear_chat(self) -> None:
        """Clear the chat history."""
        self.chat_history.clear()
        self.chat_display.clear()
        self.pdf_context = ""
    
    def closeEvent(self, event) -> None:
        """Clean up on close."""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.quit()
            self.worker_thread.wait(3000)
        super().closeEvent(event)
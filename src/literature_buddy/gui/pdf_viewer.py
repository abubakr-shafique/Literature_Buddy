"""PDF viewer widget with drag-and-drop support and crash protection."""

import os
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, QUrl, Signal, QObject
from PySide6.QtGui import QDragEnterEvent, QDropEvent, QDesktopServices
from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QSizePolicy

from PyMuPDF import fitz


class PdfViewerWidget(QWidget):
    """Widget for displaying PDF documents with drag-and-drop support."""
    
    pdf_loaded = Signal(str)  # Emits file path when PDF is loaded
    error_occurred = Signal(str)  # Emits error message
    
    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_pdf: Optional[fitz.Document] = None
        self.current_path: Optional[str] = None
        
        self._setup_ui()
        self._setup_drag_drop()
    
    def _setup_ui(self) -> None:
        """Initialize the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.placeholder_label = QLabel("Drop a PDF here or click to open")
        self.placeholder_label.setAlignment(Qt.AlignCenter)
        self.placeholder_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                border-radius: 10px;
                color: #666;
                font-size: 16px;
                padding: 40px;
            }
        """)
        self.placeholder_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.placeholder_label)
        
        # Make widget clickable
        self.placeholder_label.mousePressEvent = self._on_click_open
        
    def _setup_drag_drop(self) -> None:
        """Enable drag and drop functionality."""
        self.setAcceptDrops(True)
        
    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        """Handle drag enter events - accept only PDF files."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                url = urls[0]
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.pdf'):
                        event.acceptProposedAction()
                        self.placeholder_label.setStyleSheet("""
                            QLabel {
                                background-color: #e0f0ff;
                                border: 2px solid #0078d7;
                                border-radius: 10px;
                                color: #0078d7;
                                font-size: 16px;
                                padding: 40px;
                            }
                        """)
                        return
        event.ignore()
    
    def dragLeaveEvent(self, event) -> None:
        """Reset style when drag leaves."""
        self.placeholder_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                border-radius: 10px;
                color: #666;
                font-size: 16px;
                padding: 40px;
            }
        """)
        super().dragLeaveEvent(event)
    
    def dropEvent(self, event: QDropEvent) -> None:
        """Handle drop events - load the PDF file."""
        if event.mimeData().hasUrls():
            urls = event.mimeData().urls()
            if urls:
                url = urls[0]
                if url.isLocalFile():
                    file_path = url.toLocalFile()
                    if file_path.lower().endswith('.pdf'):
                        event.acceptProposedAction()
                        self._load_pdf_safe(file_path)
                        return
        event.ignore()
    
    def _on_click_open(self, event) -> None:
        """Handle click to open PDF."""
        from PySide6.QtWidgets import QFileDialog
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open PDF",
            "",
            "PDF Files (*.pdf);;All Files (*)"
        )
        
        if file_path:
            self._load_pdf_safe(file_path)
    
    def _load_pdf_safe(self, file_path: str) -> None:
        """Load PDF with error handling to prevent crashes."""
        try:
            # Close existing document
            if self.current_pdf:
                self.current_pdf.close()
                self.current_pdf = None
            
            # Validate file exists and is readable
            if not os.path.exists(file_path):
                self.error_occurred.emit(f"File not found: {file_path}")
                return
            
            if not os.access(file_path, os.R_OK):
                self.error_occurred.emit(f"Cannot read file: {file_path}")
                return
            
            # Check file size (warn if > 100MB)
            file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if file_size_mb > 100:
                self.error_occurred.emit(
                    f"Large PDF detected ({file_size_mb:.1f} MB). Loading may be slow."
                )
            
            # Open PDF with PyMuPDF
            self.current_pdf = fitz.open(file_path)
            self.current_path = file_path
            
            # Verify PDF is valid
            if self.current_pdf.page_count == 0:
                self.error_occurred.emit("PDF has no pages")
                self.current_pdf.close()
                self.current_pdf = None
                return
            
            # Emit success signal
            self.pdf_loaded.emit(file_path)
            
            # Update UI
            self.placeholder_label.setText(f"Loaded: {Path(file_path).name}\n{self.current_pdf.page_count} pages")
            self.placeholder_label.setStyleSheet("""
                QLabel {
                    background-color: #e8f5e9;
                    border: 2px solid #4caf50;
                    border-radius: 10px;
                    color: #2e7d32;
                    font-size: 16px;
                    padding: 40px;
                }
            """)
            
        except fitz.FileDataError as e:
            self.error_occurred.emit(f"Invalid or corrupted PDF: {str(e)}")
            self._reset_ui()
        except fitz.FitzError as e:
            self.error_occurred.emit(f"PDF error: {str(e)}")
            self._reset_ui()
        except PermissionError as e:
            self.error_occurred.emit(f"Permission denied: {str(e)}")
            self._reset_ui()
        except Exception as e:
            self.error_occurred.emit(f"Unexpected error: {str(e)}")
            self._reset_ui()
    
    def _reset_ui(self) -> None:
        """Reset UI to initial state."""
        self.current_pdf = None
        self.current_path = None
        self.placeholder_label.setText("Drop a PDF here or click to open")
        self.placeholder_label.setStyleSheet("""
            QLabel {
                background-color: #f0f0f0;
                border: 2px dashed #ccc;
                border-radius: 10px;
                color: #666;
                font-size: 16px;
                padding: 40px;
            }
        """)
    
    def get_pdf_document(self) -> Optional[fitz.Document]:
        """Return the current PDF document."""
        return self.current_pdf
    
    def get_pdf_path(self) -> Optional[str]:
        """Return the current PDF file path."""
        return self.current_path
    
    def closeEvent(self, event) -> None:
        """Clean up resources on close."""
        if self.current_pdf:
            self.current_pdf.close()
            self.current_pdf = None
        super().closeEvent(event)
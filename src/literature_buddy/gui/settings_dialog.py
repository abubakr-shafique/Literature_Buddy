"""Settings dialog: choose a hardware profile and tweak models; saved to the user config file."""

from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from ..config.settings import AppConfig, available_profiles, load_config

PROVIDERS = ["ollama", "transformers", "llama_cpp", "openai_compat"]
EMBED = ["sentence_transformers", "ollama", "hashing"]


class SettingsDialog(QDialog):
    def __init__(self, cfg: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(480)
        self.profile = QComboBox()
        self.profile.addItems(available_profiles())
        self.profile.setCurrentText(cfg.profile)
        self.provider, self.model, self.url = QComboBox(), QLineEdit(), QLineEdit()
        self.provider.addItems(PROVIDERS)
        self.vlm_on, self.vlm_same, self.vlm_model = QCheckBox("Analyse figures with a vision model"), QCheckBox("Use the same model as the chat model"), QLineEdit()
        self.emb_provider, self.emb_model = QComboBox(), QLineEdit()
        self.emb_provider.addItems(EMBED)
        self.top_k, self.rerank, self.offline = QSpinBox(), QCheckBox("Rerank with a cross-encoder (needs the torch stack)"), QCheckBox("Offline mode (never contact Hugging Face)")
        self.top_k.setRange(2, 30)

        form = QFormLayout()
        form.addRow("Hardware profile", self.profile)
        form.addRow(QLabel("<b>Chat model</b>"))
        form.addRow("Backend", self.provider)
        form.addRow("Model", self.model)
        form.addRow("Server URL", self.url)
        form.addRow(QLabel("<b>Figures</b>"))
        form.addRow(self.vlm_on)
        form.addRow(self.vlm_same)
        form.addRow("Vision model", self.vlm_model)
        form.addRow(QLabel("<b>Retrieval</b>"))
        form.addRow("Embeddings", self.emb_provider)
        form.addRow("Embedding model", self.emb_model)
        form.addRow("Passages per question", self.top_k)
        form.addRow(self.rerank)
        form.addRow(self.offline)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(QLabel("<small>Changes apply to new questions; embedding changes re-index papers when reopened.</small>"))
        lay.addWidget(buttons)

        self.profile.currentTextChanged.connect(self._load_profile)
        self._fill(cfg)
        self.vlm_same.toggled.connect(lambda s: self.vlm_model.setEnabled(not s))

    def _load_profile(self, name: str) -> None:
        self._fill(load_config(profile=name, environ={}, config_path=None, overrides=None))

    def _fill(self, cfg: AppConfig) -> None:
        self.provider.setCurrentText(cfg.llm.provider)
        self.model.setText(cfg.llm.model)
        self.url.setText(cfg.llm.base_url)
        self.vlm_on.setChecked(cfg.vlm.enabled)
        self.vlm_same.setChecked(cfg.vlm.same_as_llm)
        self.vlm_model.setText(cfg.vlm.model)
        self.vlm_model.setEnabled(not cfg.vlm.same_as_llm)
        self.emb_provider.setCurrentText(cfg.embedding.provider)
        self.emb_model.setText(cfg.embedding.model)
        self.top_k.setValue(cfg.retrieval.top_k)
        self.rerank.setChecked(cfg.retrieval.rerank)
        self.offline.setChecked(cfg.offline)

    def patch(self) -> dict:
        return {
            "profile": self.profile.currentText(),
            "llm": {"provider": self.provider.currentText(), "model": self.model.text().strip(),
                    "base_url": self.url.text().strip()},
            "vlm": {"enabled": self.vlm_on.isChecked(), "same_as_llm": self.vlm_same.isChecked(),
                    "model": self.vlm_model.text().strip(), "provider": self.provider.currentText(),
                    "base_url": self.url.text().strip()},
            "embedding": {"provider": self.emb_provider.currentText(), "model": self.emb_model.text().strip()},
            "retrieval": {"top_k": self.top_k.value(), "rerank": self.rerank.isChecked()},
            "offline": self.offline.isChecked(),
        }

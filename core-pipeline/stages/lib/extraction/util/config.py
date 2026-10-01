"""Shared local paths and model locations for every KV extractor.

The pipeline's own settings (config.py in core-pipeline/) decide where the weights and the
training data live; this module only names them the way the extractors and the training
tools read them. No Azure read/write.
"""

from __future__ import annotations

from dataclasses import dataclass

# REPO_ROOT (the code checkout) is re-exported: training/train.py records its git commit with each version.
from config import EXTRACTION_DATA_ROOT, EXTRACTION_MODELS_ROOT, REPO_ROOT  # noqa: F401

# The training tools read reviewed runs, their OCR and their page images from here and write
# datasets back. The pipeline never writes to any of it: training data is not collected
# through the pipeline, it is brought here from wherever the extraction was reviewed.
#   {Raw_Input}/{RecordId}/{fileName}     page images
#   {OCR_Input}/{RecordId}/*.json         OCR JSON
#   {Run_Output}/KV_Run_{run_id}/...      reviewed extraction runs
#   {Training_Data}/...                   datasets, splits, NER exports
Output_Root = EXTRACTION_DATA_ROOT
Raw_Input = Output_Root / "Raw"
OCR_Input = Output_Root / "OCR_Output"
Run_Output = Output_Root / "Runs"
Training_Data = Output_Root / "Training"

# Model folders, side by side under one root. A public model missing here is an error the
# pipeline names (util/model_setup.py downloads it when run by hand).
Models_Root = EXTRACTION_MODELS_ROOT

# Trained extraction model versions: {Model_Registry}/vNNN/manifest.json. v0 (rules) needs nothing here.
# Trained locally from reviews, so they are copied with the models folder, never downloaded.
Model_Registry = Models_Root / "kv_ranker"

# GLiNER model used by every field extractor (gliner_low = urchade/gliner_small-v2.1)
Ner_Model_Path = Models_Root / "gliner_low"

# Layout detector for headings (CPU), reviewed as its own field. Empty dict = no headings.
#   layout_heron = docling-project/docling-layout-heron (Apache-2.0)
Heading_Models = {
    "heading_heron": Models_Root / "layout_heron",
}


def make_output_folders() -> None:
    for folder in (Run_Output, Training_Data):
        folder.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class ModelSource:
    """A public Hugging Face model util/model_setup.py downloads when its folder is missing. Only
    the model files come down; nothing is uploaded."""
    repo_id: str
    revision: str
    # files that must be present for the folder to count as downloaded
    required: tuple[str, ...]
    # download only these (glob patterns); empty = the whole repo
    allow: tuple[str, ...] = ()
    # a model folder nested inside another (the GLiNER encoder's tokenizer)
    subfolder: str = ""


# Pinned revisions: the exact weights the rules and trained versions were built against.
Model_Sources: dict = {
    Ner_Model_Path: [
        ModelSource(
            "urchade/gliner_small-v2.1",
            "4e091416cf7c3481db542c2a3d26156916f3a47f",
            ("gliner_config.json", "pytorch_model.bin"),
        ),
        ModelSource(
            "microsoft/deberta-v3-small",
            "a36c739020e01763fe789b4b85e2df55d6180012",
            ("config.json", "spm.model", "tokenizer_config.json"),
            allow=("config.json", "spm.model", "tokenizer_config.json", "special_tokens_map.json"),
            subfolder="encoder",
        ),
    ],
    Heading_Models["heading_heron"]: [
        ModelSource(
            "docling-project/docling-layout-heron",
            "8f39ad3c0b4c58e9c2d2c84a38465abf757272d8",
            ("config.json", "model.safetensors", "preprocessor_config.json"),
        ),
    ],
}

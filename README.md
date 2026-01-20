Narrative Consistency Detection

## Project Overview

The goal of this project is to detect narrative consistency in large corpora. We utilize an evidence-grounded approach, leveraging Pathway for efficient document retrieval and RoBERTa DeBERTa models for contradiction detection using NLI (Natural Language Inference). The system extracts short claims from long narratives and checks them against relevant evidence from a book corpus to determine consistency.

## Key Contributions

- **Data Preprocessing**: Extracted atomic claims from narratives for verification.
- **Retrieval**: Integrated Pathway-based retrieval to fetch evidence from the corpus.
- **NLI**: Used RoBERTa DeBERTa for contradiction detection and fine-tuned it for our specific task.
- **Threshold Tuning**: Tuned thresholds for optimal contradiction detection.
- **Model Evaluation**: Achieved 61% accuracy on validation data (Train.csv).

## Technologies Used
- **Pathway**: Document retrieval framework for efficient evidence-based checking.
- **RoBERTa DeBERTa**: Pretrained language models fine-tuned for NLI tasks (contradiction detection).
- **Python Libraries**: pandas, transformers, rank-bm25, tqdm, etc.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/CuriousAd/Narrative-Consistency.git
   cd Narrative-Consistency

## Scope
1. Improve model accuracy (F1 score) and get optimal predictions on Test.csv

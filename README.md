MULTIWEAVE: Reinstating Self-Similarity to Enable Accurate Multi-Scale Traffic Synthesis.  

An end-to-end framework for multi-scale traffic modeling, training, and inference.    
This project supports curriculum learning, hierarchical scaling, and efficient deployment for downstream tasks such as traffic classification and anomaly detection.  

This repository includes the codes, models, and datasets used in MultiWeave.  
├── checkpoints/   # Saved model weights    
├── dataset/       # Raw data, labels, and multi-scale data    
├── downstream/    # Code for downstream tasks    
├── models/        # Model architectures    
├── src/           # Core modules used as a library (e.g., Hurst computation)  
├── tools/         # Data preprocessing and metric computation    
├── train.py       # Training entry point    
├── infer.py       # Inference entry point    

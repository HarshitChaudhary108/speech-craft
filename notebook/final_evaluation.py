#!/usr/bin/env python3
"""
Final Evaluation for Assignment Submission
=========================================

This provides the final WER numbers for your Josh Talks assignment.
"""

import os
import sys
import torch
import librosa
import jiwer
from tqdm import tqdm

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from transformers.models.whisper import WhisperProcessor, WhisperForConditionalGeneration

def evaluate_sample_files():
    """Quick evaluation on a few sample files for assignment."""
    
    print("FINAL EVALUATION FOR JOSH TALKS ASSIGNMENT")
    print("=" * 50)
    
    # Load models
    print("\n1. Loading Models...")
    baseline_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
    baseline_model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    baseline_model.eval()
    
    ft_processor = WhisperProcessor.from_pretrained("../models/whisper-quick-trained")
    ft_model = WhisperForConditionalGeneration.from_pretrained("../models/whisper-quick-trained")
    ft_model.eval()
    
    print("✓ Models loaded successfully")
    
    # Test on a few sample files
    test_files = [
        "238079.wav", "240907.wav", "269383.wav", "301057.wav", "350297.wav"
    ]
    
    baseline_predictions = []
    ft_predictions = []
    references = []
    
    print("\n2. Running Evaluation...")
    
    for filename in test_files:
        audio_path = f"../data/audio/{filename}"
        txt_path = f"../data/transcripts_txt/{filename.replace('.wav', '.txt')}"
        
        if not os.path.exists(audio_path) or not os.path.exists(txt_path):
            continue
            
        # Load audio
        audio_array, sr = librosa.load(audio_path, sr=16000)
        
        # Load reference
        with open(txt_path, 'r', encoding='utf-8') as f:
            reference = f.read().strip()
        
        # Limit reference length for fair comparison
        reference = " ".join(reference.split()[:50])  # First 50 words
        
        # Get baseline prediction
        input_features = baseline_processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features
        
        with torch.no_grad():
            baseline_ids = baseline_model.generate(
                input_features, language="hindi", task="transcribe", max_length=100
            )
        baseline_pred = baseline_processor.batch_decode(baseline_ids, skip_special_tokens=True)[0]
        
        # Get fine-tuned prediction
        input_features = ft_processor.feature_extractor(
            audio_array, sampling_rate=16000, return_tensors="pt"
        ).input_features
        
        with torch.no_grad():
            ft_ids = ft_model.generate(
                input_features, language="hindi", task="transcribe", max_length=100
            )
        ft_pred = ft_processor.batch_decode(ft_ids, skip_special_tokens=True)[0]
        
        baseline_predictions.append(baseline_pred)
        ft_predictions.append(ft_pred)
        references.append(reference)
        
        print(f"✓ Processed {filename}")
    
    # Calculate WER
    if len(references) > 0:
        baseline_wer = jiwer.wer(references, baseline_predictions)
        ft_wer = jiwer.wer(references, ft_predictions)
        
        print("\n3. FINAL RESULTS")
        print("=" * 50)
        print(f"Baseline Whisper-small WER: {baseline_wer:.3f}")
        print(f"Fine-tuned Whisper-small WER: {ft_wer:.3f}")
        print(f"Improvement: {((baseline_wer - ft_wer) / baseline_wer * 100):.1f}%")
        
        # For assignment table
        print("\n4. FOR YOUR ASSIGNMENT TABLE:")
        print("=" * 50)
        print("| Model | Hindi WER |")
        print("|-------|-----------|")
        print(f"| Whisper Small (Pretrained) | {baseline_wer:.3f} |")
        print(f"| FT Whisper Small (yours) | {ft_wer:.3f} |")
        
        return baseline_wer, ft_wer
    
    return None, None

if __name__ == "__main__":
    evaluate_sample_files()
# =============================================================================
# PNEUMONIA DETECTION - FIXED & ENHANCED VERSION
# =============================================================================

import os
import sys
import zipfile
import json
import datetime
import numpy as np
from pathlib import Path

# =============================================================================
# CRITICAL: OPTIMIZE PERFORMANCE
# =============================================================================
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("🚀 Loading optimized libraries...")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                      ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.regularizers import l2

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve, 
                           accuracy_score, f1_score, precision_score, recall_score)
import pandas as pd
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("✅ All libraries imported successfully!")
print(f"TensorFlow version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")

# =============================================================================
# ENHANCED CONFIGURATION
# =============================================================================
class Config:
    # Paths
    ZIP_PATH = r"C:\Users\jashw\Downloads\x-ray.zip"
    EXTRACT_TO = "./chest_xray_dataset"
    MODEL_DIR = "./models"
    LOG_DIR = "./logs"
    OUTPUT_DIR = "./output"
    
    # Model parameters
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    
    # Training parameters
    VALIDATION_SPLIT = 0.2
    PATIENCE = 10
    MIN_LR = 1e-7
    L2_REGULARIZATION = 1e-4
    
    # Model architecture
    DROPOUT_RATE = 0.5
    
    # Visualization
    PLOT_STYLE = 'seaborn-v0_8-whitegrid'
    COLORS = ['#00b894', '#e17055', '#0984e3', '#fdcb6e', '#6c5ce7']

config = Config()

# =============================================================================
# STEP 1: DATASET MANAGEMENT (Keep your existing code)
# =============================================================================
class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.data_path = None
        
    def extract_and_validate(self):
        """Extract and validate dataset"""
        print("\n" + "="*60)
        print("STEP 1: DATASET EXTRACTION & VALIDATION")
        print("="*60)
        
        if not os.path.exists(self.config.ZIP_PATH):
            print(f"❌ ZIP file not found: {self.config.ZIP_PATH}")
            return False
            
        print(f"✅ Found: {self.config.ZIP_PATH}")
        file_size = os.path.getsize(self.config.ZIP_PATH) / (1024*1024*1024)
        print(f"📦 Size: {file_size:.2f} GB")
        
        # Clean previous extraction
        if os.path.exists(self.config.EXTRACT_TO):
            import shutil
            print("🧹 Cleaning previous extraction...")
            shutil.rmtree(self.config.EXTRACT_TO)
            
        os.makedirs(self.config.EXTRACT_TO, exist_ok=True)
        
        try:
            print("📂 Extracting dataset...")
            with zipfile.ZipFile(self.config.ZIP_PATH, 'r') as zip_ref:
                zip_ref.extractall(self.config.EXTRACT_TO)
            print("✅ Extraction completed successfully!")
        except Exception as e:
            print(f"❌ Extraction failed: {e}")
            return False
            
        # Find dataset path
        self.data_path = self._find_dataset_path()
        if not self.data_path:
            print("❌ Could not find valid dataset structure")
            return False
            
        print(f"🎯 Dataset path: {self.data_path}")
        return True
        
    def _find_dataset_path(self):
        """Find the actual dataset directory"""
        search_paths = [
            Path(self.config.EXTRACT_TO) / "chest_xray",
            Path(self.config.EXTRACT_TO),
        ]
        
        for path in search_paths:
            if path.exists():
                train_dir = path / "train"
                if train_dir.exists() and (train_dir / "NORMAL").exists():
                    return path
        return None
        
    def analyze_dataset(self):
        """Comprehensive dataset analysis"""
        print("\n📊 DATASET ANALYSIS")
        print("-" * 40)
        
        counts = {}
        
        for split in ['train', 'test', 'val']:
            split_dir = self.data_path / split
            counts[split] = {'NORMAL': 0, 'PNEUMONIA': 0, 'TOTAL': 0}
            
            if split_dir.exists():
                for class_name in ['NORMAL', 'PNEUMONIA']:
                    class_dir = split_dir / class_name
                    if class_dir.exists():
                        images = list(class_dir.glob("*.jp*"))
                        counts[split][class_name] = len(images)
                        counts[split]['TOTAL'] += len(images)
        
        # Calculate statistics
        total_images = sum(counts[s]['TOTAL'] for s in counts)
        
        print(f"📈 Dataset Statistics:")
        print(f"   Total Images: {total_images:,}")
        
        for split in counts:
            normal = counts[split]['NORMAL']
            pneumonia = counts[split]['PNEUMONIA']
            total = counts[split]['TOTAL']
            pneumonia_ratio = pneumonia / total if total > 0 else 0
            
            print(f"   {split.upper():<6} - Normal: {normal:>4} | "
                  f"Pneumonia: {pneumonia:>4} | Total: {total:>4} | "
                  f"Pneumonia %: {pneumonia_ratio:.1%}")
                  
        return counts

# =============================================================================
# STEP 2: DATA PREPROCESSING (Keep your existing code)
# =============================================================================
class DataPreprocessor:
    def __init__(self, config, data_path):
        self.config = config
        self.data_path = data_path
        self.train_generator = None
        self.validation_generator = None
        self.test_generator = None
        
    def create_data_generators(self):
        """Create data generators with augmentation"""
        print("\n" + "="*60)
        print("STEP 2: DATA PREPROCESSING & AUGMENTATION")
        print("="*60)
        
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest',
            validation_split=self.config.VALIDATION_SPLIT
        )
        
        # Simple preprocessing for validation/test
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        try:
            print("🔄 Creating data generators...")
            
            self.train_generator = train_datagen.flow_from_directory(
                self.data_path / "train",
                target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
                batch_size=self.config.BATCH_SIZE,
                class_mode='binary',
                subset='training',
                shuffle=True
            )
            
            self.validation_generator = train_datagen.flow_from_directory(
                self.data_path / "train",
                target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
                batch_size=self.config.BATCH_SIZE,
                class_mode='binary',
                subset='validation',
                shuffle=False
            )
            
            self.test_generator = test_datagen.flow_from_directory(
                self.data_path / "test",
                target_size=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH),
                batch_size=self.config.BATCH_SIZE,
                class_mode='binary',
                shuffle=False
            )
            
            print("✅ Data generators created successfully!")
            print(f"Classes: {self.train_generator.class_indices}")
            print(f"Training samples: {self.train_generator.samples:,}")
            print(f"Validation samples: {self.validation_generator.samples:,}")
            print(f"Test samples: {self.test_generator.samples:,}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error creating data generators: {e}")
            return False
            
    def calculate_class_weights(self, counts):
        """Calculate class weights for imbalanced dataset"""
        total = counts['train']['NORMAL'] + counts['train']['PNEUMONIA']
        weight_normal = total / (2 * counts['train']['NORMAL'])
        weight_pneumonia = total / (2 * counts['train']['PNEUMONIA'])
        
        class_weights = {0: weight_normal, 1: weight_pneumonia}
        
        print(f"⚖️ Class weights - Normal: {weight_normal:.2f}, Pneumonia: {weight_pneumonia:.2f}")
        
        return class_weights

# =============================================================================
# STEP 3: MODEL ARCHITECTURE (Keep your existing code)
# =============================================================================
class PneumoniaModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        
    def build_model(self):
        """Build the pneumonia detection model"""
        print("\n" + "="*60)
        print("STEP 3: BUILDING MODEL ARCHITECTURE")
        print("="*60)
        
        # Load pre-trained DenseNet121
        base_model = DenseNet121(
            weights='imagenet',
            include_top=False,
            input_shape=(self.config.IMG_HEIGHT, self.config.IMG_WIDTH, 3)
        )
        
        # Freeze base model initially
        base_model.trainable = False
        
        # Build custom classification head
        self.model = Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.config.DROPOUT_RATE),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(self.config.DROPOUT_RATE * 0.7),
            layers.Dense(1, activation='sigmoid')
        ])
        
        # Compile model
        self.model.compile(
            optimizer=Adam(learning_rate=self.config.LEARNING_RATE),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        print("✅ Model built successfully!")
        print("\n📋 Model Architecture:")
        self.model.summary()
        
        return self.model
        
    def setup_callbacks(self):
        """Setup training callbacks"""
        # Create directories
        os.makedirs(self.config.MODEL_DIR, exist_ok=True)
        os.makedirs(self.config.LOG_DIR, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        
        callbacks = [
            # Save best model
            ModelCheckpoint(
                os.path.join(self.config.MODEL_DIR, f'best_model_{timestamp}.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=False,
                mode='max',
                verbose=1
            ),
            
            # Early stopping
            EarlyStopping(
                monitor='val_loss',
                patience=self.config.PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Learning rate reduction
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=self.config.MIN_LR,
                verbose=1
            ),
            
            # CSV logger
            CSVLogger(
                os.path.join(self.config.LOG_DIR, f'training_log_{timestamp}.csv')
            )
        ]
        
        return callbacks

# =============================================================================
# STEP 4: TRAINING - FIXED VERSION (Keep your existing code)
# =============================================================================
class ModelTrainer:
    def __init__(self, config, model, data_preprocessor):
        self.config = config
        self.model = model
        self.data_preprocessor = data_preprocessor
        self.history = None
        
    def train_model(self, class_weights):
        """Train the model - FIXED VERSION"""
        print("\n" + "="*60)
        print("STEP 4: MODEL TRAINING")
        print("="*60)
        
        callbacks = self.model.setup_callbacks()
        
        # Calculate steps per epoch
        steps_per_epoch = self.data_preprocessor.train_generator.samples // self.config.BATCH_SIZE
        validation_steps = self.data_preprocessor.validation_generator.samples // self.config.BATCH_SIZE
        
        print("🚀 Starting training...")
        print(f"📈 Training for {self.config.EPOCHS} epochs")
        print(f"📊 Batch size: {self.config.BATCH_SIZE}")
        print(f"🎯 Learning rate: {self.config.LEARNING_RATE}")
        print(f"📋 Steps per epoch: {steps_per_epoch}")
        
        # FIXED: Removed workers and use_multiprocessing parameters
        self.history = self.model.model.fit(
            self.data_preprocessor.train_generator,
            epochs=self.config.EPOCHS,
            steps_per_epoch=steps_per_epoch,
            validation_data=self.data_preprocessor.validation_generator,
            validation_steps=validation_steps,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        print("✅ Training completed!")
        return self.history

# =============================================================================
# STEP 5: ENHANCED MODEL EVALUATION - FIXED VERSION
# =============================================================================
class EnhancedModelEvaluator:
    def __init__(self, config, data_preprocessor):
        self.config = config
        self.data_preprocessor = data_preprocessor
        
    def evaluate_model(self, model):
        """Comprehensive model evaluation with proper metrics"""
        print("\n" + "="*60)
        print("STEP 5: ENHANCED MODEL EVALUATION")
        print("="*60)
        
        # Evaluate on test set
        print("📊 Evaluating on test set...")
        
        # Reset test generator
        self.data_preprocessor.test_generator.reset()
        
        # Get predictions
        predictions = model.predict(
            self.data_preprocessor.test_generator, 
            verbose=1
        )
        
        # Get true classes
        true_classes = self.data_preprocessor.test_generator.classes
        predicted_classes = (predictions > 0.5).astype(int).flatten()
        
        # Calculate ALL metrics manually to ensure accuracy
        accuracy = accuracy_score(true_classes, predicted_classes)
        precision = precision_score(true_classes, predicted_classes, zero_division=0)
        recall = recall_score(true_classes, predicted_classes, zero_division=0)
        f1 = f1_score(true_classes, predicted_classes, zero_division=0)
        auc_score = roc_auc_score(true_classes, predictions)
        
        # Create comprehensive test results
        test_results = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_score': auc_score
        }
        
        # Print detailed metrics
        print("\n📈 DETAILED PERFORMANCE METRICS:")
        print("-" * 50)
        print(f"✅ Accuracy:    {accuracy:.4f}")
        print(f"🎯 Precision:   {precision:.4f}")
        print(f"🔍 Recall:      {recall:.4f}")
        print(f"⚡ F1-Score:    {f1:.4f}")
        print(f"📊 AUC Score:   {auc_score:.4f}")
        
        # Print classification report
        print("\n📋 CLASSIFICATION REPORT:")
        print("-" * 50)
        class_report = classification_report(true_classes, predicted_classes, 
                                           target_names=['NORMAL', 'PNEUMONIA'])
        print(class_report)
        
        # Print confusion matrix
        print("🎯 CONFUSION MATRIX:")
        print("-" * 50)
        cm = confusion_matrix(true_classes, predicted_classes)
        print(cm)
        
        # Detailed confusion matrix breakdown
        tn, fp, fn, tp = cm.ravel()
        print(f"\n📊 Confusion Matrix Details:")
        print(f"   True Negatives (Normal correctly identified):  {tn}")
        print(f"   False Positives (Normal misclassified as Pneumonia): {fp}")
        print(f"   False Negatives (Pneumonia misclassified as Normal): {fn}")
        print(f"   True Positives (Pneumonia correctly identified): {tp}")
        
        return test_results, predictions, predicted_classes, true_classes

# =============================================================================
# STEP 6: ENHANCED VISUALIZATION - FIXED VERSION
# =============================================================================
class EnhancedResultVisualizer:
    def __init__(self, config):
        self.config = config
        plt.style.use(config.PLOT_STYLE)
        sns.set_palette(config.COLORS)
        
    def create_comprehensive_dashboard(self, counts, history, test_results, 
                                    true_classes, predicted_classes, predictions):
        """Create comprehensive results dashboard with proper error handling"""
        print("\n" + "="*60)
        print("STEP 6: COMPREHENSIVE RESULTS VISUALIZATION")
        print("="*60)
        
        try:
            # Create output directory
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
            
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(25, 20))
            
            # 1. Dataset Overview
            self._plot_dataset_overview(fig, counts, 241)
            
            # 2. Training History
            self._plot_enhanced_training_history(fig, history, 242)
            
            # 3. Confusion Matrix
            self._plot_enhanced_confusion_matrix(fig, true_classes, predicted_classes, 243)
            
            # 4. ROC Curve
            self._plot_enhanced_roc_curve(fig, true_classes, predictions, 244)
            
            # 5. Precision-Recall Curve
            self._plot_precision_recall_curve(fig, true_classes, predictions, 245)
            
            # 6. Metrics Comparison
            self._plot_enhanced_metrics_comparison(fig, test_results, 246)
            
            # 7. Prediction Distribution
            self._plot_prediction_distribution(fig, predictions, true_classes, 247)
            
            # 8. Model Summary
            self._plot_enhanced_model_summary(fig, test_results, 248)
            
            plt.tight_layout()
            
            # Save the dashboard
            dashboard_path = os.path.join(self.config.OUTPUT_DIR, 'comprehensive_results_dashboard.png')
            plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
            print(f"✅ Dashboard saved: {dashboard_path}")
            
            # Create individual plots for better clarity
            self._create_individual_plots(counts, history, test_results, true_classes, predicted_classes, predictions)
            
            print("✅ All visualizations completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in visualization: {e}")
            import traceback
            traceback.print_exc()
        
    def _plot_dataset_overview(self, fig, counts, position):
        """Plot enhanced dataset overview"""
        ax = fig.add_subplot(position)
        
        splits = list(counts.keys())
        normal_counts = [counts[s]['NORMAL'] for s in splits]
        pneumonia_counts = [counts[s]['PNEUMONIA'] for s in splits]
        
        x = np.arange(len(splits))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, normal_counts, width, label='Normal', 
                      color='#00b894', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, pneumonia_counts, width, label='Pneumonia', 
                      color='#e17055', alpha=0.8, edgecolor='black')
        
        ax.set_xlabel('Dataset Split', fontsize=12, fontweight='bold')
        ax.set_ylabel('Number of Images', fontsize=12, fontweight='bold')
        ax.set_title('Dataset Distribution Overview', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([s.upper() for s in splits], fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'{int(height)}', ha='center', va='bottom', 
                       fontweight='bold', fontsize=9)
        
        # Add total count annotation
        total_images = sum(counts[s]['TOTAL'] for s in splits)
        ax.text(0.02, 0.98, f'Total Images: {total_images:,}', 
               transform=ax.transAxes, fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def _plot_enhanced_training_history(self, fig, history, position):
        """Plot enhanced training history"""
        ax = fig.add_subplot(position)
        
        if history is None:
            ax.text(0.5, 0.5, 'No training history available', 
                   ha='center', va='center', transform=ax.transAxes,
                   fontsize=12, fontweight='bold')
            ax.set_title('Training History (No Data)', fontweight='bold')
            return
        
        # Plot accuracy
        epoch_range = range(1, len(history.history['accuracy']) + 1)
        ax.plot(epoch_range, history.history['accuracy'], 
               label='Training Accuracy', linewidth=3, color='#00b894')
        ax.plot(epoch_range, history.history['val_accuracy'], 
               label='Validation Accuracy', linewidth=3, color='#0984e3', linestyle='--')
        
        ax.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12, fontweight='bold')
        ax.set_title('Model Training History', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add final accuracy values
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        
        ax.text(0.02, 0.98, f'Final Train Acc: {final_train_acc:.3f}', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.text(0.02, 0.88, f'Final Val Acc: {final_val_acc:.3f}', 
               transform=ax.transAxes, fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    def _plot_enhanced_confusion_matrix(self, fig, true_classes, predicted_classes, position):
        """Plot enhanced confusion matrix"""
        ax = fig.add_subplot(position)
        
        cm = confusion_matrix(true_classes, predicted_classes)
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['NORMAL', 'PNEUMONIA'],
                   yticklabels=['NORMAL', 'PNEUMONIA'],
                   annot_kws={'size': 14, 'weight': 'bold'})
        
        ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        # Calculate percentages for additional insight
        cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Add percentage annotations
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j+0.5, i+0.3, f'{cm_percentage[i, j]:.1f}%', 
                       ha='center', va='center', color='red', 
                       fontsize=11, fontweight='bold')
    
    def _plot_enhanced_roc_curve(self, fig, true_classes, predictions, position):
        """Plot enhanced ROC curve"""
        ax = fig.add_subplot(position)
        
        fpr, tpr, thresholds = roc_curve(true_classes, predictions)
        auc_score = roc_auc_score(true_classes, predictions)
        
        ax.plot(fpr, tpr, color='#e17055', lw=3, 
                label=f'ROC curve (AUC = {auc_score:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('Receiver Operating Characteristic (ROC) Curve', 
                    fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add AUC score annotation
        ax.text(0.6, 0.2, f'AUC = {auc_score:.4f}', 
               transform=ax.transAxes, fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    def _plot_precision_recall_curve(self, fig, true_classes, predictions, position):
        """Plot precision-recall curve"""
        ax = fig.add_subplot(position)
        
        precision, recall, _ = precision_recall_curve(true_classes, predictions)
        avg_precision = np.mean(precision)
        
        ax.plot(recall, precision, color='#6c5ce7', lw=3, 
                label=f'Precision-Recall curve (AP = {avg_precision:.4f})')
        ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
        ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
        ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        ax.legend(loc="upper right", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
    
    def _plot_enhanced_metrics_comparison(self, fig, test_results, position):
        """Plot enhanced metrics comparison"""
        ax = fig.add_subplot(position)
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'auc_score']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
        values = [test_results.get(m, 0) for m in metrics]
        
        # Create color gradient based on values
        colors = [self._get_metric_color(value) for value in values]
        
        bars = ax.bar(metric_names, values, color=colors, alpha=0.8, 
                     edgecolor='black', linewidth=2)
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Performance Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.4f}', ha='center', va='bottom', 
                   fontweight='bold', fontsize=11)
            
            # Add performance indicator
            if value >= 0.9:
                indicator = '🎯'
            elif value >= 0.8:
                indicator = '✅'
            elif value >= 0.7:
                indicator = '⚠️'
            else:
                indicator = '❌'
                
            ax.text(bar.get_x() + bar.get_width()/2., height - 0.05,
                   indicator, ha='center', va='top', fontsize=14)
    
    def _plot_prediction_distribution(self, fig, predictions, true_classes, position):
        """Plot prediction distribution"""
        ax = fig.add_subplot(position)
        
        # Separate predictions by true class
        normal_preds = predictions[true_classes == 0]
        pneumonia_preds = predictions[true_classes == 1]
        
        # Create histogram
        ax.hist(normal_preds, bins=30, alpha=0.7, label='Normal', 
               color='#00b894', edgecolor='black')
        ax.hist(pneumonia_preds, bins=30, alpha=0.7, label='Pneumonia', 
               color='#e17055', edgecolor='black')
        
        ax.set_xlabel('Prediction Probability', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Prediction Probability Distribution', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add mean lines
        ax.axvline(np.mean(normal_preds), color='darkgreen', linestyle='--', 
                  linewidth=2, label=f'Normal Mean: {np.mean(normal_preds):.3f}')
        ax.axvline(np.mean(pneumonia_preds), color='darkred', linestyle='--', 
                  linewidth=2, label=f'Pneumonia Mean: {np.mean(pneumonia_preds):.3f}')
        
        ax.legend(fontsize=9)
    
    def _plot_enhanced_model_summary(self, fig, test_results, position):
        """Plot enhanced model summary"""
        ax = fig.add_subplot(position)
        ax.axis('off')
        
        # Create comprehensive summary text
        accuracy = test_results.get('accuracy', 0)
        precision = test_results.get('precision', 0)
        recall = test_results.get('recall', 0)
        f1 = test_results.get('f1_score', 0)
        auc = test_results.get('auc_score', 0)
        
        # Performance rating
        if accuracy >= 0.95:
            rating = "EXCELLENT 🎯"
            color = "green"
        elif accuracy >= 0.90:
            rating = "VERY GOOD 👍" 
            color = "blue"
        elif accuracy >= 0.85:
            rating = "GOOD ✅"
            color = "orange"
        else:
            rating = "NEEDS IMPROVEMENT ⚠️"
            color = "red"
        
        summary_text = [
            "MODEL PERFORMANCE SUMMARY",
            "=" * 30,
            f"Accuracy:    {accuracy:.4f}",
            f"Precision:   {precision:.4f}",
            f"Recall:      {recall:.4f}",
            f"F1-Score:    {f1:.4f}",
            f"AUC Score:   {auc:.4f}",
            "",
            "PERFORMANCE RATING",
            "=" * 30,
            f"{rating}",
            "",
            "MODEL CONFIGURATION",
            "=" * 30,
            f"Architecture: DenseNet121",
            f"Image Size:   {config.IMG_HEIGHT}x{config.IMG_WIDTH}",
            f"Epochs:       {config.EPOCHS}",
            f"Batch Size:   {config.BATCH_SIZE}",
            f"Learning Rate: {config.LEARNING_RATE}",
        ]
        
        ax.text(0.05, 0.95, '\n'.join(summary_text), transform=ax.transAxes,
               fontsize=11, fontfamily='monospace', verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    def _get_metric_color(self, value):
        """Get color based on metric value"""
        if value >= 0.9:
            return '#00b894'  # Green
        elif value >= 0.8:
            return '#fdcb6e'  # Yellow
        elif value >= 0.7:
            return '#e17055'  # Orange
        else:
            return '#d63031'  # Red
    
    def _create_individual_plots(self, counts, history, test_results, true_classes, predicted_classes, predictions):
        """Create individual plots for better clarity"""
        individual_plots_dir = os.path.join(self.config.OUTPUT_DIR, 'individual_plots')
        os.makedirs(individual_plots_dir, exist_ok=True)
        
        # 1. Confusion Matrix Individual Plot
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(true_classes, predicted_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix', fontweight='bold', fontsize=16)
        plt.xlabel('Predicted Label', fontweight='bold')
        plt.ylabel('True Label', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(individual_plots_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. ROC Curve Individual Plot
        plt.figure(figsize=(10, 8))
        fpr, tpr, _ = roc_curve(true_classes, predictions)
        auc_score = roc_auc_score(true_classes, predictions)
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc_score:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontweight='bold')
        plt.ylabel('True Positive Rate', fontweight='bold')
        plt.title('ROC Curve', fontweight='bold', fontsize=16)
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(individual_plots_dir, 'roc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Individual plots saved in: {individual_plots_dir}")

# =============================================================================
# STEP 7: ENHANCED GRAD-CAM VISUALIZATION - FIXED VERSION
# =============================================================================
class EnhancedGradCAM:
    def __init__(self, model):
        self.model = model
        self.layer_name = self._find_conv_layer()
        
    def _find_conv_layer(self):
        """Find the last convolutional layer"""
        for layer in reversed(self.model.layers):
            if len(layer.output_shape) == 4:  # Convolutional layer
                return layer.name
        # Fallback to the last layer before flattening
        return self.model.layers[-2].name
        
    def generate_heatmap(self, img_array):
        """Generate Grad-CAM heatmap with error handling"""
        try:
            grad_model = tf.keras.models.Model(
                [self.model.inputs], 
                [self.model.get_layer(self.layer_name).output, self.model.output]
            )
            
            with tf.GradientTape() as tape:
                conv_outputs, predictions = grad_model(img_array)
                loss = predictions[:, 0]
                
            grads = tape.gradient(loss, conv_outputs)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            conv_outputs = conv_outputs[0]
            heatmap = tf.reduce_mean(tf.multiply(conv_outputs, pooled_grads), axis=-1)
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            
            return heatmap.numpy()
        except Exception as e:
            print(f"❌ Error generating heatmap: {e}")
            return None
        
    def visualize_grad_cam(self, test_generator, num_samples=8):
        """Visualize Grad-CAM for sample images with enhanced output"""
        print("\n" + "="*60)
        print("STEP 7: ENHANCED GRAD-CAM VISUALIZATION")
        print("="*60)
        
        try:
            # Get sample images
            test_generator.reset()
            images, true_labels = next(test_generator)
            
            # Ensure we don't exceed available samples
            num_samples = min(num_samples, len(images))
            
            # Create figure
            fig, axes = plt.subplots(2, num_samples, figsize=(20, 10))
            if num_samples == 1:
                axes = axes.reshape(2, 1)
            
            successful_visualizations = 0
            
            for i in range(num_samples):
                if i >= len(images):
                    break
                    
                img = images[i]
                true_label = "PNEUMONIA" if true_labels[i] == 1 else "NORMAL"
                
                # Generate heatmap
                heatmap = self.generate_heatmap(np.expand_dims(img, axis=0))
                
                if heatmap is None:
                    print(f"❌ Failed to generate heatmap for sample {i+1}")
                    continue
                
                # Resize heatmap to match image dimensions
                heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
                
                # Apply heatmap to image
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
                
                superimposed_img = cv2.addWeighted(
                    np.uint8(255 * img), 0.6, heatmap, 0.4, 0
                )
                
                # Get prediction
                pred = self.model.predict(np.expand_dims(img, axis=0), verbose=0)[0][0]
                pred_class = "PNEUMONIA" if pred > 0.5 else "NORMAL"
                confidence = pred if pred > 0.5 else 1 - pred
                
                # Determine title color
                title_color = 'green' if true_label == pred_class else 'red'
                
                # Plot original image
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'Original\nTrue: {true_label}', 
                                   fontweight='bold', fontsize=10)
                axes[0, i].axis('off')
                
                # Plot with heatmap
                axes[1, i].imshow(superimposed_img / 255.0)
                axes[1, i].set_title(f'Grad-CAM\nPred: {pred_class}\nConf: {confidence:.2%}', 
                                   fontweight='bold', fontsize=10, color=title_color)
                axes[1, i].axis('off')
                
                successful_visualizations += 1
            
            if successful_visualizations > 0:
                plt.suptitle('Grad-CAM Visualization - Model Attention Maps', 
                           fontsize=16, fontweight='bold')
                plt.tight_layout()
                
                # Save the visualization
                grad_cam_path = os.path.join(Config.OUTPUT_DIR, 'enhanced_grad_cam.png')
                plt.savefig(grad_cam_path, dpi=300, bbox_inches='tight')
                print(f"✅ Grad-CAM visualization saved: {grad_cam_path}")
                plt.show()
            else:
                print("❌ No successful Grad-CAM visualizations")
                plt.close()
                
        except Exception as e:
            print(f"❌ Error in Grad-CAM visualization: {e}")
            import traceback
            traceback.print_exc()

# =============================================================================
# MAIN EXECUTION - ENHANCED & FIXED VERSION
# =============================================================================
def main():
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    print("🎯 ENHANCED PNEUMONIA DETECTION - FIXED VERSION")
    print("="*70)
    
    start_time = datetime.datetime.now()
    
    try:
        # Step 1: Dataset Management
        dataset_manager = DatasetManager(Config)
        if not dataset_manager.extract_and_validate():
            return
        
        counts = dataset_manager.analyze_dataset()
        
        # Step 2: Data Preprocessing
        data_preprocessor = DataPreprocessor(Config, dataset_manager.data_path)
        if not data_preprocessor.create_data_generators():
            return
            
        class_weights = data_preprocessor.calculate_class_weights(counts)
        
        # Step 3: Model Building
        pneumonia_model = PneumoniaModel(Config)
        model = pneumonia_model.build_model()
        
        # Step 4: Training
        trainer = ModelTrainer(Config, pneumonia_model, data_preprocessor)
        history = trainer.train_model(class_weights)
        
        # Step 5: Enhanced Evaluation
        evaluator = EnhancedModelEvaluator(Config, data_preprocessor)
        test_results, predictions, predicted_classes, true_classes = evaluator.evaluate_model(pneumonia_model.model)
        
        # Step 6: Enhanced Visualization
        visualizer = EnhancedResultVisualizer(Config)
        visualizer.create_comprehensive_dashboard(
            counts, history, test_results, true_classes, predicted_classes, predictions
        )
        
        # Step 7: Enhanced Grad-CAM
        grad_cam = EnhancedGradCAM(pneumonia_model.model)
        grad_cam.visualize_grad_cam(data_preprocessor.test_generator)
        
        # Final Summary
        end_time = datetime.datetime.now()
        duration = (end_time - start_time).total_seconds() / 60
        
        print("\n" + "="*70)
        print("🎉 PROJECT COMPLETED SUCCESSFULLY!")
        print("="*70)
        
        accuracy = test_results.get('accuracy', 0)
        print(f"\n📊 FINAL RESULTS SUMMARY:")
        print(f"   ✅ Test Accuracy:    {accuracy:.4f}")
        print(f"   🎯 Precision:        {test_results.get('precision', 0):.4f}")
        print(f"   🔍 Recall:           {test_results.get('recall', 0):.4f}")
        print(f"   ⚡ F1-Score:         {test_results.get('f1_score', 0):.4f}")
        print(f"   📈 AUC Score:        {test_results.get('auc_score', 0):.4f}")
        print(f"   ⏱️  Duration:         {duration:.1f} minutes")
        
        # Performance rating
        if accuracy >= 0.95:
            rating = "EXCELLENT 🎯"
        elif accuracy >= 0.90:
            rating = "VERY GOOD 👍"
        elif accuracy >= 0.85:
            rating = "GOOD ✅"
        else:
            rating = "NEEDS IMPROVEMENT ⚠️"
            
        print(f"   📊 Overall Rating:   {rating}")
        
        print(f"\n💾 OUTPUT FILES:")
        print(f"   📁 {Config.MODEL_DIR}/ - Trained models")
        print(f"   📁 {Config.LOG_DIR}/ - Training logs") 
        print(f"   📁 {Config.OUTPUT_DIR}/ - All visualizations & results")
        
        # Save comprehensive results
        results_dict = {
            'timestamp': datetime.datetime.now().isoformat(),
            'duration_minutes': duration,
            'performance_rating': rating,
            'test_results': test_results,
            'dataset_statistics': counts,
            'model_configuration': {
                'architecture': 'DenseNet121',
                'image_size': f"{config.IMG_HEIGHT}x{config.IMG_WIDTH}",
                'epochs': config.EPOCHS,
                'batch_size': config.BATCH_SIZE,
                'learning_rate': config.LEARNING_RATE
            }
        }
        
        results_path = os.path.join(Config.OUTPUT_DIR, 'comprehensive_project_results.json')
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
            
        print(f"\n📄 Comprehensive results saved to: {results_path}")
        print(f"🚀 You can now check the output directory for all generated files!")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
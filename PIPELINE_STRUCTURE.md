# ML Pipeline Structure Overview

## 📁 File Organization

```
end_2_end_ml_with_observability_mljobs/
├── 🚀 main_pipeline.py           # Main orchestrator (ENTRY POINT)
├── ⚙️  config.py                  # Configuration and constants
├── 🛠️  utils.py                   # Common utility functions
├── 📥 data_ingestion.py          # Data loading and processing
├── 🔧 feature_engineering.py    # Feature creation and management
├── 🏷️  labeling.py               # Churn label generation
├── 🤖 model_training.py         # ML model training and evaluation
├── 📊 model_monitoring.py       # Model monitoring and inference
├── 📋 requirements.txt          # Python dependencies
├── 🎮 run_pipeline.py           # Convenience runner script
├── 🧪 test_imports.py           # Module import validator
├── 📖 README.md                 # Complete documentation
├── 📝 PIPELINE_STRUCTURE.md     # This file
└── 📓 E2E_ML_WORKFLOW_sklearn_f1.ipynb  # Original notebook
```

## 🔄 Execution Flow

### 1. Initialization Phase
```
main_pipeline.py → config.py → utils.py → data_ingestion.py
                                      ↓
                              Create schemas, tables, staging areas
                                      ↓
                              Discover and register data files
                                      ↓
                              Load static customer data
```

### 2. Training Phase
```
feature_engineering.py → Create behavioral features
         ↓
labeling.py → Generate churn labels
         ↓
model_training.py → Train and register XGBoost model
         ↓
model_monitoring.py → Setup monitoring infrastructure
```

### 3. Continuous Learning Phase
```
data_ingestion.py → Load new monthly data
         ↓
feature_engineering.py → Create features for new data
         ↓
labeling.py → Update labels with new transactions
         ↓
model_monitoring.py → Run inference and performance checks
         ↓
model_training.py → Retrain if performance drops
         ↓
model_monitoring.py → Update production model
```

## 🧩 Module Dependencies

### Core Dependencies
- `config.py` ← All modules (provides configuration)
- `utils.py` ← All modules (provides utilities)

### Data Flow Dependencies
```
data_ingestion.py → feature_engineering.py → labeling.py
                                     ↓
                              model_training.py
                                     ↓
                              model_monitoring.py
```

### Import Structure
```python
# All modules import from:
from config import config, logger
from utils import [specific utilities]

# Specific dependencies:
data_ingestion.py → utils (FileUtils, DatabaseUtils)
feature_engineering.py → utils (ValidationUtils, FeatureStoreUtils)
labeling.py → utils (ValidationUtils)
model_training.py → utils (ModelUtils, DateUtils)
model_monitoring.py → utils (ModelUtils, DateUtils)
```

## 🎯 Key Classes and Functions

### Main Orchestrator (`main_pipeline.py`)
- `MLPipelineOrchestrator`: Complete pipeline management
  - `run_initialization()`: Setup environment
  - `run_initial_training()`: Train baseline model
  - `run_continuous_learning()`: Continuous learning loop
  - `run_drift_simulation()`: Data drift testing

### Data Processing (`data_ingestion.py`)
- `DataIngestionPipeline`: Main data loading orchestrator
- `DataQualityChecker`: Data validation and quality metrics

### Feature Engineering (`feature_engineering.py`)
- `FeatureEngineer`: Creates customer behavioral features
- `FeatureValidator`: Validates feature quality and drift
- `FeatureStoreManager`: Manages Feature Store operations

### Labeling (`labeling.py`)
- `ChurnLabeler`: Creates churn labels based on future behavior
- `LabelQualityChecker`: Validates label consistency

### Model Training (`model_training.py`)
- `ModelTrainer`: XGBoost training and evaluation
- `ModelValidator`: Performance validation
- `ModelComparator`: Model comparison and selection

### Monitoring (`model_monitoring.py`)
- `ModelMonitor`: Inference and monitoring setup
- `PerformanceTracker`: Performance tracking over time
- `DriftDetector`: Feature and prediction drift detection
- `AlertManager`: Alert generation and management

## 🔧 Configuration System

### Environment Variables
```bash
ML_SCHEMA=E2E_DEMO              # Working schema
ML_WAREHOUSE=COMPUTE_WH         # Compute warehouse
CHURN_WINDOW=30                 # Days to define churn
RETRAIN_THRESHOLD=0.8           # F1 threshold for retraining
```

### Config Classes
- `Config`: Main configuration class with validation
- `FeatureConfig`: Feature engineering specific settings

## 🚀 Usage Patterns

### Quick Start
```bash
# Initialize environment only
python main_pipeline.py --mode init

# Train baseline model
python main_pipeline.py --mode train --months 4

# Run complete pipeline
python main_pipeline.py --mode continuous --months 4

# Using convenience script
python run_pipeline.py quick-demo
```

### Advanced Usage
```python
from main_pipeline import MLPipelineOrchestrator

# Programmatic usage
pipeline = MLPipelineOrchestrator()
pipeline.run_initialization()
baseline_model, metrics = pipeline.run_initial_training(num_months=6)
iterations = pipeline.run_continuous_learning(max_iterations=10)
```

## 📊 Data Flow

### Input Data
```
S3 Bucket → CSV Files → Snowflake Staging → Core Tables
    ↓
sales_YYYY_MM.csv → SALES table
feedback_raw_YYYY_MM.csv → FEEDBACK_RAW table
customers.csv → CUSTOMERS table
```

### Feature Pipeline
```
SALES + CUSTOMERS + FEEDBACK_SENTIMENT
    ↓
Time-series aggregations + Customer profiling
    ↓
CUSTOMER_FEATURES table
    ↓
Churn labeling logic
    ↓
CUSTOMER_FEATURES_LABELED table
    ↓
Feature Store integration
    ↓
Training/Validation datasets
```

### Model Pipeline
```
Training Dataset → XGBoost Pipeline → Model Registry
    ↓
Validation Dataset → Performance Metrics
    ↓
Production Inference → Monitoring Tables
    ↓
Performance Tracking → Retraining Triggers
```

## 🔍 Monitoring & Observability

### Performance Metrics
- F1 Score (primary metric)
- Precision, Recall, Accuracy
- Training vs Test performance gaps

### Drift Detection
- Feature distribution changes
- Prediction distribution shifts
- Label distribution evolution

### Quality Checks
- Data completeness and consistency
- Feature validation and correlation
- Label quality and temporal consistency

## 🎛️ Customization Points

### Easy Customizations
- Change churn window: Modify `CHURN_WINDOW` in config
- Adjust retraining threshold: Change `RETRAIN_THRESHOLD`
- Add new features: Extend `NUMERICAL_COLS` or `CATEGORICAL_COLS`
- Modify model: Replace XGBoost in `ModelTrainer`

### Advanced Customizations
- New data sources: Extend `DataIngestionPipeline`
- Custom features: Add methods to `FeatureEngineer`
- Different algorithms: Create new trainer classes
- Custom monitoring: Extend `ModelMonitor` and `DriftDetector`

## 🚨 Error Handling

### Built-in Safeguards
- Data validation at each step
- Configuration validation on startup
- Graceful handling of missing data
- Automatic rollback on training failures

### Logging & Debugging
- Comprehensive logging at all levels
- Performance timing for each operation
- Error context and stack traces
- Progress indicators for long operations

---

**This structure provides a complete, production-ready ML pipeline that's easy to understand, modify, and deploy in CI/CD environments.**

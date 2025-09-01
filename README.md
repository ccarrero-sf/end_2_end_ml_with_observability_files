# End-to-End ML Workflow with Observability

A complete machine learning pipeline for customer churn detection using Snowflake's ML capabilities. This project demonstrates a production-ready ML workflow with automated data ingestion, feature engineering, model training, monitoring, and observability.

## 🎯 Project Overview

This pipeline implements a comprehensive customer churn detection system with the following key features:

- **Customer churn prediction** using sales and feedback data
- **Sentiment analysis** of customer feedback using Cortex AI
- **Feature engineering** with Snowflake analytical functions
- **Model Registry** and Feature Store integration
- **Model monitoring** and drift detection
- **Automated retraining** based on performance metrics
- **Full ML observability** and lineage tracking

## 📁 Project Structure

```
├── main_pipeline.py           # Main orchestrator script
├── config.py                  # Configuration and constants
├── utils.py                   # Common utility functions
├── data_ingestion.py          # Data loading and processing
├── feature_engineering.py    # Feature creation and management
├── labeling.py               # Churn label generation
├── model_training.py         # ML model training and evaluation
├── model_monitoring.py       # Model monitoring and inference
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── E2E_ML_WORKFLOW_sklearn_f1.ipynb  # Original notebook
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Snowflake account with ML capabilities
- Required Python packages (see requirements.txt)

### Installation

1. Clone the repository and navigate to the project directory:
   ```bash
   cd end_2_end_ml_with_observability_mljobs
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up Snowflake connection (ensure you have an active Snowflake session)

### Running the Pipeline

The pipeline supports different execution modes:

#### 1. Initialize Environment Only
```bash
python main_pipeline.py --mode init
```

#### 2. Run Initial Training
```bash
python main_pipeline.py --mode train --months 4
```

#### 3. Run Complete Pipeline with Continuous Learning
```bash
python main_pipeline.py --mode continuous --months 4 --max-iterations 10
```

#### 4. Run Data Drift Simulation
```bash
python main_pipeline.py --mode drift --months 4
```

#### 5. Or Run Data all Steps at Once:
```bash
python main_pipeline.py --mode all
```


## 📊 Pipeline Components

### 1. Configuration (`config.py`)
- Environment variables and constants
- Feature engineering parameters
- Model training settings
- Table and schema definitions

### 2. Data Ingestion (`data_ingestion.py`)
- **DataIngestionPipeline**: Main data loading orchestrator
- **DataQualityChecker**: Data validation and quality checks
- Automated file discovery and processing
- Sentiment analysis using Snowflake Cortex AI

### 3. Feature Engineering (`feature_engineering.py`)
- **FeatureEngineer**: Creates customer behavioral features
- **FeatureValidator**: Validates feature quality and drift
- **FeatureStoreManager**: Manages Feature Store operations
- Time-series aggregations and customer profiling

### 4. Labeling (`labeling.py`)
- **ChurnLabeler**: Creates churn labels based on customer behavior
- **LabelQualityChecker**: Validates label consistency and patterns
- Automated label updates with new transaction data

### 5. Model Training (`model_training.py`)
- **ModelTrainer**: XGBoost model training and evaluation
- **ModelValidator**: Performance validation and overfitting checks
- **ModelComparator**: Model comparison and selection
- Automated model registration in Model Registry

### 6. Model Monitoring (`model_monitoring.py`)
- **ModelMonitor**: Inference and monitoring infrastructure
- **PerformanceTracker**: Performance tracking over time
- **DriftDetector**: Feature and prediction drift detection
- **AlertManager**: Alert generation and management

### 7. Main Pipeline (`main_pipeline.py`)
- **MLPipelineOrchestrator**: Complete pipeline orchestration
- Supports multiple execution modes
- Error handling and logging
- Performance tracking and reporting

## 🔧 Configuration

Key configuration parameters in `config.py`:

```python
# ML Parameters
CHURN_WINDOW = 30              # Days to define churn
RETRAIN_THRESHOLD = 0.8        # F1 threshold for retraining
TEST_SIZE = 0.2               # Train/test split ratio

# Snowflake Settings
SCHEMA = 'E2E_DEMO'           # Working schema
WAREHOUSE = 'COMPUTE_WH'      # Compute warehouse

# Feature Engineering
CATEGORICAL_COLS = ['GENDER', 'LOCATION', 'CUSTOMER_SEGMENT']
NUMERICAL_COLS = [...]        # Numerical feature columns
```

## 📈 Pipeline Workflow

### Phase 1: Initialization
1. **Environment Setup**: Create schemas, tables, and staging areas
2. **File Discovery**: Identify and register available data files
3. **Static Data Loading**: Load customer reference data

### Phase 2: Initial Training
1. **Data Ingestion**: Load training data (4 months by default)
2. **Feature Engineering**: Create behavioral features using Snowflake analytics
3. **Label Generation**: Create churn labels based on future behavior
4. **Model Training**: Train XGBoost classifier with preprocessing pipeline
5. **Model Registration**: Register baseline model in Model Registry
6. **Monitoring Setup**: Create monitoring infrastructure and baseline predictions

### Phase 3: Continuous Learning
1. **Incremental Data Loading**: Process new monthly data files
2. **Feature Updates**: Create features for new data
3. **Performance Monitoring**: Track model performance using F1 score
4. **Automated Retraining**: Retrain models when performance drops
5. **Model Selection**: Choose best performing model as production default

### Phase 4: Drift Detection (Optional)
1. **Drift Data Introduction**: Load data with different patterns
2. **Drift Monitoring**: Detect feature and prediction drift
3. **Adaptive Retraining**: Automatically adapt to changing data patterns

## 📊 Key Features

### Feature Engineering
- **Time-series aggregations**: Sales patterns over different time windows
- **Sentiment analysis**: Customer feedback sentiment using Cortex AI
- **Customer profiling**: Demographics and behavioral features
- **Automated feature validation**: Quality checks and drift detection

### Model Training
- **Preprocessing pipeline**: Automatic encoding and scaling
- **Cross-validation**: Train/test split with stratification
- **Performance metrics**: F1, precision, recall, accuracy
- **Model validation**: Overfitting and performance checks

### Monitoring & Observability
- **Real-time performance tracking**: F1 score monitoring
- **Feature drift detection**: Statistical drift analysis
- **Automated alerts**: Performance degradation notifications
- **Model lineage**: Complete tracking from data to predictions


## 🎛️ Environment Variables

You can customize the pipeline behavior using environment variables:

```bash
export ML_SCHEMA="MY_DEMO_SCHEMA"
export ML_WAREHOUSE="MY_WAREHOUSE"
export CHURN_WINDOW="45"
export RETRAIN_THRESHOLD="0.75"
```

## 📝 Logging

The pipeline provides comprehensive logging:
- **Console output**: Real-time progress and metrics
- **Log file**: Detailed execution logs (`ml_pipeline.log`)
- **Performance tracking**: Execution time for each component
- **Error handling**: Detailed error messages and stack traces

## 🔍 Monitoring Dashboards

The pipeline creates several monitoring views:

1. **Model Performance History**: F1 scores over time
2. **Feature Drift Analysis**: Statistical drift metrics
3. **Data Quality Metrics**: Completeness and consistency scores
4. **Pipeline Execution Logs**: Runtime and performance data

## 🚨 Troubleshooting

### Common Issues

1. **Session Connection**: Ensure active Snowflake session
2. **Permissions**: Verify schema and warehouse access
3. **Dependencies**: Check all required packages are installed
4. **Data Availability**: Confirm source data files are accessible

### Debug Mode

Run with detailed logging:
```python
from config import setup_logging
import logging

setup_logging(logging.DEBUG)
```

## 🔄 CI/CD Integration

The modular design supports CI/CD deployment:

1. **Environment-specific configs**: Use environment variables
2. **Automated testing**: Unit tests for each component
3. **Staged deployment**: Dev → Test → Prod progression
4. **Monitoring integration**: Connect to existing monitoring systems

## 📚 Additional Resources

- [Snowflake ML Documentation](https://docs.snowflake.com/en/developer-guide/snowpark-ml/index)
- [Feature Store Guide](https://docs.snowflake.com/en/developer-guide/snowpark-ml/feature-store/overview)
- [Model Registry Guide](https://docs.snowflake.com/en/developer-guide/snowpark-ml/model-registry/overview)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Built with ❤️ using Snowflake ML Platform**

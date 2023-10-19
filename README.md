# Northern Leaf Blight Detection in Maize Using Image Classification

## Overview

This project focuses on the early detection of Northern Leaf Blight (NLB), a common disease in maize, using image classification. NLB leads to significant crop loss and poses a threat to food security, especially in regions heavily reliant on maize as a staple food. By implementing machine learning and computer vision techniques, this tool provides an efficient, accurate, and user-friendly solution for farmers, agricultural researchers, and policymakers.

## Prerequisites

### Software Requirements
- **Python 3.8+**: Ensure you have the latest version installed.
- **Jupyter Notebook or Google Colab**: For interactive code execution and modification.

### Libraries

This project uses several Python libraries,including: 
TensorFlow and Keras: For building, training, and evaluating the deep learning model.
OpenCV: Used for image manipulation and processing, essential for preparing the data for model training.
Scikit-learn: For splitting the dataset and model evaluation metrics.
Matplotlib and Seaborn: For creating visualizations to analyze the dataset and interpret the model's performance.
Pandas and Numpy: For data manipulation and analysis.
 
 You can install them using pip:
```
pip install numpy pandas matplotlib seaborn tensorflow scikit-learn
```

### Hardware Requirements

- **Processor**: Intel i7 or equivalent.
- **Memory**: 16 GB RAM.
- **Storage**: At least 10 GB of available space.

## Dataset

The dataset used is publicly available through the [Open Science Framework (OSF)] (https://osf.io/p67rz/). It includes images of healthy maize plants and those infected with NLB, providing a diverse and comprehensive foundation for training our machine learning model.

## Getting Started

1. **Clone or download this repository** to your machine.
2. **Install necessary Python libraries** or open the notebook in Google Colab.
3. **Download the dataset** from the provided link and add it to your project directory.

## Structure
- `NLB_Detection_Capstone_Project.ipynb`: https://github.com/Felix-Awino/Capstone_Project_The_Dark_Code_Hub/tree/main/COLAB%20NOTEBOOK
- `/data/`: https://osf.io/p67rz/
- `/results/`: Graphs, visualizations, and analytical results-PPT
- `/website/`: Find our website here [**NLB_Maize_detection_Engine**](https://robinzulu-nlb-detection-model-app-5d8z5m.streamlit.app/)


## Usage

1. Open the collab notebook `NLB_Detection_Capstone_Project.ipynb`.
2. If using locally, run all the cells in sequence. If using Google Colab, you can simply click on `Runtime > Run all`.
3. The notebook includes detailed comments and explanations. Follow each section to understand the data preprocessing, model building, and evaluation steps. Below we have briefly outlined the methology used:

**Methodology**

This project follows a systematic approach to identify Northern Leaf Blight (NLB) in maize plants using image classification. The methodology is broken down into several key phases:

### 1. Data Collection
The first step in our analysis involved gathering a diverse set of maize plant images, including both healthy and NLB-infected samples. These images were sourced from [Open Science Framework (OSF)](https://osf.io/p67rz/), providing a comprehensive dataset conducive to robust model training.

### 2. Data Preprocessing
Once collected, the images underwent several preprocessing steps to optimize them for training. This phase included:
   - **Class balancing**: Data distribution between the classes was imbalanced. Oversampling, random selection, and class weights were used to balance the classes.
   - **Resizing**: To ensure consistency in input size, all images were resized to uniform dimensions.
   - **Normalization**: The pixel values were normalized to a specific range suitable for the model's input.
   - **Data Augmentation**: To enhance the model's generalization, data augmentation techniques such as rotation, zooming, and horizontal flipping were applied to the   
       training images.
   - **Splitting**: The dataset was divided into training, validation, and testing sets to evaluate the model's performance accurately.

### 3. Model Development
Model 1: Baseline Model CNN
- Constructed a convolutional neural network (CNN) with convolutional layers, activation functions, batch normalization, pooling layers, dropout layers, and dense layers, ending with a softmax output.
- Compiled using categorical cross-entropy loss and the Adam optimizer.
- One-hot encoded labels and evaluated on test data.

Improved Model - Data Augmentation and Balanced Training
- Created a balanced dataset.
- Utilized data augmentation techniques with ImageDataGenerator.
- Enhanced the CNN model with more layers and dropout.
- Lowered the learning rate for the Adam optimizer.
- Evaluated on an untouched test set.

Transfer Learning Model (VGG16)
- Employed VGG16 as the base model for our custom classifier.
- Applied data augmentation.
- Adjusted class weights to address class imbalance.
- Trained the model with class weights.
- Utilized a learning rate scheduler for optimization.

### 5. Model Deployment
The final step involved deploying the trained VGG16 model for real-world use. The deployment was structured to ensure ease of use, where end users can upload maize plant images and receive instant feedback on the health status based on the model's prediction.Link to the website [**NLB_Maize_detection_Engine**](https://robinzulu-nlb-detection-model-app-5d8z5m.streamlit.app/)

## Results
Our  best-performing model model demonstrated promising performance in identifying Northern Leaf Blight in maize plants. The evaluation metrics highlighted the model's ability to classify the health status of the plants accurately, underscoring its potential as a valuable tool in agricultural management.

- **Accuracy**: 0.8183
- **Precision**: 0.8228
- **Recall**: 0.9807
- **F1-Score**: 0.8948

The table below outlines all the model results: 
|          Model          |        AUC         |      F1 Score      |     Precision      |       Recall       |      Accuracy      |
|-------------------------|--------------------|--------------------|--------------------|--------------------|--------------------|
|     Baseline Model      | 0.6451284550485099 | 0.7707108498129343 | 0.8645083932853717 | 0.695274831243973  | 0.6740121580547113 |
|     Improved Model      | 0.6323866405367012 | 0.6509433962264152 | 0.8376327769347496 | 0.532304725168756  | 0.5501519756838906 |
| Transfer Learning Model | 0.7191443473211601 | 0.8948526176858776 | 0.8228155339805825 | 0.9807135969141755 | 0.8183890577507599 |

## Recommendations
1.Adopt Transfer Learning: Given the superior performance of the transfer learning model, it is recommended to adopt this approach for tasks similar to the one at hand, especially when high accuracy and generalization are required.

2.Enhance Data Processing: All models could potentially benefit from more diverse data and enhanced augmentation techniques. This could involve adding more variation in image data or implementing advanced augmentation strategies to increase the robustness of the models to variations in input.

3.Regularization and Early Stopping: To combat overfitting and improve generalization, especially for the baseline and improved models, incorporate more stringent regularization techniques and early stopping mechanisms. These strategies could help models generalize better to unseen data.

4.Hyperparameter Optimization: Systematic hyperparameter tuning, perhaps through grid search or randomized search, could fine-tune the models for potentially better performance. This is particularly relevant for the learning rate and optimization algorithms.

5.Model Evaluation: Continue to use comprehensive metrics for model evaluation. While accuracy is important, metrics such as AUC, F1 Score, precision, and recall provide a more holistic view of model performance, especially in tasks where false positives or false negatives have significant implications.

## Next Steps
1.Experiment with Advanced Models: Explore more complex architectures or ensemble methods to see if these approaches can further improve accuracy and generalization.

2.Invest in Data Quality: If possible, invest in higher-quality data or more diverse datasets to provide the models with a richer learning environment. The quality of input data is often a determining factor in the success of machine learning models.

3.Feedback Loop: Implement a system for continuous feedback and learning. As the model is used, it's important to collect data on its predictions and actual outcomes to refine and improve the model over time.

4.Operationalization: With the transfer learning model showing promising results, begin preparations for deploying the model in a real-world environment. This involves setting up an infrastructure that can handle model requests, ensuring security protocols are followed, and establishing a monitoring system for model performance.

5.User Testing: Before full-scale deployment, conduct extensive user testing to gather feedback on model performance in real-world scenarios. This step is crucial for identifying unforeseen issues and areas for improvement.



## Contributing

We welcome contributions! Please read the contribution guidelines at: [**Contributing Guidelines**](https://github.com/Felix-Awino/Capstone_Project_The_Dark_Code_Hub/tree/main/Contributing%20guidelines)

## Contact

For help or issues involving this project, please contact us at:

imelda.masika@student.moringaschool.com

sheila.kamanzi@student.moringaschool.com

bryan.okwach@student.moringaschool.com

felix.awino@student.moringaschool.com

muthoni.kahura@student.moringaschool.com

robin.mutai@student.moringaschool.com


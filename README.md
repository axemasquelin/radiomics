# Perinodular Parenchymal Features Improve Indeterminate Lung Nodule Classification
## Abstract
**Purpose:** To evaluate the consistency with which perinodular radiomic features extracted from low-dose computed tomography serve to distinguish benign from cancerous pulmonary nodules using a variety of machine learning approaches.

**Materials and Methods:** Using the National Lung Screening Trial (NLST), we selected for individuals with pulmonary nodules between 4mm to 20mm in diameter. Nodules were then segmented to generate four unique datasets; a tumor-specific dataset describing tumor-specific features, a 10mm Band dataset that contains features from the segmented nodule boundary to 10mm out from the boundary, a 15mm Band dataset, and a Lung-RADS diameter dataset containing the nodules maximum diameter. Models to predict malignancy were constructed using support-vector machine (SVM), random forest (RF), and least absolute shrinkage and selection operator (LASSO) approaches. Five-fold cross validation with 10 repetitions per fold was used to evaluate the performance of each methodology and dataset.

**Results:** With respect to the RF, the Tumor dataset achieved an area-under the receiver-operator curve (AUC), sensitivity, and specificity of 84.44%, 74.40%, 81.39%, respectively. The 10mm Band dataset performed similarly to the Tumor dataset 84.09% (adj. p-value 1.00), with regards to AUC. A drop in performance is seen with the 15mm Band dataset, achieving an AUC of 81.57% (adj. p-value 6.534×10^(-4)). However, when combining tumor-specific features with perinodular features (10mm Band + Tumor), the 10mm Band + Tumor and 15mm Band + Tumor datasets (AUC 87.87% and 86.75%, respectively) performed significantly better than the Maximum-diameter dataset (66.76%) or the Tumor dataset. Similarly, the SVM’s and LASSO’s AUCs were 84.71% and 88.91% for the 10mm Band + Tumor, respectively.

**Conclusions:** The 10mm Band + Tumor dataset improved the differentiations of benign and malignant lung nodules, compared to the Tumor datasets across all methodologies. Parenchymal features capture novel diagnostic information beyond that present in the nodule and significantly improve nodule discrimination.   

## Running the Program

Available Methodologies:  
&nbsp; 1. Random Forest  
&nbsp; 2. Support Vector Machine  
&nbsp; 3. LASSO Regression  
&nbsp; 4. LungRads Cutoffs (Based on Guidelines)  


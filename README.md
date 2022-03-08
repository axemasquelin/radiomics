# Perinodular Parenchymal Features Improve Indeterminate Lung Nodule Classification
## Abstract
**Background:** Radiomics, quantitative features extracted from images, provides a non-invasive methodology to analyses malignant and benign pulmonary nodules.

**Purpose:** In this study, we evaluate the consistency with which perinodular radiomic features extracted from low dose computed tomography serve to distinguish malignant pulmonary nodules. 

**Materials and Methods:** Using the National Lung Screening Trial (NLST), we selected  individuals with pulmonary nodules between 4mm to 20mm in diameter. Nodules were segmented to generate four distinct datasets; 1) a Tumor dataset containing tumor-specific features, 2) a 10 mm Band dataset containing parenchymal features between the segmented nodule boundary and 10mm out from the boundary, 3) a 15mm Band dataset, and 4) a Tumor Size dataset containing the maximum nodule diameter. Models to predict malignancy were constructed using support-vector machine (SVM), random forest (RF), and least absolute shrinkage and selection operator (LASSO) approaches. Ten-fold cross validation with 10 repetitions per fold was used to evaluate the performance of each approach applied to each dataset. 

**Results:** With respect to the RF, the Tumor, 10mm Band, and 15mm Band datasets achieved area-under the receiver-operator curve (AUC) of 84.44%, 84.09%, and 81.57%, respectively. Significant difference in performance was observed between the Tumor and 15mm Band datasets (adj. p-value <.001). However, when combining tumor-specific features with perinodular features (10mm Band + Tumor), the 10mm Band + Tumor and 15mm Band + Tumor datasets (AUC 87.87% and 86.75%, respectively) performed significantly better than the Maximum-diameter dataset (66.76%) or the Tumor dataset. Similarly, the SVM’s and LASSO’s AUCs were 84.71% and 88.91% for the 10mm Band + Tumor, respectively.

**Conclusions:** The 10mm Band + Tumor dataset improved the differentiations of benign and malignant lung nodules, compared to the Tumor datasets across all methodologies. Parenchymal features capture novel diagnostic information beyond that present in the nodule and significantly improve nodule discrimination.    

## Running the Program

Available Methodologies:  
&nbsp; 1. Random Forest  
&nbsp; 2. Support Vector Machine  
&nbsp; 3. LASSO Regression  
&nbsp; 4. LungRads Cutoffs (Based on Guidelines)  


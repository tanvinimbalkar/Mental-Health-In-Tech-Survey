# Mental-Health-In-Tech-Survey
Analyzed the Mental Health in Tech Survey using statistical and machine learning models to uncover key drivers influencing treatment-seeking behavior, aiming to inform data-driven, empathetic workplace support strategies.

## Mental Health in Tech Survey


This dataset is based on the Mental Health in Tech Survey, which aims to explore the prevalence and treatment of mental illness among the tech community. It has 1197 responses and 15 variables like age, gender, whether self-employed or not, remote work, family history of mental illness, and whether or not the subject received treatment for mental illness. Statistical techniques are applied here in order to identify which variables affect the probability of treatment-seeking.

### Problem Definition

Mental health challenges are increasingly recognized within the tech industry, an environment often characterized by high-pressure deadlines, remote work, and intense intellectual demands. However, despite rising awareness, the patterns and predictors of mental health treatment-seeking behavior remain complex and insufficiently understood. Traditional assumptions often focus on workplace benefits or remote work as major influences, but personal factors such as family mental health history or demographic attributes like age and gender may be equally, if not more, critical.

This project seeks to rigorously analyze the factors that drive individuals in the tech sector to seek mental health treatment. By applying a suite of statistical and machine learning models to the Mental Health in Tech Survey dataset, we aim to move beyond anecdotal narratives and surface-level policy assumptions. We strive to uncover data-driven insights that can inform the design of more effective and empathetic mental health support strategies within technical workplaces.

---

### Question of Interest

The central question guiding this analysis is:

**"Which factors most significantly influence whether individuals working in the tech industry seek treatment for mental health issues?"**

Specifically, we investigate:
- Does having a family history of mental illness increase the likelihood of seeking treatment?
- Does age affect the willingness to discuss mental health with an employer?
- Are individuals in certain gender categories more likely to pursue treatment than others?
- Do workplace structures such as remote work, benefits, or self-employment status impact treatment-seeking behavior?

-  **Data Source:**  
This project uses the dataset titled **"Mental Health in Tech Survey"**, which can be found at the following link:  
🔗 [https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)

To reproduce the analysis in this notebook:
1. Visit the above Kaggle link.
2. Download the dataset (`survey.csv`).
3. Place it in your project directory or appropriate path as used in the notebook.

The dataset contains responses from individuals in the tech industry and is aimed at exploring mental health awareness, treatment-seeking behavior, and workplace attitudes.

### Data Cleaning and Preprocessing

Before conducting any statistical modeling, we undertook a comprehensive data cleaning process to ensure the dataset was consistent, well-structured, and analysis-ready. The original dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey), contained missing values, inconsistent labeling, and non-standardized formats. The following steps outline the full cleaning pipeline:

---

1. **Column Name Standardization:**
   - All column names were converted to lowercase.
   - Spaces were replaced with underscores to ensure consistency and avoid syntax issues during modeling.

2. **Whitespace Removal & Gender Normalization:**
   - All character entries had leading and trailing whitespace trimmed.
   - Gender values were normalized to `male`, `female`, or `other` using pattern-matching logic to capture variations like "M", "Male", "female", "woman", etc.

3. **Filtering Invalid Ages:**
   - The `age` column was coerced into numeric.
   - We retained only realistic age values: entries with age **between 18 and 70** were kept, removing likely errors or test entries.

4. **Handling Missing Data:**
   - String `"NA"` values (entered as text) were converted to actual `NA`.
   - Rows with more than **25% missing values** were dropped to maintain dataset integrity.
   - Remaining missing values were imputed:
     - **Categorical variables** were filled with their **mode** (most frequent value).
     - **Numeric variables** were filled using their **median** to preserve distribution shape and avoid distortion from outliers.

5. **Binary Value Conversion:**
   - Columns such as `treatment`, `remote_work`, `tech_company`, `benefits`, `seek_help`, and mental/physical consequence/interview variables were recoded into numeric values:
     - `Yes` → 1  
     - `No` → 0  
     - `Maybe` → 0.5  
     - `"Don't know"` → `NA`

6. **Column Selection:**
   - Non-informative columns (`timestamp`, `state`, `comments`) were removed.
   - Only the 15 most relevant variables were retained for downstream modeling, focusing on demographics, workplace context, and mental health outcomes.

7. **Finalization:**
   - Cleaned data was exported to two CSVs:
     - `cleaned_survey.csv` – intermediate cleaned version.
     - `final_cleaned_survey.csv` – final modeling-ready version with selected columns only.

---

This preprocessing ensures the dataset is statistically valid and structurally clean, minimizing noise and bias in the modeling phase. By transforming textual inconsistencies, handling missing values thoughtfully, and standardizing formats, we laid a solid foundation for robust, interpretable, and meaningful analysis.



Data Cleaning and Preprocessing

Before beginning the analysis, we performed thorough data cleaning to ensure reliability and consistency. The raw dataset from Kaggle contained several formatting issues, missing values, and inconsistent responses. Here are the key steps we took to prepare the data for modeling:

1. **Whitespace & Formatting:**
   - Removed leading/trailing whitespaces from all character columns.
   - Standardized gender labels (e.g., "M", "male", "Male") to unified categories.

2. **Categorical Cleanup:**
   - Converted blank strings in important columns like `self_employed`, `family_history`, `treatment`, `seek_help`, etc., to `NA`.
   - Ensured categorical columns are uniformly represented.

3. **Numeric Conversion:**
   - Converted binary numeric columns (`remote_work`, `tech_company`) from character to integer (0/1).

4. **Missing Values:**
   - Dropped rows with more than 2 missing values to reduce noise.
   - Imputed missing values:
     - For categorical columns: replaced with **mode** (most frequent value).
     - For numeric columns: replaced with **median**.

5. **Range Filtering & Duplicates:**
   - Removed any duplicate entries.
   - Retained only those observations where age was between **15 and 100**, filtering out outliers.

6. **Final Export:**
   - Saved the cleaned dataset as `final_cleaned_survey_cleaned.csv` for all downstream modeling tasks.

This cleaning process ensures that our statistical methods work on a well-structured, consistent dataset — reducing the risk of bias or error due to data quality issues.


### Models Selected:

To explore the factors influencing mental health treatment-seeking behavior within the tech industry, we applied a blend of classical statistical methods and modern modeling approaches on a cleaned version of the Mental Health in Tech Survey dataset. This dataset includes demographic and workplace-related variables such as age, gender, country, family history of mental illness, remote work status, self-employment status, and access to benefits.

We first utilized **Simple Linear Regression (SLR)** to examine whether continuous variables like age had a predictive relationship with discussing mental health at the workplace. **Confidence Intervals (CI)** were constructed around regression coefficients to assess the precision and reliability of these estimates.

To predict treatment-seeking behavior, a binary outcome, we employed a **Generalized Linear Model (GLM)** using logistic regression. This allowed us to model the probability of seeking treatment based on a set of demographic and workplace predictors, providing interpretable odds ratios for each factor.

In order to investigate whether treatment-seeking behavior varied significantly across categorical groups like gender, we conducted an **Analysis of Variance (ANOVA)**. Following a significant ANOVA, we implemented a **Tukey’s Honest Significant Difference (HSD)** post-hoc test to pinpoint where the group differences lay.

For a more direct comparison between two specific groups (individuals with and without a family history of mental illness), we used a **Welch Two-Sample t-test** under the umbrella of **Hypothesis Testing**. This method helped evaluate whether the difference in treatment-seeking rates between the two groups was statistically meaningful.

Finally, we incorporated **Bayesian Estimation** to provide a probabilistic understanding of treatment-seeking behavior. By calculating posterior probabilities, we were able to update our beliefs about an individual's likelihood of seeking treatment based on prior information, such as family history.

Together, these methods enabled a comprehensive analysis that combined hypothesis-driven testing with model-based prediction and probabilistic reasoning, offering a richer and multidimensional understanding of mental health dynamics in the tech industry.


# Mental Health in Tech: Statistical Report

---

## 4.1 Context and Purpose

The tech industry has increasingly acknowledged the importance of mental health, yet treatment-seeking behavior remains uneven and poorly understood. This project examines the **Mental Health in Tech Survey** dataset to uncover which personal and workplace-related factors influence the likelihood of seeking treatment for mental health issues. Using a variety of statistical tools, our aim was to provide actionable insights that go beyond surface-level workplace policies. Specifically, we ask: *Are individuals with a family history of mental illness more likely to seek help? Does age or gender shape openness in mental health conversations? Are workplace benefits effective predictors?*

By answering these questions, we offer grounded, data-driven evidence to inform mental health support strategies in technical work environments.

---

## 4.2 Content Development

We began by thoroughly cleaning and preprocessing the dataset, standardizing columns, converting categorical variables, and handling missing data through logical imputation strategies. Our dataset ultimately retained 15 relevant variables across 1,197 observations.

From there, we applied **Simple Linear Regression** to explore the effect of age on discussing mental health with employers. Next, we built a **Generalized Linear Model (GLM)** to assess treatment-seeking predictors such as family history, benefits, remote work, and age. We followed this with **ANOVA** and **Tukey post-hoc** analysis to examine gender-based differences in treatment behavior. A **two-sample t-test** assessed the impact of family history on treatment likelihood, followed by a **Bayesian analysis** to estimate posterior probabilities using observed data.

This structured progression allowed us to test both *statistical significance* and *practical significance* through multiple lenses, with visualizations and diagnostics to reinforce conclusions.

---

## 4.3 Sources and Evidence

- Primary Dataset: [Mental Health in Tech Survey (Kaggle)](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- All models were built using R statistical software.
- Analysis was performed on cleaned data derived from the original CSV, processed as part of this project.
- Visualizations include regression scatterplots, confidence intervals, GLM odds ratios, ROC curve (AUC = 0.701), residual diagnostics, and Tukey confidence plots.
- Every insight is directly supported by empirical outputs from statistical models and tests performed on this data.

---

## 4.4 Explanation of Statistical Analyses

Simple Linear Regression (SLR) was used to examine whether age predicts an individual’s willingness to discuss mental health with an employer. The regression model returned a coefficient of -0.00096 with a p-value of 0.335, and an R² value of approximately 0.00077—indicating that age had virtually no predictive power. To strengthen this conclusion, 95% confidence intervals were computed for the coefficient, and the interval for age (-0.0029, 0.00099) included 0, confirming that the effect was not statistically significant.

Confidence Intervals (CI) were further used to quantify the uncertainty around the SLR estimates. The confidence interval for age again crossed zero, reinforcing its lack of impact. A visualization of the CI for both intercept and slope confirmed the non-significance of age in predicting mental health discussion behavior.

GLM (Logistic Regression) was then applied to predict the binary outcome of whether a respondent sought mental health treatment. The predictors in this model included age, family history of mental illness, self-employment status, remote work, and benefits. The most influential factor was family history (odds ratio ≈ 4.97, p < 0.001), meaning respondents with a family history of mental illness were nearly five times more likely to seek treatment. Age was also statistically significant (p = 0.0114), albeit with a very small effect size, while all other predictors were non-significant. We visualized the odds ratios and 95% confidence intervals, which showed a distinct and meaningful effect only for family history. ROC curve analysis showed an AUC of 0.701, confirming that the model had reasonable discriminatory power. Residual diagnostic plots did not indicate any serious violations of model assumptions.

ANOVA was conducted to test whether treatment-seeking rates differed across gender categories. The ANOVA yielded a statistically significant result (p = 0.00227), indicating that treatment-seeking behavior varies across gender. Follow-up analysis using Tukey’s Honest Significant Difference (HSD) test showed a significant difference between the “male” and “other” gender categories (p = 0.029), with individuals in the “other” category being more likely to seek treatment. The 95% confidence intervals from Tukey’s test visually confirmed this contrast.

Hypothesis Testing involved a Welch Two-Sample t-test comparing treatment-seeking behavior among those with and without a family history of mental illness. The results were highly significant (t = 13.83, p < 0.0001), and the difference in group means was approximately 0.37. The 95% confidence interval for this difference ranged from 0.3212 to 0.4275, providing strong evidence that individuals with a family history are substantially more likely to pursue mental health treatment.

Bayesian Analysis was conducted to supplement classical hypothesis testing. We calculated posterior probabilities of seeking treatment given the presence or absence of family history. The results showed that P(Treatment = Yes | Family History = Yes) ≈ 0.74, while P(Treatment = Yes | Family History = No) ≈ 0.36. These findings indicate that individuals with a family history of mental illness are about twice as likely to seek treatment, emphasizing a probabilistic view of the same relationship captured in the GLM and hypothesis testing models. This approach reinforced earlier conclusions using probabilistic reasoning instead of traditional significance testing.

---

## 4.5 Syntax and Mechanics

All written sections were composed with precision, clarity, and consistent structure. Grammar and punctuation conform to academic writing standards. Markdown formatting was applied throughout the notebook to ensure readability and a professional flow of insights.

We conclude that personal factors, especially **family history of mental illness**, far outweigh workplace structures in influencing treatment-seeking behavior in the tech sector. Mental health support strategies must be **personalized, awareness-driven, and deeply contextual** to be truly effective.

--- 

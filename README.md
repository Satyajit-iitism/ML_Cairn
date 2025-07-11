# ML_Cairn

This repository contains machine learning models used for **log prediction** from LAS files.

We aim to predict the **sonic log (VP)** using other available logs.  
The process involves data cleaning, feature engineering (including optional geological inputs), and training ML models.

#order to execute the code
##LAS to CSV
##Encode_formation
##Run any of the one models (Random_Forest_Train_Test, random_forest_with_Train_Test_validation.py, XGBoost_with_validation_wells.py)
##Save_model
##Predict_new_wells


---

## 🧠 Objective

Work with **log data (LAS files)** and build ML models to predict:

- **Target**: `VP` (Sonic)
- **Inputs**: `GR`, `RHOB`, `TNPH`, `LLD`, and optionally `FORMATION`

---

## 📊 Data Preparation

To ensure model consistency and accuracy, follow these guidelines:

1. **Use wells from a single basin or field**  
   (e.g., Barmer Basin) for geologically consistent results.

2. **Convert LAS files to CSV**  
   For easier data handling and preprocessing.

3. **Add formation markers as input features** (optional, but improves accuracy).  
   This requires interpreting geological data from the field — make sure you understand the subsurface structure before applying.

4. **How to add FORMATION column:**

Example:

| DEPT | GR | RHOB | TNPH | LLD | VP | FORMATION    |
|------|----|------|------|-----|----|--------------|
| 1    | x  | x    | x    | x   | x  | formation-1  |
| 2    | x  | x    | x    | x   | x  | formation-1  |
| 3    | x  | x    | x    | x   | x  | formation-1  |
| 4    | x  | x    | x    | x   | x  | formation-2  |
| 5    | x  | x    | x    | x   | x  | formation-2  |
| ...  | ...| ...  | ...  | ... | ...| ...          |

5. **Clean the data**  
   Delete all rows where **any** of the following columns are missing:




---

## 🔁 Encoding the FORMATION Column

If using formation as a feature, convert the categorical values into boolean (one-hot) encoding.

Example:

| DEPT | GR | RHOB | TNPH | LLD | VP | FORMATION_1 | FORMATION_2 | FORMATION_3 |
|------|----|------|------|-----|----|--------------|--------------|--------------|
| 1    | x  | x    | x    | x   | x  | TRUE         | FALSE        | FALSE        |
| 2    | x  | x    | x    | x   | x  | TRUE         | FALSE        | FALSE        |
| 4    | x  | x    | x    | x   | x  | FALSE        | TRUE         | FALSE        |
| 8    | x  | x    | x    | x   | x  | FALSE        | FALSE        | TRUE         |

---

## ✅ Summary

- Use **consistent well data** from a single basin.
- Preprocess LAS to CSV and clean missing values.
- Use **formation encoding** if geological interpretation is available.
- Train ML models to predict **VP** using the selected logs.

---

> **Note**: Proper preprocessing is critical. Garbage in = Garbage out!

###Here I am not using Depth as a feature but you can use it if your filed doesn't have a variable depth





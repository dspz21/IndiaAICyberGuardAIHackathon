# IndiaAICyberGuardAIHackathon - Team S4AI

This repository contains code and resources for the **IndiaAICyberGuardAIHackathon**, created by Team S4AI. It demonstrates training and testing models using various approaches and deploying them using Modal.

## Steps to Run the Project - Approach A

Import the approach_A_Sentence_Transformer.ipynb in Google colab. Upload the mounted_dir in colab. Run Colab

## Prerequisites - Approach B and C

Before proceeding, ensure you have the following:

1. **Python**: Version 3.8 or later installed.
2. **Modal Account**: Create an account and generate an API token at [modal.com](https://modal.com).
3. **Modal Client**: Install the Modal Python client:
   ```bash
   pip install modal
   ```

## Steps to Run the Project - Approach B and C

### 1. Clone the Repository

Clone this repository and navigate to the project folder:

```bash
git clone <repository-url>
cd <repository-folder>
```

### 2. Set Modal Token

Set your Modal API token:

```bash
modal token set [INSERT TOKEN VALUE HERE]
```

### 3. Run the Application

Use the following command to run your file via Modal:

```bash
modal run [FileName]
```

Replace `[FileName]` with the relevant Python script (e.g., `test_approach_B_C.py` for Models B and C).

---

## Model Outputs After Training

The models trained using `train.csv` (with three approaches) can be downloaded from the links below:

- **Model A**: [Download](https://drive.google.com/file/d/1-LG9rGpjYR6OBZ6NMBYRv2azNg50jGC2/view?usp=sharing)
- **Model B**: [Download](https://drive.google.com/file/d/1IOqK0LXF_pcyu6Ioou-WrCVCDLo-Muqk/view?usp=sharing)
- **Model C**: [Download](https://drive.google.com/file/d/1qEZEdUeSMOhoEIujnHZleJjv3RuLo75w/view?usp=sharing)

---

## Testing the Models

### Model A

Testing code is included in the **same notebook** used for training.

### Model B and Model C

Use the script `test_approach_B_C.py` to test Models B and C. Ensure the script is properly refactored and ready for execution.

## Test Results

Test predictions are available in Test_pred_Approach_A/B/C.csv files.

---

## Notes

- Refer to [Modal Documentation](https://modal.com/docs) for troubleshooting or advanced configurations.

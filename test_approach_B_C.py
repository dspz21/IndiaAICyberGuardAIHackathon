import modal
import os
# import torch
# from transformers import DistilBertTokenizer
from pathlib import Path


# Define constants
MODEL_NAME = "distilbert-base-uncased"
OUTPUT_MODEL_FILE = "/workspace/pytorch_distilbert_news"
OUTPUT_VOCAB_FILE = "/workspace/vocab_distilbert_news.bin"
MOUNT_DIR = "./mounted_dir"

# Stub definition
app = modal.App("test_distill")

# Mount a local directory
image = modal.Image.debian_slim().pip_install(["torch", "transformers", "pandas", "nltk", "imbalanced-learn","numpy"])

# app.mount(MOUNT_DIR, remote_path="/workspace")

volume = modal.Volume.from_name("model-weights-vol-z21", create_if_missing=True)
MODEL_DIR = Path("/models")

@app.function(image=image, mounts=[modal.Mount.from_local_dir(MOUNT_DIR, remote_path="/workspace")], gpu="any", timeout=66000, volumes={MODEL_DIR: volume})
def test_model():
    import torch
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    
    from nltk.corpus import stopwords
    from transformers import DistilBertTokenizer
    from collections import Counter
    import numpy as np
    from transformers import DistilBertTokenizer
    import pandas 
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import nltk
    from nltk.corpus import stopwords
    import torch
    import transformers
   

    from transformers import DistilBertModel, DistilBertTokenizer
        
    from imblearn.over_sampling import RandomOverSampler
    from collections import Counter


    # Load the trained model
    class DistillBERTClass(torch.nn.Module):
        def __init__(self, num_labels):
            super(DistillBERTClass, self).__init__()
            self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.pre_classifier = torch.nn.Linear(768, 128)
            self.dropout = torch.nn.Dropout(0.2)
            self.classifier = torch.nn.Linear(128, num_labels)

        def forward(self, input_ids, attention_mask):
            output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output_1[0]
            pooler = hidden_state[:, 0]
            pooler = self.pre_classifier(pooler)
            pooler = torch.nn.ReLU()(pooler)
            pooler = self.dropout(pooler)
            output = self.classifier(pooler)
            return output

    # Load tokenizer and labels
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    
    label2id = {'Any Other Cyber Crime - Other': 0, 'Child Pornography CPChild Sexual Abuse Material CSAM - ': 1, 'Cryptocurrency Crime - Cryptocurrency Fraud': 2, 'Cyber Attack/ Dependent Crimes - Data Breach/Theft': 3, 'Cyber Attack/ Dependent Crimes - Denial of Service (DoS)/Distributed Denial of Service (DDOS) attacks': 4, 'Cyber Attack/ Dependent Crimes - Hacking/Defacement': 5, 'Cyber Attack/ Dependent Crimes - Malware Attack': 6, 'Cyber Attack/ Dependent Crimes - Ransomware Attack': 7, 'Cyber Attack/ Dependent Crimes - SQL Injection': 8, 'Cyber Attack/ Dependent Crimes - Tampering with computer source documents': 9, 'Cyber Terrorism - Cyber Terrorism': 10, 'Hacking  Damage to computercomputer system etc - Damage to computer computer systems etc': 11, 'Hacking  Damage to computercomputer system etc - Email Hacking': 12, 'Hacking  Damage to computercomputer system etc - Tampering with computer source documents': 13, 'Hacking  Damage to computercomputer system etc - Unauthorised AccessData Breach': 14, 'Hacking  Damage to computercomputer system etc - Website DefacementHacking': 15, 'Online Cyber Trafficking - Online Trafficking': 16, 'Online Financial Fraud - Business Email CompromiseEmail Takeover': 17, 'Online Financial Fraud - DebitCredit Card FraudSim Swap Fraud': 18, 'Online Financial Fraud - DematDepository Fraud': 19, 'Online Financial Fraud - EWallet Related Fraud': 20, 'Online Financial Fraud - Fraud CallVishing': 21, 'Online Financial Fraud - Internet Banking Related Fraud': 22, 'Online Financial Fraud - UPI Related Frauds': 23, 'Online Gambling  Betting - Online Gambling  Betting': 24, 'Online and Social Media Related Crime - Cheating by Impersonation': 25, 'Online and Social Media Related Crime - Cyber Bullying  Stalking  Sexting': 26, 'Online and Social Media Related Crime - EMail Phishing': 27, 'Online and Social Media Related Crime - FakeImpersonating Profile': 28, 'Online and Social Media Related Crime - Impersonating Email': 29, 'Online and Social Media Related Crime - Intimidating Email': 30, 'Online and Social Media Related Crime - Online Job Fraud': 31, 'Online and Social Media Related Crime - Online Matrimonial Fraud': 32, 'Online and Social Media Related Crime - Profile Hacking Identity Theft': 33, 'Online and Social Media Related Crime - Provocative Speech for unlawful acts': 34, 'Ransomware - Ransomware': 35, 'RapeGang Rape RGRSexually Abusive Content - ': 36, 'Report Unlawful Content - Against Interest of sovereignty or integrity of India': 37, 'Sexually Explicit Act - ': 38, 'Sexually Obscene material - ': 39}
    id2label = {id: label for label, id in label2id.items()}


    # Initialize and load model
    model = DistillBERTClass(num_labels=len(label2id))
    model.load_state_dict(torch.load(MODEL_DIR / "model4.bin", map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    nltk.download('stopwords')

    # Preprocessing
    def preprocess_data(df):
        stop = stopwords.words('english')
        df['crimeaditionalinfo'] = df['crimeaditionalinfo'].fillna('')
        df['crimeaditionalinfo'] = df['crimeaditionalinfo'].apply(
            lambda x: ' '.join([word for word in x.split() if word not in (stop)])
        )
        df['sub_category'] = df['sub_category'].fillna('')
        df['label'] = df['category'] + ' - ' + df['sub_category']
        return df[['crimeaditionalinfo', 'label']]

    df_test = pandas.read_csv('/workspace/test.csv')
    df_test = preprocess_data(df_test)

    # Check for NaN or infinite values in 'label' column
    print(df_test['label'].isna().sum())  # Number of NaN values
    print((df_test['label'] == float('inf')).sum())  # Number of infinite values
    # Drop rows where 'text' or 'label' is blank or NaN
    df_test = df_test.dropna(subset=['crimeaditionalinfo', 'label'])  # Remove rows where either 'text' or 'label' is NaN
    
    df_test['label'] = df_test['label'].map(label2id).fillna(0).astype(int)
    #df_test['label'] = pandas.to_numeric(df_test['label'], errors='coerce')


    class Triage(Dataset):
        def __init__(self, dataframe, tokenizer, max_len):
            self.len = len(dataframe)
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __getitem__(self, index):
            try:
                targets = torch.tensor(self.data.label[index], dtype=torch.long)
            except Exception as e:
                print(f"Error at index {index}: {self.data.label[index]}")
                raise e
            text = str(self.data.crimeaditionalinfo[index])  # text field
            text = " ".join(text.split())
            inputs = self.tokenizer.encode_plus(
                text,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            targets = self.data.label[index]

            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'targets': targets,
                'text': text  # Add text to the returned data
            }

        def __len__(self):
            return self.len

    # Test loader
    MAX_LEN = 512
    VALID_BATCH_SIZE = 16
    test_set = Triage(df_test, tokenizer, MAX_LEN)
    print(test_set.data)
    
    test_loader = DataLoader(test_set, batch_size=VALID_BATCH_SIZE, shuffle=False)

    def evaluate():
        n_correct = 0
        n_total = 0
        all_texts = []
        all_predictions = []
        all_targets = []
        
        model.eval()
        with torch.no_grad():
            for _, data in tqdm(enumerate(test_loader), total=len(test_loader)):
                ids = data['ids'].to("cuda", dtype=torch.long)
                mask = data['mask'].to("cuda", dtype=torch.long)
                targets = data['targets'].to("cuda", dtype=torch.long)

                outputs = model(ids, mask)
                _, preds = torch.max(outputs, dim=1)
                n_correct += (preds == targets).sum().item()
                n_total += targets.size(0)

                # Collect data for saving predictions
                all_texts.extend(data['text'])  # Ensure `text` is part of the dataloader
                all_predictions.extend(preds.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        # Calculate accuracy
        accuracy = n_correct / n_total * 100
        print(f"Accuracy on Test Data: {accuracy:.2f}%")

        # Save predictions to CSV
        predictions = [id2label[pred] for pred in all_predictions]  # Convert predictions to label names
        targets = [id2label[target] for target in all_targets]      # Convert ground truth to label names
        output_df = pandas.DataFrame({
            "text": all_texts,
            "true_label": targets,
            "predicted_label": predictions
        })
        output_df.to_csv(MODEL_DIR / "test_preds_22_11_model0.csv", index=False)
        print("Test predictions saved to test_preds_22_11_model0.csv in the model volume.")

        return accuracy

    # Call evaluation
    accuracy = evaluate()
    return accuracy

@app.local_entrypoint()
def main():
    # Call test_model
    accuracy = test_model.remote()
    print(f"Test Accuracy: {accuracy}")

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
app = modal.App("distilbert-fine-tuning")

# Mount a local directory
image = modal.Image.debian_slim().pip_install(["torch", "transformers", "pandas", "nltk"])

# app.mount(MOUNT_DIR, remote_path="/workspace")

volume = modal.Volume.from_name("model-weights-vol-z21", create_if_missing=True)
MODEL_DIR = Path("/models")

@app.function(image=image, mounts=[modal.Mount.from_local_dir(MOUNT_DIR, remote_path="/workspace")], gpu="H100:2",timeout=66000, volumes={MODEL_DIR: volume})
def train_model():
    from transformers import DistilBertTokenizer
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from tqdm import tqdm
    import nltk
    from nltk.corpus import stopwords
    import torch
    import transformers
    from transformers import DistilBertModel, DistilBertTokenizer

    # Your dataset and preprocessing code here
    # Assuming train_dataset_df and test_dataset_df are prepared
    # Load tokenizer


    with open(MODEL_DIR / "abc.txt", 'w') as f:
        f.write('Testing write permissions.')
       

    OUTPUT_MODEL_FILE
    
    MAX_LEN = 512
    TRAIN_BATCH_SIZE = 256
    VALID_BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 3e-05
    DROPOUT = 0.3
    GRADIENT_ACCUMULATION_STEPS = 4

    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)


    df_train = pd.read_csv('/workspace/train.csv')
    df_test = pd.read_csv('/workspace/test.csv')

    
    nltk.download('stopwords')

    stop = stopwords.words('english')
    def update_dataframe(df):
        df['crimeaditionalinfo'] = df['crimeaditionalinfo'].fillna('')
        df['crimeaditionalinfo'] = df['crimeaditionalinfo'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

        df_category = df[['category', 'sub_category']].sort_values(['category', 'sub_category'])
        df_category = df_category.drop_duplicates().reset_index()
        df_category['sub_category'] = df_category['sub_category'].fillna('')
        df_category['label'] = df_category['category'] + ' - ' + (df_category['sub_category'])

        df['sub_category'] = df['sub_category'].fillna('')
        df['label'] = df['category'] + ' - ' + (df['sub_category'])
        dx = df[['crimeaditionalinfo', 'label']]
        dx.columns = ['text', 'label']
        return dx, df_category[['label']]

    print('Transforming Train DF')
    df_train, df_category = update_dataframe(df_train)
    print('Transforming Test DF')
    df_test, _ = update_dataframe(df_test)

    id2label = df_category[['label']].to_dict()['label']
    label2id = {id: label for label, id in id2label.items()}
    print(label2id)
    print('-----')
    print(id2label)
    def load_dataset(df, label2id, is_train = True) -> Dataset:
        """Load dataset."""

        df["label"] = df["label"].astype(str)
        df["label"] = df["label"].map(
                label2id
            )


        return df


    train_dataset_df = load_dataset(df_train, label2id)
    test_dataset_df = load_dataset(df_test, label2id, False)


    # Define Dataset class (unchanged from your code)
    class Triage(Dataset):
        def __init__(self, dataframe, tokenizer, max_len):
            self.len = len(dataframe)
            self.data = dataframe
            self.tokenizer = tokenizer
            self.max_len = max_len

        def __getitem__(self, index):
            title = str(self.data.text[index])
            title = " ".join(title.split())
            inputs = self.tokenizer.encode_plus(
                title,
                None,
                add_special_tokens=True,
                max_length=self.max_len,
                pad_to_max_length=True,
                return_token_type_ids=True,
                truncation=True
            )
            ids = inputs['input_ids']
            mask = inputs['attention_mask']
            # dx = self.data.label[index]


            return {
                'ids': torch.tensor(ids, dtype=torch.long),
                'mask': torch.tensor(mask, dtype=torch.long),
                'targets': torch.tensor(int(self.data.label[index]), dtype=torch.long)
            }

        def __len__(self):
            return self.len

    training_set = Triage(train_dataset_df, tokenizer, MAX_LEN)
    testing_set = Triage(test_dataset_df, tokenizer, MAX_LEN)

    train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

    test_params = {'batch_size': VALID_BATCH_SIZE,
                    'shuffle': True,
                    'num_workers': 0
                    }

    training_loader = DataLoader(training_set, **train_params)
    # testing_loader = DataLoader(testing_set, **test_params)

    def calcuate_accu(big_idx, targets):
        n_correct = (big_idx==targets).sum().item()
        return n_correct

    # Define training logic (unchanged from your code)
    def train(epoch):
        tr_loss = 0
        n_correct = 0
        nb_tr_steps = 0
        nb_tr_examples = 0
        model.train()
        print("Total Size: " + str(len(training_loader)))
        for idd,data in (enumerate(training_loader, 0)):
            print(str(idd), end=' ')
            ids = data['ids'].to(device, dtype = torch.long)
            mask = data['mask'].to(device, dtype = torch.long)
            targets = data['targets'].to(device, dtype = torch.long)

            outputs = model(ids, mask)
            loss = loss_function(outputs, targets)
            tr_loss += loss.item()
            big_val, big_idx = torch.max(outputs.data, dim=1)
            n_correct += calcuate_accu(big_idx, targets)

            nb_tr_steps += 1
            nb_tr_examples+=targets.size(0)

            if idd%500==0:
                loss_step = tr_loss/nb_tr_steps
                accu_step = (n_correct*100)/nb_tr_examples
                print(f"Training Loss per 500 steps: {loss_step}")
                print(f"Training Accuracy per 500 steps: {accu_step}")

            optimizer.zero_grad()
            loss.backward()
            # # When using GPU
            optimizer.step()

        print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
        epoch_loss = tr_loss/nb_tr_steps
        epoch_accu = (n_correct*100)/nb_tr_examples
        print(f"Training Loss Epoch: {epoch_loss}")
        print(f"Training Accuracy Epoch: {epoch_accu}")
        torch.save(model.state_dict(), MODEL_DIR/ f"model{epoch}.bin")
    

        return

    # Model definition
    class DistillBERTClass(torch.nn.Module):
        def __init__(self):
            super(DistillBERTClass, self).__init__()
            self.l1 = DistilBertModel.from_pretrained("distilbert-base-uncased")
            self.pre_classifier = torch.nn.Linear(768, 128)
            self.dropout = torch.nn.Dropout(0.2)
            self.classifier = torch.nn.Linear(128, len(label2id))

        def forward(self, input_ids, attention_mask):
            output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
            hidden_state = output_1[0]
            pooler = hidden_state[:, 0]
            pooler = self.pre_classifier(pooler)
            pooler = torch.nn.ReLU()(pooler)
            pooler = self.dropout(pooler)
            output = self.classifier(pooler)
            return output

    from torch import cuda
    device = 'cuda' if cuda.is_available() else 'cpu'
    print(device)

    # Initialize model and train
    model = DistillBERTClass()
    model.to(device)

    # Creating the loss function and optimizer
    loss_function = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)

    # Fine-tune model
    for epoch in tqdm(range(EPOCHS)):
        train(epoch)

    # Save model and tokenizer to the mounted directory
    tokenizer.save_vocabulary(MOUNT_DIR)

    print("Model and vocabulary saved!")

@app.local_entrypoint()
def main():
    train_model.remote()
    print("Training complete. Check your local mounted directory for artifacts.")

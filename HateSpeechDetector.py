from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from sklearn.metrics import accuracy_score, classification_report
import torch
import numpy as np
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import os

class BertClassifierWrapper:
    def __init__(self, epochs=3, batch_size=16, lr=1e-5, max_length = 128, model_dir = "saved_model"):
        
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.max_length = max_length
        self.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
        self.model_dir = model_dir

        # Cargar el tokenizer y el modelo
        
        print("Initializing new model...")
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=2)

        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.lr, eps=1e-8)

    def encode(self, texts):
        # Tokenización eficiente
        return self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)

    def create_data_loader(self, texts, labels, type):
        # Tokenizar textos y crear tensores
        encodings = self.encode(texts)

        input_ids = encodings['input_ids'].to(self.device)
        attention_mask = encodings['attention_mask'].to(self.device)
        labels = torch.tensor(labels).to(self.device)

        dataset = TensorDataset(input_ids, attention_mask, labels)

        sampler = RandomSampler(dataset) if type == "train" else RandomSampler(dataset)

        return DataLoader(dataset, batch_size=self.batch_size, sampler=sampler)
    
    def flat_accuracy(self, preds, labels):
        pred_flat =np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat)/len(labels_flat)

    def train(self, train_texts, train_labels, val_texts, val_labels):

        if os.path.exists(self.model_dir):
            print(f"Loading model from {self.model_dir}...")
            self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_dir)
            self.model = DistilBertForSequenceClassification.from_pretrained(self.model_dir)
            self.model.to(self.device)
            return

        # Crear DataLoader para entrenamiento y validación
        train_loader = self.create_data_loader(train_texts, train_labels, type="train")
        val_loader = self.create_data_loader(val_texts, val_labels, type="validation")
        print(train_loader , val_loader)
        total_loss = 0
        total_steps = len(train_loader) * self.epochs

        scheduler = get_linear_schedule_with_warmup(self.optimizer,
                                                    num_warmup_steps=0,
                                                    num_training_steps=total_steps)
        loss_values = []
        for epoch in tqdm(range(0, self.epochs), desc = "Training"):
            
            gold_labels = []

            predicted_labels = []
  
            print(len(train_texts))
            print(len(train_loader))
            self.model.train()
            final_score = ""

            for step, batch in tqdm(enumerate(train_loader), desc="Batch"):
                b_input_ids = batch[0].to(self.device)
                b_input_mask = batch[1].to(self.device)
                b_labels = batch[2].to(self.device)

                self.model.zero_grad()

                outputs = self.model(b_input_ids,
                                attention_mask = b_input_mask,
                                labels = b_labels)
                
                loss = outputs[0]

                total_loss += loss.item()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm = 1.0)

                self.optimizer.step()
                scheduler.step()

            avg_train_loss = total_loss/len(train_loader)

            loss_values.append(avg_train_loss)

            self.model.eval()

            eval_accuracy = 0

            for batch in val_loader:
                
                batch = tuple(t.to(self.device) for t in batch)
                b_input_ids, b_input_mask, b_labels = batch

                with torch.no_grad():
                    outputs = self.model(b_input_ids,
                                         attention_mask = b_input_mask)
                
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                labels_ids= b_labels.to('cpu').numpy()

                #calculate Accuracy
                tmp_eval_accuracy = self.flat_accuracy(logits, labels_ids)

                eval_accuracy += tmp_eval_accuracy

                pred_flat = np.argmax(logits, axis=1).flatten()

                labels_flat = labels_ids.flatten()

                gold_labels.extend(labels_flat)

                predicted_labels.extend(pred_flat)

                final_score = classification_report(labels_flat, pred_flat, digits=4)

                #print(classification_report(labels_flat, pred_flat, digits=4))
            print(final_score)
            print(eval_accuracy)

        self.save_model()

        return

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        print(f"Saving model to {self.model_dir}...")
        self.model.save_pretrained(self.model_dir)
        self.tokenizer.save_pretrained(self.model_dir)


    def predict(self, texts):
      self.model.eval()  
      all_preds = []  
      
      for text in texts:
          
          encodings = self.encode([text])
          input_ids = encodings['input_ids'].to(self.device)
          attention_mask = encodings['attention_mask'].to(self.device)

          
          with torch.no_grad():
              outputs = self.model(input_ids, attention_mask=attention_mask)
              logits = outputs.logits
              preds = torch.argmax(logits, dim=-1)
              all_preds.append(preds.cpu().item())  
      return np.array(all_preds)
    
import pandas as pd
from sklearn.model_selection import train_test_split


class HateBert:
    def __init__(self):
        self.model = BertClassifierWrapper(epochs=1, batch_size=32, model_dir="hatebert_model")

        df = pd.read_csv('cleaned_data.csv')
        df.dropna(inplace=True)

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            df['tweet'], df['class'], test_size=0.2, random_state=42
        )

        print("getting ready")
        self.model.train(train_texts.tolist(), train_labels.tolist(), test_texts.tolist(), test_labels.tolist())
        print("Finish")

        return
    def prefict(self, texts):
        return self.model.predict(texts)
        

    

if __name__ == "__main__":
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = HateBert()

    with open("retrain_reddit_abuse_test.txt", 'r') as f:
        lines = f.readlines()
        ans = model.prefict(lines)
        bad = sum(ans)
        good = len(ans) - bad
        print(f"good {good} bad {bad}")
        print(f"good {good/len(ans)} bad {bad/len(ans)}")
        
   # ans = model.prefict(["I hate you so much", "I love you" , "Why you got to be like that?", "Lets play something"])
    #print(ans)

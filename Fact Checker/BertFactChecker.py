from transformers import BertTokenizer, BertForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.model_selection import train_test_split
import torch

class FactCheckerBERT:
    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

    def get_dataset(self):
        print("Loading the data")
        dataset = load_dataset('fever', 'v1.0')
        train_data = dataset["train"]
        self.training_input = self.prepare_data(train_data)
        test_data = dataset["paper_test"]
        self.test_inputs = self.prepare_data(test_data)
        return

    def prepare_data(self, data):
        print("Preparing Data")
        claims = [item['claim'] for item in data]
        evidences = [item['evidence_wiki_url'] for item in data]  # Using the evidence URL as a placeholder for now
        
        # Convert the label strings to integers for training. Typically: SUPPORTS=0, REFUTES=1, NOT ENOUGH INFO=2
        label_mapping = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}
        labels = [label_mapping[item['label']] if item['label'] in label_mapping else -1 for item in data]

        # Tokenize claims and evidences together
        inputs = self.tokenizer(claims, evidences, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs["labels"] = torch.tensor(labels)
        print("Finished Preperation")
        return inputs

    def train(self, epochs=3):
        #self.training_input
        #claims = data["claim"].tolist()
        #evidences = data["evidence"].tolist()
        #labels = data["label"].tolist()
        print("Started Training the Model")
        inputs = self.training_input#self.prepare_data(claims, evidences, labels)

        # Split the dataset into training and validation sets
        dataset = torch.utils.data.TensorDataset(inputs["input_ids"], inputs["attention_mask"], inputs["labels"])
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

        train_dataloader = torch.utils.data.DataLoader(train_dataset, shuffle=True, batch_size=16)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=16)

        # Training setup: optimizer, scheduler, device
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        total_steps = len(train_dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)

        # Training loop
        for epoch in range(epochs):
            self.model.train()
            for batch in train_dataloader:
                self.model.zero_grad()
                inputs = batch[0].to(device)
                masks = batch[1].to(device)
                labels = batch[2].to(device)
                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Validation
            self.model.eval()
            val_loss = 0
            for batch in val_dataloader:
                inputs = batch[0].to(device)
                masks = batch[1].to(device)
                labels = batch[2].to(device)
                with torch.no_grad():
                    outputs = self.model(inputs, attention_mask=masks, labels=labels)
                val_loss += outputs.loss.item()
            print(f"Epoch: {epoch}, Val Loss: {val_loss/len(val_dataloader)}")

    def validate(self, claim, evidence):
        self.model.eval()
        inputs = self.prepare_data([claim], [evidence], [0])  # label is a dummy and doesn't matter
        with torch.no_grad():
            outputs = self.model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        return "True" if torch.argmax(probs) == 1 else "False"
    def evaluate(self):
        """Evaluate the model's performance on a given dataloader (usually validation or test set)."""

        test_dataset = torch.utils.data.TensorDataset(self.test_inputs["input_ids"], self.test_inputs["attention_mask"], self.test_inputs["labels"])
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

        self.model.eval()
        
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        for batch in test_dataloader:
            inputs = batch[0].to(device)
            masks = batch[1].to(device)
            labels = batch[2].to(device)
            
            with torch.no_grad():
                outputs = self.model(inputs, attention_mask=masks, labels=labels)
                
            # Accumulate the validation loss.
            total_eval_loss += outputs.loss.item()

            # Move logits and labels to CPU
            logits = outputs.logits.detach().cpu().numpy()
            label_ids = labels.to('cpu').numpy()

            # Calculate the accuracy for this batch and accumulate it over all batches.
            total_eval_accuracy += self.flat_accuracy(logits, label_ids)
        
        avg_val_accuracy = total_eval_accuracy / len(test_dataloader)
        avg_val_loss = total_eval_loss / len(test_dataloader)

        return avg_val_accuracy, avg_val_loss

    def flat_accuracy(self, preds, labels):
        """Helper function to compute the accuracy."""
        pred_flat = np.argmax(preds, axis=1).flatten()
        labels_flat = labels.flatten()
        return np.sum(pred_flat == labels_flat) / len(labels_flat)


    def predict(self, claim, evidence):
        inputs = self.tokenizer(claim, evidence, return_tensors="pt", truncation=True)
        outputs = self.model(**inputs)
        probs = outputs.logits.softmax(dim=1)  # Convert logits to probabilities
        return probs[0][1].item()  # Return the probability of the claim being true
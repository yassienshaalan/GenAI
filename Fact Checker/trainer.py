from BertFactChecker import *


bert_checker = FactCheckerBERT()
bert_checker.get_dataset()
bert_checker.train()
test_accuracy, test_loss = bert_checker.evaluate()
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")
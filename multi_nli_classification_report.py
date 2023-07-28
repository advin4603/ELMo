from sentence_classifier import LitSentenceClassifier
from train_multinli import HIDDEN_STATE_SIZE, LSTM_CLASSIFIER_LAYERS, DECODER_LAYERS, embeddings, \
    test_dataset
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

classifier = LitSentenceClassifier.load_from_checkpoint("multinli_classifier.ckpt", classes=3,
                                                        embeddings_matrix=embeddings,
                                                        forward_lm_checkpoint_path="forward_multinli.ckpt",
                                                        backward_lm_checkpoint_path="backward_multinli.ckpt", stacks=2,
                                                        hidden_state_size=HIDDEN_STATE_SIZE,
                                                        lstm_classifier_layers=LSTM_CLASSIFIER_LAYERS,
                                                        decoder_layers=DECODER_LAYERS, ).sentence_classifier

classifier.eval()
y_test, y_pred = [], []
for i in range(len(test_dataset)):
    x, y = test_dataset[i]
    pred = classifier(x).argmax(0)
    y_test.append(y)
    y_pred.append(pred)

print(classification_report(y_test, y_pred))

labels = ["entailment", "neutral", "contradiction"]

cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
color = 'white'
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot()
plt.show()

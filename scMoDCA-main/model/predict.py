import torch
import torch.nn as nn


def classify_cells(model, dl, real_label):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.eval()

    classified_label = []
    groundtruth_label = []
    prob = []

    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        for i, batch_sample in enumerate(dl):
            x = batch_sample['data'].to(device).float()
            x = torch.reshape(x, (x.size(0), -1))
            test_label = batch_sample['label']

            outputs = model(x)
            vmf_logits = outputs[-1]

            probs_full = softmax(vmf_logits)
            max_values, max_indices = torch.max(probs_full, dim=1)

            max_indices = max_indices.cpu().numpy()
            max_values = max_values.cpu().numpy()
            test_label = test_label.cpu().numpy()

            for j in range(x.size(0)):
                classified_label.append(real_label[max_indices[j]])
                groundtruth_label.append(real_label[test_label[j]])
                prob.append(max_values[j])

    return classified_label, groundtruth_label, prob

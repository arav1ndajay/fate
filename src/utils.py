from torch.utils.data import TensorDataset

class CustomDataset(TensorDataset):
    def __init__(self, input_ids, attention_masks, labels, normal_idx=None, outlier_idx=None):
        super().__init__(input_ids, attention_masks, labels)
        self.normal_idx = normal_idx
        self.outlier_idx = outlier_idx

# load sentences from files
def get_sentences(file_path):
    with open(file_path, encoding='utf-8') as f:
        sentences = [
        line 
        for line in f.read().splitlines()
        if (len(line) > 0 and not line.isspace())
        ]
    
    return sentences
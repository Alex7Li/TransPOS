import torch
from torch.distributions import Categorical
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

def hardToSoftLabel(
    hard_label: torch.Tensor, n_classes: int
):
    BS, SEQ_LEN = hard_label.shape
    # Ensure the ignored indicies don't crash the code.
    hard_label = torch.clone(hard_label)
    hard_label[hard_label == -100] += 100
    one_hot = F.one_hot(hard_label, n_classes)
    # Scale one_hot up because weight initalization usually assumes random N(0, 1) distribution for the previous layer
    one_hot = one_hot.float().to(device) * torch.sqrt(torch.tensor(n_classes, dtype=torch.float32, device=hard_label.device))
    # one_hot -= torch.unsqueeze(torch.mean(one_hot, dim=2), 2)
    return one_hot

def softToHardLabel(label):
    assert len(label.shape) == 3
    # label is of shape [batch, L, self.n_y_labels]
    label = Categorical(logits=label).sample()
    return label
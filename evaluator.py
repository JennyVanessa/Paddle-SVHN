from dataset import Dataset
import paddle
from paddle.vision import transforms

def label_convert(label_tensor,len_label):
        for i in range(0,len_label):
            if label_tensor[i]==10:
                label_tensor[i]=10-label_tensor[i]
            else:
                label_tensor[i]=label_tensor[i]              
        return label_tensor

class Evaluator(object):
    def __init__(self, path_to_lmdb_dir):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self._loader = paddle.io.DataLoader(Dataset(path_to_lmdb_dir, transform), batch_size=128, shuffle=False)

    def evaluate(self, model):
        num_correct = 0
        needs_include_length = False    

        with paddle.no_grad():
            for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):               
                images, length_labels, digits_labels = images, length_labels, [digit_label for digit_label in digits_labels]
                model.eval()
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model(images)
    
                length_prediction = paddle.argmax(length_logits,1)
                digit1_prediction = paddle.argmax(digit1_logits,1)
                digit2_prediction = paddle.argmax(digit2_logits,1)
                digit3_prediction = paddle.argmax(digit3_logits,1)
                digit4_prediction = paddle.argmax(digit4_logits,1)
                digit5_prediction = paddle.argmax(digit5_logits,1)

                if needs_include_length:
                    num_correct += (length_prediction.equal(length_labels) &
                                    digit1_prediction.equal(digits_labels[0]) &
                                    digit2_prediction.equal(digits_labels[1]) &
                                    digit3_prediction.equal(digits_labels[2]) &
                                    digit4_prediction.equal(digits_labels[3]) &
                                    digit5_prediction.equal(digits_labels[4])).sum()
                else:
                    num_correct += (digit1_prediction.equal(digits_labels[0])).logical_and(
                                    digit2_prediction.equal(digits_labels[1])).logical_and(
                                    digit3_prediction.equal(digits_labels[2])).logical_and(
                                    digit4_prediction.equal(digits_labels[3])).logical_and(
                                    digit5_prediction.equal(digits_labels[4])).astype('int64').sum()
                 
        accuracy = num_correct / len(self._loader.dataset)
        return accuracy


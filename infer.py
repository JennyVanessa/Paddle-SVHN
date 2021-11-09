import argparse
import paddle

from PIL import Image
from paddle.vision import transforms

from model import Model

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to checkpoint, e.g. ./logs/model-100.pdparams')
parser.add_argument('input', type=str, help='path to input image')


def _infer(path_to_checkpoint_file, path_to_input_image):
    model = Model()
    param_state_dict=paddle.load(path_to_checkpoint_file)
    model.set_dict(param_state_dict)

    with paddle.no_grad():
        transform = transforms.Compose([
            transforms.Resize([64, 64]),
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        image = Image.open(path_to_input_image)
        image = image.convert('RGB')
        image = transform(image)
        images = image.unsqueeze(axis=0)
        model.eval()
        length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model(images)

        length_prediction = paddle.argmax(length_logits,1)
        digit1_prediction = paddle.argmax(digit1_logits,1)
        digit2_prediction = paddle.argmax(digit2_logits,1)
        digit3_prediction = paddle.argmax(digit3_logits,1)
        digit4_prediction = paddle.argmax(digit4_logits,1)
        digit5_prediction = paddle.argmax(digit5_logits,1)

        print('length:', length_prediction.item())
        print('digits:', digit1_prediction.item(), digit2_prediction.item(), digit3_prediction.item(), digit4_prediction.item(), digit5_prediction.item())


def main(args):
    path_to_checkpoint_file = args.checkpoint
    path_to_input_image = args.input

    _infer(path_to_checkpoint_file, path_to_input_image)


if __name__ == '__main__':
    main(parser.parse_args())
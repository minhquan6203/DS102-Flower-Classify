import argparse
from get_config import get_config
from cnn_task import CNN_Classify_task
from svm_task import SVM_Classify_task
parser = argparse.ArgumentParser()
parser.add_argument("--config-file", type=str, required=True)


args = parser.parse_args()

config = get_config(args.config_file)
type=get_config(args.type_model)
if type=="CNN":
    task=CNN_Classify_task(config)
    task.training() #traning, khi nào muốn predict thì cmt lại
    task.evaluate() #đánh giá trên test data
if type=="SVM":
    task=SVM_Classify_task(config)
    task.training() #traning, khi nào muốn predict thì cmt lại
    task.evaluate() #đánh giá trên test data
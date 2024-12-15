import argparse
import configparser
import json
import os

# our source code
from chatclassifier.classifier import ChatClassifier


# READ ARGUMENTS
# we only support reading a config file
# which defaults to config/config.ini
parser = argparse.ArgumentParser()
parser.add_argument(
	"--config_file",
	"--config-file", 
	default = "config/config.ini",
	help="path to the config file"
)

args = parser.parse_args()

# READ CONFIG
config = configparser.ConfigParser()
config.read(args.config_file)

# paths in the config file are defined relative to this script
# get the absolute paths
def get_abs_path(path):
	return os.path.join(
		os.path.dirname(os.path.abspath(__file__)), path)

config['HuggingFace']['LOG_PATH'] = \
	get_abs_path(config['HuggingFace']['LOG_PATH'])
	
config['HuggingFace']['MODEL_PATH'] = \
	get_abs_path(config['HuggingFace']['RESOURCE_PATH'])

# CREATE A CLASSIFIER FROM CONFIG
hf_classifier = ChatClassifier(config['HuggingFace'])


# READ CHAT DATA
# in production this could be an endpoint serving real-time requests
data_path = config['Input']['DATA_PATH']
conversations = [
	j for j in os.listdir(data_path) if os.path.splitext(j)[-1]==".json"
	]

for convo in conversations:
	print(f'# LABELS for {convo}')
	with open(os.path.join(data_path,convo)) as cfile:
		message_history = json.load(cfile)
	
	for message in message_history:
		if message['role'] == 'assistant':
			continue
		
		print(f"MESSAGE: {message['content'][:50]}")
		pred_s = hf_classifier.predict("SENTIMENT", message['content'])
		print(f"SENTIMENT: {pred_s}")
		pred_i = hf_classifier.predict("INTENT", message['content'])
		print(f"INTENT: {pred_i}")

		
		print(message['role'])


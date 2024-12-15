
import os
import json
from huggingface_hub import snapshot_download
from transformers import pipeline
import logging
import datetime
import json
import numpy as np


logger = logging.getLogger("Intent Classifier")

class ChatClassifier:
	
	def __init__(self, config):
		
		log_file = os.path.join(
			config['LOG_PATH'],
			f"log_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.log"
		)
		logging.basicConfig(
			filename=log_file, 
			encoding='utf-8', 
	        format='%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
			level=logging.INFO
		)
		

		logger.info('downloading resources if they do not exist')
		
		self.repo_id = config['REPO_ID']
		self.model_path = config['MODEL_PATH']
		snapshot_download(
			repo_id=self.repo_id,
			local_dir=self.model_path)
		
		logger.info('loading zero-shot model')
		self.classifier = pipeline(
			"zero-shot-classification",
			model=self.model_path
		)
		
		# We support different types of predictions
		# each defined with a list of labels and defining text
		self.prediction_types = json.loads(config['PREDICTION_TYPES'])
		
		# AFAIR huggingface could return results in a different order 
		# than provided one, so we build an inverse map to ensure
		# correct mapping
		self.inverse = \
			{pred: {text: label for label,text in definition.items()} 
				for pred, definition in self.prediction_types.items()}
		

	def predict(self, prediction_type, input_text):
		if prediction_type not in self.prediction_types:
			logger.error(
				f'Prediction type {prediction_type} is not configured')
			return 
		
		try:
			logger.info(f'Predicting {prediction_type} for')
			logger.info(f'\t {input_text}')
			labels = list(self.prediction_types[prediction_type].values())
			results = self.classifier(input_text, labels)
			pred = results['labels'][np.argmax(results['scores'])]
			final_pred = self.inverse[prediction_type][pred]
			logger.info(f'Prediction: {final_pred}')
			return final_pred
		except:
			logger.error(
				f'Could not generate a prediction!')
			return			

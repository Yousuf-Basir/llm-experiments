from transformers import pipeline
import time

start_time = time.time()
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love my life but sometimes life is hard")
print(result)
print("Execution time: %s seconds" % (time.time() - start_time))

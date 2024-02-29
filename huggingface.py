from transformers import pipeline
import time

url = "https://www.kkbox.com/sg/en/song/8kFm4nAIsZ01P7mWgu"

start_time = time.time()
classifier = pipeline("sentiment-analysis", model="distilbert/distilbert-base-uncased-finetuned-sst-2-english")
result = classifier("I love my life but sometimes life is hard")
print(result)
print("Execution time: %s seconds" % (time.time() - start_time))

from llama_index.llms.ollama import Ollama
from llama_index.core import Settings, SummaryIndex
from llama_index.readers.web import SimpleWebPageReader
import time

Settings.llm = Ollama(
    model="tinyllama",
    base_url="http://89.116.167.82:11434"
)
url = "https://www.kkbox.com/sg/en/song/8kFm4nAIsZ01P7mWgu"
documents = SimpleWebPageReader(html_to_text=True).load_data([url])

start_time = time.time()
index = SummaryIndex.from_documents(documents)

query_engine = index.as_query_engine(streaming=True, similarity_top_k=1)
streaming_response = query_engine.query("What is the song about?")

streaming_response.print_response_stream()
print("\nExecution time: %s seconds" % (time.time() - start_time))
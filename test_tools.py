from tools import wikipedia_search, visit_webpage, run_python, web_search

print(wikipedia_search.invoke({"query": "Giganotosaurus"})[:300], "\n---")
print(visit_webpage.invoke({"url": "https://en.wikipedia.org/wiki/Giganotosaurus"})[:300], "\n---")
print(run_python.invoke({"code": "print(sum(range(10)))"}), "\n---")
print(web_search.invoke({"query": "Mercedes Sosa discography"})[:300])
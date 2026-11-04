from tools import run_python, visit_webpage, web_search, wikipedia_search

print(wikipedia_search.invoke({"query": "Giganotosaurus"})[:300], "\n---")
print(visit_webpage.invoke({"url": "https://en.wikipedia.org/wiki/Giganotosaurus"})[:300], "\n---")
print(run_python.invoke({"code": "print(sum(range(10)))"}), "\n---")
print(web_search.invoke({"query": "Mercedes Sosa discography"})[:300])

import wikipedia
from tqdm import tqdm
import db
import sys
from tqdm import tqdm
sys.tracebacklimit = 0
wikipedia.set_lang("en")
db = db.db("Wikipedia.db")

def get_random_pages(num:int):
    i = 0
    while i < num:
        try:
            page = wikipedia.random(1)
            summary = wikipedia.summary(page, sentences=5)
            db.append(summary)
            i += 1
            if i % 100:
                print(i)
        except wikipedia.exceptions.PageError as e:
            print(n, e.title)

        except wikipedia.exceptions.DisambiguationError as e:
            print(n, f"disambiguation error with \"{page}\"")

        except:
            pass


if __name__ == '__main__':
    n = input("How many pages? > ")
    get_random_pages(int(n))
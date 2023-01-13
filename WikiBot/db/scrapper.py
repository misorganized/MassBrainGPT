import wikipedia
from tqdm import tqdm
import db
import sys
from tqdm import tqdm
sys.tracebacklimit = 0
wikipedia.set_lang("en")
db = db.db("Local.db")

def get_random_pages(num:int):
    for n in tqdm(range(num)):
        try:
            page = wikipedia.random(1)
            summary = wikipedia.summary(page, sentences=5)
            db.append(summary)
        except wikipedia.exceptions.PageError as e:
            # print(n, e)
            pass

        except wikipedia.exceptions.DisambiguationError as e:
            # print(n, f"disambiguation error with \"{page}\"")
            pass

        except:
            pass

if __name__ == '__main__':
    n = input("How many pages? > ")
    get_random_pages(int(n))
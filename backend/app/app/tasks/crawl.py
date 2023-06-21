import re
from playwright.sync_api import Playwright, sync_playwright
from playwright_stealth import stealth_sync
import random
import io
from bs4 import BeautifulSoup
import httpx
import fitz
from app.db.session import SessionLocal
from app.schemas.source import SourceUpdate
from app.schemas.job import JobUpdate
from app import crud
from celery import shared_task
import logging


# def batch(iterable, n=0):
#     l = len(iterable)
#     for ndx in range(-1, l, n):
#         yield iterable[ndx : min(ndx + n, l)]


def fetch_pdf(url):
    try:
        response = httpx.get(url)
        return response

    except:
        return None


def html2text(html):
    soup = BeautifulSoup(html, "html.parser")
    results = []
    data = []
    if soup.title is not None:
        data.append(soup.title.string)
    tags = ["p"]
    data.extend(soup.find_all(tags))

    for t in data:
        if t:
            t = re.sub(r"(\s\s+|\t+|\n+)", "", t.get_text()).strip()
            if len(t) > 20:
                results.append(t)
    return " ".join(results)


def pdf2text(httpx_PDF_response):
    with io.BytesIO(httpx_PDF_response.content) as pdf_file:
        text = ""
        with fitz.open(filetype="pdf", stream=pdf_file.read()) as doc:
            for p in doc:
                text = text + " " + str(p.get_text())
    lines = []
    for line in text.split("\n"):
        if line != "":
            lines.append(line.strip())
    return " ".join(lines)


def urls2text(urls):
   
    resp_dict={}
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            "/tmp", headless=True, timeout=5000
        )
        page = context.new_page()
        stealth_sync(page)
        for url in urls:
            try:
                # try playwright stealth for html
                response = page.goto(url, wait_until="domcontentloaded")
                if "html" in response.headers.get("content-type"):
                    resp_dict[url] = html2text(page.content())
                    continue
            except:
                pass #fallback to httpx
            
            try:
                httpx_response = httpx.get(url)
                if "html" in httpx_response.headers["content-type"]:
                    resp_dict[url] = html2text(httpx_response.text)
                    continue
                else:
                    # content-type is not always set corrently on response.headers, so we assume PDF content and try to extract text.
                    resp_dict[url] = pdf2text(httpx_response)
                    continue
            
            except Exception as e:
                logging.info(f"no txt retrieved {url}")
                resp_dict[url] = None
                continue
        context.close()
        return resp_dict


@shared_task(
    name="ingress:job_crawl",
    bind=True,
    default_retry_delay=30,
    max_retries=3,
    soft_time_limit=10000,
)
def job_crawl(self, job_id):
    db = SessionLocal()
    job = crud.job.get(db=db, id=job_id)
    try:
        sources = [source for source in job.sources if not source.text]
        urls = [source.url for source in sources]
        url2text = urls2text(urls)
        for source in sources:
            source_update = SourceUpdate(text=url2text[source.url])
            crud.source.update(db=db, db_obj=source, obj_in=source_update)

        db.close()
        return job_id
    except Exception as e:
        logging.info(f"job {job.id} status = failed")
        job_update = JobUpdate(status="failed")
        crud.job.update(db=db, db_obj = job, obj_in=job_update)
        


def main():
    EXT_PATH = r"C:\Users\PvR\Code\Sandbox\playwright\I_dont_care_about_cookies\3.4.1_0"
    USER_DIR = r"C:/Users/PvR/AppData/Local/Google/Chrome/User Data/Profile 2"

    ARGS = [
        "--disable-extensions-except={}".format(EXT_PATH),
        f"--load-extension={EXT_PATH}",
    ]

    URLS = [
        "https://www.assets.signify.com/is/content/Signify/Assets/signify/global/ir/supplement-to-the-2017-sustainability-statements.pdf",
        "http://realpython.com",
        "http://www.heroku.com",
        "http://python-tablib.org",
        "http://httpbin.org",
        "http://fakedomain/",
        "https://www.nu.nl",
    ]

    r = urls2text(URLS, USER_DIR, ARGS)


if __name__ == "__main__":
    main()

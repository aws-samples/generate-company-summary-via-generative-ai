"""
Utility functions.

To run tests: python3 -m doctest utilities.py
"""

import os, re
from typing import Iterable, List, Any, Optional, Tuple
from uuid import uuid4
from io import BytesIO
from pathlib import Path
import xml.etree.ElementTree as ET
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from langchain_community.llms import Bedrock
from langchain_community.document_loaders import AmazonTextractPDFLoader


def google_search(q: str, num_results: int = 5, timeout: int = 20):
    """ return JSON from Google via SERP API """
    print(f"google_search '{q}' num_results={num_results} timeout={timeout}")
    params = dict(
        q=q,
        api_key=os.environ["SERPAPI_API_KEY"],
        engine="google",
        num=num_results
    )
    response = requests.get("https://serpapi.com/search/",
                            params=params, timeout=timeout)
    return response.json()


user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0"


def download_web_page(URL: str, mime_type: str = "text/html",
                      headers: dict = None) -> str:
    print(f"download_web_page {URL}")
    headers = headers or {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Encoding": "gzip, deflate, br",
        "Accept-Language": "en-US,en;q=0.5"
    }
    session = requests.Session()
    try:
        resp = session.get(URL, headers=headers, timeout=30)
    except (requests.exceptions.Timeout, ConnectionError, TimeoutError) as ex:
        print(f"Failed to download {URL}: {ex}")
        return None
    # (Path.cwd() / "raw-download.html").write_text(resp.text)
    if mime_type is None:
        return resp
    elif mime_type not in resp.headers["Content-Type"]:
        return None
    else:
        return resp.text if "text/html" in mime_type else resp.content


def download_web_page_as_text(
    URL: str, s3_client, textract_client,
    bucket_name: str,
    headers: dict = None,
    allow_only_single_page_PDFs: bool = False) -> Optional[str]:
    """
    Return a textual version of this page, whether the page is HTML or PDF.
    """
    resp = download_web_page(URL, mime_type=None, headers=headers)
    if resp and resp.status_code == 200:
        if "text/html" in resp.headers["Content-Type"]:
            html_contents = remove_noise_from_web_page(resp.text)
            # print(f"~#tokens {round(len(html_contents)*(4/3))}")
            # (Path.cwd() / "html-contents.text").write_text(html_contents)
            return html_contents
        elif "application/pdf" in resp.headers["Content-Type"]:
            docs = convert_pdf_to_text(s3_client, textract_client,
                                       bucket_name, resp.content)
            if allow_only_single_page_PDFs and len(docs) > 1:
                print(f"Ignoring multi-page PDF at {URL}")
                return None
            elif len(docs) == 0:
                print(f"Got no documents back from {URL}")
                return None
            else:
                text = convert_textract_output_to_text(docs)
                return text
        else:
            print(f'Ignoring content type: {resp.headers["Content-Type"]}')
            return None
    else:
        print(f"Failed to download from {URL}, response: {resp}")
        return None


def extract_site_from_URL(URL: str) -> str:
    """
    >>> extract_site_from_URL("https://foo.com/bar")
    'foo.com'
    """
    tmp = urlparse(URL)
    return tmp.netloc


def extract_TLD_from_URL(URL: str) -> str:
    """
    >>> extract_TLD_from_URL("https://www.foo.com")
    'foo.com'
    >>> extract_TLD_from_URL("https://www.foo.co.uk")
    'foo.co.uk'

    """
    tmp = urlparse(URL)
    parts = tmp.netloc.split(".")
    if parts[-2] == "co" and len(parts[-1]) == 2:
        # attempt to handle TLDs like "foo.co.uk"
        return ".".join(parts[-3:])
    else:
        return ".".join(parts[-2:])


def remove_noise_from_web_page(html_contents: str) -> str:
    try:
        soup = BeautifulSoup(html_contents, features="html.parser")
        [body] = soup.findAll("body")
        return soup.get_text()
    except Exception as ex:
        print(f"Caught in remove_noise_from_web_page: {ex}")
        return ""


def extract_sets_of_tags(input_text: str) -> List[dict]:
    """
    Given a text response from a model, parse out any Xml-ish tags
    and return a sequence of dictionarys of them. Handles consecutive
    sets of tags.

    >>> extract_sets_of_tags('''\
    ... <a>1</a>
    ... <b>2</b>
    ... <c>3</c>
    ... <a>4</a>
    ... <b>5</b>
    ... <c>6</c>
    ... ''')
    [{'a': '1', 'b': '2', 'c': '3'}, {'a': '4', 'b': '5', 'c': '6'}]

    """
    result = []
    root = ET.fromstring(f"<root>{escape_text_for_xml(input_text)}</root>")
    current = {}
    for child in root:
        if child.tag in current.keys():
            result.append(current)
            current = {}
        current[child.tag] = child.text
    if current:
        result.append(current)
    return result


def escape_text_for_xml(text: str) -> str:
    return text.replace("&", "&amp;")


def extract_multiple_tags(s: str) -> dict:
    """
    @todo: support nested tags.

    >>> extract_multiple_tags('''\
    ... <a>1</a>
    ... <b>2</b>
    ... <c>3</c>
    ... ''')
    {'a': '1', 'b': '2', 'c': '3'}

    """
    result = {}
    patn = r"<(?P<tag_name>[a-zA-Z0-9_-]+)>(.*)</(?P=tag_name)>"
    regex = re.compile(patn, re.DOTALL)
    while True:
        m = regex.search(s)
        if m:
            tag_name, contents = m.groups()
            result[tag_name] = contents.strip()
            _start_idx, stop_idx = m.span()
            s = s[stop_idx:]
        else:
            return result


def extract_tag(response: str, name: str, greedy: bool = True,
                opening_optional: bool = False) -> Tuple[str, int]:
    """
    >>> extract_tag("foo <a>baz</a> bar", "a")
    ('baz', 10)

    >>> extract_tag("baz</a> bar", "a", opening_optional=True)
    ('baz', 3)

    """
    if opening_optional:
        patn = f"\\s*(.*)</{name}>" if greedy else f"\\s*(.*?)</{name}>"
        match = re.match(patn, response, re.DOTALL)
        if match:
            return match.group(1).strip(), match.end(1)
    else:
        patn = f"<{name}>(.*)</{name}>" if greedy else\
               f"<{name}>(.*?)</{name}>"
        match = re.search(patn, response, re.DOTALL)
        if match:
            return match.group(1).strip(), match.end(1)
    print(f"Couldn't find tag {name} in <<<{response}>>>")
    return "", -1


# log_file = Path.cwd() / "prompt-log.txt" # uncomment to log prompts
log_file = None


class WrappedBedrock (Bedrock):

    """
    A plug-in replacement for the Bedrock class that automatically
    takes care of adding "Human: ... Assistant: ..." for Claude.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager=None,
        **kwargs: Any,
    ) -> str:
        if log_file:
            with log_file.open("a") as f:
                f.write(f"-----Prompt:------\n{prompt}\n"
                        "-----end prompt-----\n")

        if not re.match(r"^[\n]{2,}Human:", prompt):
            if prompt.startswith("Human:"):
                prompt = "\n\n" + prompt
            else:
                prompt = f"""\n\nHuman:\n{prompt}\n\nAssistant:"""
        response = super()._call(prompt, stop, run_manager, **kwargs)

        if log_file:
            with log_file.open("a") as f:
                f.write(f"-----Response:------\n{response}"
                        "\n-----end response-----\n")
        return response


def convert_pdf_to_text(s3_client,
                        textract_client,
                        bucket_name: str,
                        pdf_contents: bytes) -> str:
    """
    Given a PDF file (represented as a bytes object), use textract to
    parse out the text. Only works on single-page documents.
    """
    key = f"temp-pdf-files/{uuid4()}.pdf"
    with BytesIO(pdf_contents) as f:
        s3_client.upload_fileobj(f, Bucket=bucket_name, Key=key)
    URL = f"s3://sgh-misc/{key}"
    print(URL)
    loader = AmazonTextractPDFLoader(URL, client=textract_client)
    documents = loader.load()
    return documents


def convert_textract_output_to_text(docs: Iterable) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def escape_HTML_text(text: str) -> str:
    """
    Avoid triggering MathTex within Jupyter notebooks.

    >>> escape_HTML_text("$42.0")
    '&#36;42.0'
    """
    return text.replace("$", "&#36;").replace("<", "&lt;").replace(">", "&gt;")


def same_TLD(netloc1: str, netloc2: str) -> bool:
    """
    Do two URLs come from the same web site, or more precisely, the same
    top level domain? The netloc is as returned by urlparse(). We want
    ir.intelliatx.com and www.intelliatx.com, for ex., to be considered to
    be the same.

    @todo: does not work for country TLDs, like "foo.co.uk"

    >>> same_TLD("ir.intelliatx.com", "www.intelliatx.com")
    True

    """
    TLD1 = netloc1.split(".")[-2:]
    TLD2 = netloc2.split(".")[-2:]
    result = TLD1 == TLD2
    # print(f"same_TLD {netloc1} {netloc2} -> {result}")
    return result


def strip_multiline_whitespace(text: str) -> str:
    lines = text.split("\n")
    lines = [line.strip() for line in lines]
    return "\n".join(lines)


def most_common_elem(l: list):
    """
    Tie-breaking is arbitrary.
    
    >>> most_common_elem([1, 2, 1])
    1

    >>> most_common_elem(["a", "b", "b"])
    'b'
    """
    return max(set(l), key=l.count)

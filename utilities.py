"""
Utility functions.

To run tests: python3 -m doctest utilities.py
"""

import os, re, json
from typing import Iterable, List, Optional, Tuple, Callable, Dict, Union
from uuid import uuid4
from io import BytesIO
import xml.etree.ElementTree as ET
from urllib.parse import urlparse
from urllib3.exceptions import MaxRetryError, TimeoutError
import multiprocessing.dummy
from functools import partial
from enum import Enum
from time import sleep

import jinja2
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import AmazonTextractPDFLoader
from botocore.exceptions import ClientError, ReadTimeoutError
from IPython.display import HTML

JENV = jinja2.Environment(trim_blocks=True, lstrip_blocks=True)


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


USER_AGENT = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:109.0) Gecko/20100101 Firefox/115.0"


def download_web_page(URL: str, mime_type: str = "text/html",
                      headers: dict = None) -> str:
    print(f"download_web_page {URL}")
    headers = headers or {
        "User-Agent": USER_AGENT,
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
    allowable_mime_types = ["text/html", "application/pdf"],
    allow_only_single_page_PDFs: bool = False) -> Optional[str]:
    """
    Return a textual version of this page, whether the page is HTML or PDF.
    """
    resp = download_web_page(URL, mime_type=None, headers=headers)
    if resp and resp.status_code == 200:
        # print(f"headers: {resp.headers}")
        content_type = resp.headers.get("Content-Type", "unknown")
        if allowable_mime_types is not None and \
           not any(mime_type in content_type
                   for mime_type in allowable_mime_types):
            print(f"Discarding {URL} as mime type {content_type} "
                  f"not in the allowed set {allowable_mime_types}")
            return None
        elif "text/html" in content_type:
            html_contents = remove_noise_from_web_page(resp.text)
            return html_contents
        elif "application/pdf" in content_type:
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
            print(f'Ignoring content type: {content_type}')
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
        return body.get_text()
    except ValueError as ex:
        if "not enough values to unpack" in str(ex):
            print(f"Couldn't find <body> tag: {html_contents}")
            return str
        else:
            raise ex
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


def create_bedrock_runner(bedrock_runtime,
                          model_id: str,
                          temperature: float) -> Callable:
    """
    A convenient way to bake parameters like model_id into a callable function.
    """
    return partial(run_bedrock,
                   bedrock_runtime=bedrock_runtime,
                   model_id=model_id,
                   temperature=temperature)


def run_bedrock(prompt: str,
                bedrock_runtime,
                model_id: str,
                temperature: float) -> str:
    try:
        # print(f"prompt: {prompt}")
        response = None
        for iter in range(10):
            try:
                response = bedrock_runtime.invoke_model(
                    modelId=model_id,
                    body=json.dumps(
                        {
                            "anthropic_version": "bedrock-2023-05-31",
                            "max_tokens": 1024,
                            "temperature": temperature,
                            "messages": [
                                {
                                    "role": "user",
                                    "content": [{"type": "text",
                                                 "text": prompt}],
                                }
                            ],
                        }
                    ),
                )
                break
            except (TimeoutError, ReadTimeoutError) as ex:
                print(f"Caught {ex}; sleep & retry #{iter+1}")
                sleep(iter+1)
            except Exception as ex:
                print(f"didn't catch {ex}")
                raise ex
        if response:
            result = json.loads(response.get("body").read())
            output_list = result.get("content", [])
            return "".join(output["text"] for output in output_list
                           if output["type"] == "text")
        else:
            return ""
    except ClientError as err:
        print(f"Error invoking {model_id}:"
              f" {err.response['Error']['Code']}"
              f" {err.response['Error']['Message']}")
        raise err


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


def parallel_map(func: Callable, sequence: Iterable, num_threads: Optional[int] = 10) -> Iterable:
    """

    >>> list(parallel_map(lambda x: x+1, range(10), num_threads=1))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    >>> list(parallel_map(lambda x: x+1, range(10), num_threads=5))
    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    """
    if num_threads == 1:
        # Much easier to test if we turn off concurrency
        return map(func, sequence)
    else:
        pool = multiprocessing.dummy.Pool(num_threads)
        return pool.map(func, sequence)


def search_result_downloader(bucket_name: str,
                             headers: dict,
                             s3_client,
                             textract_client,
                             search_result: dict,
                             allowable_mime_types = ["text/html", "application/pdf"],
                             allow_only_single_page_PDFs: bool = None,
                             cached_contents: Dict[str, str] = None) -> str:
    URL = search_result["link"]
    try:
        if cached_contents and URL in cached_contents:
            content = cached_contents[URL]
        else:
            content = download_web_page_as_text(URL, s3_client, textract_client,
                                                bucket_name, headers=headers,
                                                allowable_mime_types=allowable_mime_types,
                                                allow_only_single_page_PDFs=allow_only_single_page_PDFs)
    except (ConnectionError, MaxRetryError) as ex:
        print(f"Failed to download {URL}: {ex}")
        content = None
    return content


def default_reducer(map_results: Iterable[Union[str, dict]],
                    model_runner: Callable[[str], str]) -> str:
    prompt_template = strip_multiline_whitespace("""\
        Please consider the following snippets:
    
        {% for mr in map_results %}
        <snippet>
        {{ mr }}
        </snippet>
        {% endfor %}
    
        Summarize the above snippets in 5-6 sentences. Don't include any preamble. Include your
        summary in <summary></summary> tags.
        """)
    prompt = JENV.from_string(prompt_template).render(map_results=map_results)
    result = model_runner(prompt)
    tags = extract_multiple_tags(result)
    try:
        return_value = tags["summary"]
    except KeyError:
        return_value = ""
    return return_value


class Mode(Enum):
    MAP_REDUCE = 0
    FIRST_HIT = 1

def WSAG(
    model_runner: Callable[[str], str],
    company_name: str,
    company_url: str,
    web_search_query: str,
    num_threads: int,
    search_result_downloader: Callable[[dict, dict], str] = None,
    search_result_mapper: Callable[[dict, str], Union[str, dict]] = None,
    search_result_filter: Optional[Callable[[int, dict], Tuple[bool, str]]] = None,
    search_result_evaluator: Callable[[dict, Union[str, dict]], bool] = None,
    reducer: Callable[[Iterable[Union[str, dict]]], str] = None,
    num_web_search_results: Optional[int] = 10,
    web_search_timeout: Optional[int] = 30,
    mode: Mode = Mode.MAP_REDUCE,
    result_tag: Optional[str] = "summary"):
    """
    The downloader takes a search result (a dict) and a dict of cached contents
    (URL => str) and returns the downloaded contents (a str)

    The mapper takes a str (typically the output of the downloader) and returns
    another str (typically the result of using an LLM to analyze the input str).

    The filter takes a search result (a dict)
    """
    response_json = google_search(web_search_query,
                                  num_results=num_web_search_results,
                                  timeout=web_search_timeout)
    try:
        organic_results = response_json["organic_results"]
    except KeyError:
        print("Warning: Google returned 0 results for this query")
        organic_results = []

    for result in organic_results:
        print(f"Result: {result['link']}")

    cached_contents = {}
    if search_result_filter is not None:
        results = []
        for idx, result in enumerate(organic_results):
            selected, contents = search_result_filter(idx, result)
            if selected:
                results.append(result)
                cached_contents[result["link"]] = contents
    else:
        results = organic_results

    if mode.value == Mode.MAP_REDUCE.value:
        assert search_result_mapper is not None and search_result_downloader is not None
        if reducer is None:
            reducer = partial(default_reducer, model_runner=model_runner)
        map_results = parallel_map(lambda result: search_result_mapper(
                                                    search_result=result,
                                                    text_contents=search_result_downloader(
                                                        search_result=result,
                                                        cached_contents=cached_contents)),
                                   results, num_threads=num_threads)
        map_results = filter(None, map_results) # remove empty elements
        return reducer(map_results)
    elif mode.value == Mode.FIRST_HIT.value:
        assert search_result_evaluator is not None
        for i, search_result in enumerate(results):
            print(f"Consider #{i:,} {search_result['link']}")
            mapper_result = search_result_mapper(
                              search_result=search_result,
                              text_contents=search_result_downloader(search_result=search_result,
                                                                     cached_contents=cached_contents))
            if search_result_evaluator(search_result,
                                       mapper_result=mapper_result):
                return mapper_result
    else:
        raise Exception("port me!")

def test_wrapper(caller_globals: dict):
    """
    For a given function, foo, that returns an HTML string, create a second
    function, test_foo, that may (if enable_test_harness is on) run
    the function and display the results.
    """
    def test_wrapper2(func):
        def f(*args, **kwargs):
            if caller_globals["enable_test_harness"]:
                return HTML(func(*args, **kwargs))
            else:
                return None
        caller_globals[f"test_{func.__name__}"] = f
        return func
    return test_wrapper2

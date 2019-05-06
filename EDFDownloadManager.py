import os
from pathlib import Path
import re
import requests
import urllib.request
from bs4 import BeautifulSoup

def edf_filefinder(url):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, "html.parser")

    rec_filenames = []
    for link in soup.findAll(re.compile("^a"), href=re.compile("\.rec")):
        rec_filenames.append(link.get('href'))
    hyp_filenames = []
    for link in soup.findAll(re.compile("^a"), href=re.compile("\.hyp")):
        hyp_filenames.append(link.get('href'))

    rec_links = rec_filenames.copy()
    hyp_links = hyp_filenames.copy()
    for i in range(0, len(rec_filenames)):
        rec_links[i] = url + rec_filenames[i]
        hyp_links[i] = url + hyp_filenames[i]

    for i in range(0, len(rec_filenames)):
        signal_path = 'edf-files/edf/' + rec_filenames[i]
        hypnogram_path = 'edf-files/edf/' + hyp_filenames[i]

        # Check if the files already exist before downloading
        if not Path(signal_path).is_file() or not Path(hypnogram_path).is_file() or \
                os.path.getsize(signal_path) != \
                int(requests.head(rec_filenames[i], allow_redirects=True).headers.get('content-length', None)) or \
                os.path.getsize(hypnogram_path) != \
                int(requests.head(hyp_filenames[i], allow_redirects=True).headers.get('content-length', None)):

            # Download Signal and Hypnogram files:
            print("Downloading " + rec_filenames[i] + "...")
            urllib.request.urlretrieve(rec_links[i], signal_path)
            print("Downloading " + hyp_filenames[i] + "...")
            urllib.request.urlretrieve(hyp_links[i], hypnogram_path)


def edfx_filefinder(url):
    html_page = urllib.request.urlopen(url)
    soup = BeautifulSoup(html_page, "html.parser")

    filenames = []
    for link in soup.findAll(re.compile("^a"), href=re.compile("\.edf")):
        filenames.append(link.get('href'))

    links = filenames.copy()
    for i in range(0, len(filenames)):
        links[i] = url + filenames[i]

    for i in range(0, len(filenames) - 1, 2):
        signal_path = 'edf-files/edfx/' + filenames[i]
        hypnogram_path = 'edf-files/edfx/' + filenames[i + 1]

        # Check if the files already exist before downloading
        if not Path(signal_path).is_file() or not Path(hypnogram_path).is_file() or \
                os.path.getsize(signal_path) != \
                int(requests.head(links[i], allow_redirects=True).headers.get('content-length', None)) or \
                os.path.getsize(hypnogram_path) != \
                int(requests.head(links[i + 1], allow_redirects=True).headers.get('content-length', None)):

            # Download Signal and Hypnogram files:
            print("Downloading " + filenames[i] + "...")
            urllib.request.urlretrieve(links[i], signal_path)
            print("Downloading " + filenames[i + 1] + "...")
            urllib.request.urlretrieve(links[i + 1], hypnogram_path)


url = 'https://www.physionet.org/physiobank/database/sleep-edf/'
edf_filefinder(url)

# Read Sleep Casette data:
url = 'https://www.physionet.org/physiobank/database/sleep-edfx/sleep-cassette/'
edfx_filefinder(url)

# Read Sleep Telemetry data:
url = 'https://www.physionet.org/physiobank/database/sleep-edfx/sleep-telemetry/'
edfx_filefinder(url)

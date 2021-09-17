import os
import pandas as pd

directory = []
file = []
label = []
title = []
texts = []
datapath = "./bbc/"


for dirname, _, filenames in os.walk(datapath):
    # skip the readme file
    if filenames == ["README.TXT"]:
        continue
    for filename in filenames:
        # record document attributes
        directory.append(dirname)
        file.append(filename)
        label.append(dirname.split("/")[-1])

        fullpathfile = os.path.join(dirname, filename)
        with open(fullpathfile, "r", encoding="utf8", errors="ignore") as infile:
            text = []
            firstline = True
            for line in infile:
                if firstline:
                    title.append(line.strip())
                    firstline = False
                elif line == "\n":
                    continue
                else:
                    text.append(line.strip())
            texts.append(" ".join(text))


df = pd.DataFrame(
    list(zip(directory, file, title, texts, label)),
    columns=["directory", "file", "title", "text", "label"],
)

df.to_csv("../data/text.csv", index=False)
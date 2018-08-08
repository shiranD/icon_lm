import argparse
import xml.etree.ElementTree as ET
from collections import defaultdict
import re
import os
import numpy as np
from functools import partial


def parseXml(fname):
    """
    Parse the xml and genrate a dict of the
    icon terms associated with their synonyms
    """
    words = defaultdict(list)
    polysemy = defaultdict(list)
    tree = ET.parse(fname)
    root = tree.getroot()
    children = root.getchildren()
    # make a dict for each word associated with a symbol
    # the word's values are its synonyms
    for child in children:
        wlist = []
        chd = child.attrib
        fname = chd["fileName"]
        names = child.find('names')
        phrase = names.find('en-US').text
        found = re.match(r'^[a-zA-Z]+$', phrase)
        if not found:
            continue
        wlist.append(phrase.lower())
        topic = chd["filePath"]
        topic = topic.split("\\")[0]
        # ensure a single word phrase
        syns = child.find('synonyms')
        eng_syms = syns.find('en-US')
        try:
            for syn in eng_syms.getchildren():
                syn = syn.text
                if not syn:
                    continue
                found = re.match(r'^[a-zA-Z]+$', syn)
                if not found:
                    continue
                wlist.append(syn.lower())
        except:
            pass
        code = fname + "_N_" + topic
        polysemy[phrase] += [code]
        words[code] = wlist
    return words, polysemy


def refine_icon_list(words, polysemy):
    """
    Reduces the icon list
    This function collapses identical meta-data icons.
    For other icons it selects the icon with the 
    richest metadata
    """
    new_words = defaultdict(list)
    for phrase, icon_list in polysemy.items():
        tmp = 0
        for i, icon in enumerate(icon_list):
            if len(words[icon]) > tmp:
                idx = i
                tmp = len(words[icon])
        new_words[icon_list[idx]] = words[icon_list[idx]]
    return new_words


def embd(path):
    """
    Extract the Pre-Trained embeddings
    """
    glove = defaultdict(partial(np.ndarray, 0))
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            line = line.split()
            word = line[0]
            vector = [float(item) for item in line[1:]]
            vector = np.array(vector)
            glove[word] = vector
    return glove


def newEmbd(glove, words, fdict, dim):
    """
    Creat the new Icon embedding by
    adding up the icon's coresponding
    terms in its meta data
    """
    newVecs = defaultdict(partial(np.ndarray, 0))
    newSyns = {}
    for code, wlist in words.items():
        word = wlist[0]
        syns = wlist[1:]
        newVec = np.zeros(((dim),), dtype=float)
        # try except for any term of the list
        for syn in wlist:
            try:
                new = glove[syn]
                if new.any() == 0:
                    continue
                newVec += new
            finally:
                pass

            if newVec.any() == 0:
                continue
            else:
                term = word + " " + code
                newVecs[term] = newVec
    fdict = open(fdict, "w")
    for code, wlist in words.items():
        for j, w in enumerate(wlist):
            val = glove[w]
            if val.any() != 0:
                if j == 0:
                    fdict.write(w + " " + code + " " + "main" + "\n")
                else:
                    fdict.write(w + " " + code + " " + "syns" + "\n")
    fdict.close()
    return newVecs


def validate(glove, newVecs, dim, ficoname):
    """
    Validate and Write it to a file
    """
    ficon = open(ficoname, "w")
    icon_list = []
    icon_vecs = []
    for key, val in newVecs.items():
        val = list(val)
        try:
            if len(val) == dim:
                icon_list.append(key)
                icon_vecs.append(val)
                vec2w = [str(cell) for cell in val]
                vec2w = " ".join(vec2w)
                vec2w = key + " " + vec2w
                ficon.write(vec2w)
                ficon.write("\n")
            else:
                pass
        except BaseException:
            pass
    ficon.close()

    glove_list = []
    glove_vecs = []
    for key, val in glove.items():
        try:
            val = list(val)
            if len(val) == dim:
                glove_list.append(key)
                glove_vecs.append(val)
            else:
                pass
        except BaseException:
            pass

def embed(args):
    """
    The purpose of this script is to
    generate icon embeddings
    """
    # parse the xml
    words, polysemy = parseXml(args.xmlpath)

    # refine list of embeddings
    words = refine_icon_list(words, polysemy)

    # process the embedding data
    embdDict = embd(args.embdspath)
    # create a new embedding
    newVecs = newEmbd(embdDict, words, args.fidict, args.embdim)

    # organize to lists of terms and vectors and write to a file
    validate(embdDict, newVecs, args.embdim, args.ficoname)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Retrieve icon terms and generate icon embeddings')
    parser.add_argument('--embdspath', type=str, help='embedding path')
    parser.add_argument('--embdim', type=int, help='emd dimention')
    parser.add_argument('--xmlpath', type=str, help='icon xml')
    parser.add_argument('--ficoname', type=str, help='icon embedding fname')
    parser.add_argument('--fidict', type=str, help='dict of codes')
    args = parser.parse_args()
    # assert input path validity
    assert os.path.exists(os.path.dirname(
        args.embdspath)), "%r is not a valid path" % args.embdspath
    assert os.path.exists(os.path.dirname(
        args.xmlpath)), "%r is not a valid path" % args.xmlpath
    embed(args)

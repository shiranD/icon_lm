from psychopy import visual, core, event
import numpy as np
from demo_reg import tokenize, evaluate
from embedder import sym2term
from psychopy import gui
import argparse
import os
import re
import pdf2image
import subprocess

import pdb

parser = argparse.ArgumentParser(
    description='PyTorch RNN/LSTM training for Language Models')
parser.add_argument('--aug', action='store_true', help='use icons or not')
parser.add_argument('--data', type=str, help='location of the data corpus')
parser.add_argument('--icon', type=str, help='path to icon embeddings', default='')
parser.add_argument('--iconD', type=str, help='path to icon dict', default='')
parser.add_argument('--path2pdfs', type=str, help='path to icon pdf files', default='')
parser.add_argument('--flist', type=str, help='icon pdf files names', default='')
parser.add_argument('--embd', type=str,
                    help='location of the embeddings')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--load', type=str, default='model.pt',
                    help='path to load the model')
args = parser.parse_args()

s2term = sym2term(args.embd, args.iconD)

# load icon file list
with open(args.flist) as f:
    content = f.readlines()
content = [x.strip() for x in content]
###############################################################################


# Createw window
win = visual.Window(monitor='testMonitor', units ='norm', screen = 0, color=[0.98, 0.9, 0.6])

# Create images
instr = visual.TextStim(win, units='norm', height = 0.1, text = 'Press any key to begin', color = [0,0,0])

def process_tokens(tknlist):
    new = []
    for code in tknlist:
        if '_N_' not in code:
            new.append(code)
        else:
            fname, topic = code.split('_N_')
            new.append(topic+'#'+fname)
    return new 

def gen_icons(paths, ranks, ficons):

    # generate temporary icons
    both = paths + ranks
    for p in both:
        ab = p.split("#")
        if len(ab) == 1:
            continue
        try:
            r = re.compile(".*"+ab[0]+"\\\\"+ab[1]+".pdf")
            new = list(filter(r.match, ficons))[0]
        except:
            r = re.compile(".*"+ab[0]+".*"+ab[1]+".pdf")
            new = list(filter(r.match, ficons))[0]
        # convert to jpg
        pdf2image.convert_from_path(args.path2pdfs+new, output_folder="tmp", fmt="jpg", fname = p)
        # resize
        filei = os.path.join(os.getcwd(),'tmp/'+p+'-1.jpg')
        subprocess.call(['sips', '-Z', '64', filei])

def make_obj(paths, ranks, ficons):

    gen_icons(paths, ranks, ficons)
    items = []
    paths.append('blank')
    my_path = 'tmp/' 
    for path in paths:
        pathsplt = path.split('#')
        if len(pathsplt)==1:
            if pathsplt[0] == 'blank':
                tag=""
            elif pathsplt[0] == '>':
                path="eof"
                tag="eof"
            else:
                path='ptrained'
                tag ='<'+pathsplt[-1]+'>'
        else:
            tag ='<'+pathsplt[-1]+'>'
        items.append(visual.ImageStim(win, image=my_path+path+'-1.jpg'))
        items.append(visual.TextStim(win, text=tag, color=[0,0,0], height=0.05))
    
    for n, path in enumerate(ranks):
        pathsplt = path.split('#')
        if len(pathsplt)==1:
            if pathsplt[0] == '>':
                path="eof"
                tag="eof"
            else:
                path='ptrained'
                tag='<'+pathsplt[-1]+'>'
        else:
            tag ='<'+pathsplt[-1]+'>'
        items.append(visual.ImageStim(win, image=my_path+path+'-1.jpg'))
        items.append(visual.TextStim(win, text=tag, color=[0,0,0], height=0.05))
        items.append(visual.TextStim(win, text=str(n+1)+".", color=[0,0,0], height=0.05))
        
    # History Image parameters
    im_positions = {}
    x = -0.8 
    y = 0.5
    y1 = 0.33
    for i in range(len(paths)):
        im_positions[i]=(x+i*0.3, y)
        im_positions[i+len(paths)]=(x+i*0.3, y1)
    
    # History Image parameters
    x = -0.5 
    y = -0.1
    y1 = -0.27
    for i in range(len(ranks)):
        im_positions[2*len(paths)+i]=(x+i*0.3, y)
        im_positions[2*len(paths)+i+len(ranks)]=(x+i*0.3, y1)
        im_positions[2*len(paths)+i+2*len(ranks)]=(x+i*0.3-0.13, y+0.05)

    return items,im_positions
    
# Instructions
continueRoutine = True
while continueRoutine:
    instr.draw()
    win.flip()
    if event.getKeys():
        continueRoutine = False

continueRoutine = True
rank = -1
for trials in range(5):
    if rank==-1:
        # insert text to a dialogue box
        myDlg = gui.Dlg(title="Demo number "+str(trials))
        myDlg.addField('query:')
        sentence = myDlg.show()[0]  # show dialog and wait for OK or Cancel
        if myDlg.OK:  # or if ok_data is not None
            print(sentence)
        else:
            print('user cancelled')
    else:
        sentence=sentence+" "+s2term[topk[rank]]

    q_data, tokens = tokenize(sentence)
    # make sure is of paths
    paths = process_tokens(tokens)
    topk = evaluate(q_data)
    ranks = process_tokens(topk)
    items, im_positions = make_obj(paths, ranks, content)
    
    # Create list of position keys for shuffling on every trial
    imKeys = np.array(list(im_positions.keys()))
    for j, obj in enumerate(items):
        obj.pos=(im_positions[imKeys[j]])

    # Set position before each trial
    while continueRoutine:
        for obj in items:
            # draw image
            obj.draw()

        if event.getKeys(['return']):
            conitueRoutine = False
            rank = -1
            break
        
        if event.getKeys(['1']):
            rank = 0
            conitueRoutine = False
            break

        if event.getKeys(['2']):
            rank = 1
            conitueRoutine = False
            break

        if event.getKeys(['3']):
            rank = 2
            conitueRoutine = False
            break

        if event.getKeys(['4']):
            rank = 3
            conitueRoutine = False
            break

        win.flip()
win.close()

core.quit()

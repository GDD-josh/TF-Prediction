import math
import random
import numpy as np
import pickle as pkl

def analyzedata():
    with open("boundseqs.fa") as regions:
        d = {}
        count = 0
        for r in regions:
            if ">" in r:
                continue
            count += 1
            lr = len(r.replace("\n", ""))
            if lr in d:
                d[lr] = d[lr] + 1
            else:
                d[lr] = 1
        for i, val in d.items():
            d[i] = (val)/float(count)
        return d

def randomlength(d):
    rand = random.uniform(0, 1)
    min = 0
    for i, val in d.items():
        if rand >= min and rand <= val + min:
            return i
        min += val
    return 15

def findunbound():
    d = analyzedata()
    outfile = open('unboundMotifs.txt', 'w')
    # Positive sequences
    with open("factorbookMotifPos.txt") as boundregions:
        count = 0
        lastpos = 0
        lastchr = ""
        for region in boundregions:
            count = count + 1
            if (count % 500 == 0):
                print("Lines read: " + str(count))
            # Parsing
            region = region.replace("\n", "")
            tokens = region.split("\t")
            chr = tokens[1]
            if chr != lastchr:
                lastpos = 0
                lastchr = chr
            sp = int(tokens[2])
            ep = int(tokens[3])
            rl = randomlength(d)
            rands = 0
            try:
                rands = random.randint(lastpos, sp-rl+1)
            except:
                rands = random.randint(sp-rl+1, lastpos)
            lastpos = ep
            outfile.write("none\t" + chr + "\t" + str(rands) + "\t" + str(rands + rl) + "\tnone\tnone\tnone\n")
    outfile.close()

def findseqs():
    outfile = open('unboundseqs.fa', 'w')
    # Positive sequences
    with open("unboundMotifs.txt") as boundregions:
        count = 0
        for region in boundregions:
            count = count + 1
            if (count % 500 == 0):
                print("Lines read: " + str(count))
            # Parsing
            rand = random.randint(1,13)
            if (rand != 1):
                continue
            region = region.replace("\n", "")
            tokens = region.split("\t")
            chr = tokens[1]
            startpos = int(tokens[2])
            endpos = int(tokens[3])
            tname = tokens[4]
            score = tokens[5]
            strand = tokens[6]
            # Query the chrom file
            filename = "chromFa/" + chr + ".fa"
            seq = ""
            seqlength = endpos - startpos
            # Each line has 50 characters
            startline = int(math.floor(startpos/50) - 1)
            startchar = startpos % 50
            dna = open(filename)
            for i, line in enumerate(dna):
                if (i < startline):
                    continue
                line = line.replace("\n", "")
                remainder = 50 - startchar
                # sequence is all on one line
                if (remainder - seqlength) >= 0:
                    seq = seq + line[startchar:startchar+seqlength]
                    break
                # sequence spills over to next line
                else:
                    seq = seq + line[startchar:50]
                    startline = startline + 1
                    startchar = 0
                    seqlength = seqlength - remainder
            dna.close()
            outfile.write(">" + chr + " " + tname + " " + score + " " + strand + "\n")
            outfile.write(seq + "\n")
    outfile.close()

def format_props():
    seqtype = 'unbound'
    ep = open(seqtype + 'seqs.fa.EP', 'r')
    helt = open(seqtype + 'seqs.fa.HelT', 'r')
    mgw = open(seqtype + 'seqs.fa.MGW', 'r')
    prot = open(seqtype + 'seqs.fa.ProT', 'r')
    roll = open(seqtype + 'seqs.fa.Roll', 'r')
    seqs = []

    i = 0
    while True:
        i += 1
        if i % 100 == 0:
            print(i)
        sep = ep.readline()
        sep = ep.readline()
        if not sep:
            break
        shelt = helt.readline()
        shelt = helt.readline()
        smgw = mgw.readline()
        smgw = mgw.readline()
        sprot = prot.readline()
        sprot = prot.readline()
        sroll = roll.readline()
        sroll = roll.readline()
        sep = sep.replace('\n', '').split(',')
        shelt = shelt.replace('\n', '').split(',')
        smgw = smgw.replace('\n', '').split(',')
        sprot = sprot.replace('\n', '').split(',')
        sroll = sroll.replace('\n', '').split(',')
        arr = [sep, shelt, smgw, sprot, sroll]
        seqs.append(arr)

    seqs = np.array(seqs)
    with open(seqtype + '.pkl','wb') as f:
        pkl.dump(seqs, f)

    ep.close()
    helt.close()
    mgw.close()
    prot.close()
    roll.close()

def make_sets():
    bound = pkl.load(open("bound.pkl", "rb" ))
    bound = bound.tolist()
    unbound = pkl.load(open("unbound.pkl", "rb" ))
    unbound = unbound.tolist()
    btrain = bound[0:108000]
    ubtrain = unbound[0:108000]
    bvalid = bound[108000:108000+36000]
    ubvalid = unbound[108000:108000+36000]
    btest = bound[108000+36000:108000+36000+36000]
    ubtest = unbound[108000+36000:108000+36000+36000]

    trainx = []
    trainy = []
    for i, val in enumerate(btrain):
        trainx.append(btrain[i])
        trainx.append(ubtrain[i])
        trainy.append(1)
        trainy.append(0)

    validx = []
    validy = []
    for i, val in enumerate(bvalid):
        validx.append(bvalid[i])
        validx.append(ubvalid[i])
        validy.append(1)
        validy.append(0)

    testx = []
    testy = []
    for i, val in enumerate(btest):
        testx.append(btest[i])
        testx.append(ubtest[i])
        testy.append(1)
        testy.append(0)

    trainx = np.array(trainx)
    with open('trainx.pkl','wb') as f:
        pkl.dump(trainx, f)
    trainy = np.array(trainy)
    with open('trainy.pkl','wb') as f:
        pkl.dump(trainy, f)

    validx = np.array(validx)
    with open('validx.pkl','wb') as f:
        pkl.dump(validx, f)
    validy = np.array(validy)
    with open('validy.pkl','wb') as f:
        pkl.dump(validy, f)

    testx = np.array(testx)
    with open('testx.pkl','wb') as f:
        pkl.dump(testx, f)
    testy = np.array(testy)
    with open('testy.pkl','wb') as f:
        pkl.dump(testy, f)

def check_set(name):
    s = pkl.load(open(name, "rb" ))
    s = s.tolist()
    print(s[:100])

check_set('testy.pkl')
#make_sets()
#format_props()
#findunbound()

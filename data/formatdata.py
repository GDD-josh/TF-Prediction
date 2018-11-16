import math

def boundseqs():
    outfile = open('boundseqs.txt', 'w')
    # Positive sequences
    with open("factorbookMotifPos.txt") as boundregions:
        count = 0
        for region in boundregions:
            count = count + 1
            if (count % 500 == 0):
                print("Lines read: " + str(count))
            # Parsing
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

boundseqs()

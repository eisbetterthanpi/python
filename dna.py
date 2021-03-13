
#s=input("dna seq: ")
s="tctctcacgctgtgattctgctctgaattata\
gtcggcgacgtttcgaacgaatatccacaattaa\
tactgcttaatagtcttttctcttgactataaag"
s=s.upper()
#print(s)
#print(s.count('C'));print(s.count('G'));print(s.count('A'));print(s.count('T'))

bp=len(s)
#print(bp)
tl={'A':'T','T':'A','C':'G','G':'C'}

l=20
fs=0
rs=0
fwd=s[fs:fs+l]
rev=s[bp-l-rs:bp-rs]
#print(fwd);print(rev)

gc=lambda x:sum(map(x.count,["C","G"]))
#print("gc: ",gc(fwd),"/",l)

temp=lambda x:2*(l-gc(x))+4*gc(x)


re=[tl[x] for x in reversed(rev)]
r=''.join(re)
print("reverse: ",r)



gencode={'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R', 'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 'TAC':'Y', 'TAT':'Y', 'TAA':'_', 'TAG':'_', 'TGC':'C', 'TGT':'C', 'TGA':'_', 'TGG':'W'}

codon={'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M', 'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T', 'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K', 'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R', 'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L', 'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P', 'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q', 'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R', 'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V', 'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A', 'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E', 'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G', 'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S', 'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L', 'TAC':'Y', 'TAT':'Y', 'TAA':'stop', 'TAG':'stop', 'TGC':'C', 'TGT':'C', 'TGA':'stop', 'TGG':'W'}


def h(s):
    for p in range(3):
        #len(s)//3
        tl=[codon[s[3*n+p:3*n+3+p]] for n in range((len(s)-p)//3)]
        trl=''.join(tl)
        print(trl)


#h(s)

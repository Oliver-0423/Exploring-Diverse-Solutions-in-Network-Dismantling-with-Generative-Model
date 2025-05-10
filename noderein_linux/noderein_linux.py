import os
import re
import networkx as nx
from networkx import generate_edgelist



def NodeRein_linux(G, remove_list, target):
    path = 'noderein_linux/graph.txt'
    node_num = nx.number_of_nodes(G)
    with open(path, "wb") as fh:
        for line in generate_edgelist(G, ' ', data=False):
            line = "D " + line + "\n"
            fh.write(line.encode("utf-8"))
        for i in remove_list:
            line = 'S ' + str(i) + "\n"
            fh.write(line.encode("utf-8"))
    os.system('cat ./noderein_linux/graph.txt | ./noderein_linux/reverse-greedy -t '+str(target)+' > ./noderein_linux/output.txt')
    final_remove_list = []
    with open('noderein_linux/output.txt', 'r') as fo:
        for line in fo:
            if 'S' in line:
                number = re.findall("\d+", line)
                number = int(number.pop())
                final_remove_list.append(number)
    return final_remove_list, len(final_remove_list) / node_num


import sys #library to check wether file ame give or not
try:
    with open(sys.argv[1]) as f:  
        lines = f.readlines()   #eading files in line format
except Exception:
    print("enter file name in command line")  #indicating wrong name not given
    quit()
if len(sys.argv)>2:
    print("more than one file given as input")
    quit()
num_lines = len(lines)
test_start = 0
test_end = 0
for line_no in range(num_lines) : #checking wether right format file give or not
    if lines[line_no] == ".circuit\n":
        start_line_num = line_no
        test_start = 1        
    if lines[line_no] == ".end" or lines[line_no] == ".end\n" :   
        end_line_num = line_no
        test_end = 1
if test_start * test_end == 0 :
    print("wrong file given as input")
    quit()
reqd_lines = lines[(start_line_num+1):end_line_num]  #slicing the required ines
a = []
for l in reqd_lines:
    a.append(((l.split("#")[0]).split()))  #removing comments and splitting by spaces
  
for b in a:
    b.reverse()      #reversing the elements of list
a.reverse()          #reversing the lines!!
for c in a:
    print(" ".join(c))  #printing by joining





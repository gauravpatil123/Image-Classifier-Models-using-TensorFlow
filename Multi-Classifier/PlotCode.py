import matplotlib.pyplot as plt

def plot(y1, y2, x, param, Set1, Set2, color1, color2):
    plt.figure(figsize=(50, 30), dpi = 120)
    plt.plot(x, y1, color1, linewidth = 3)
    plt.plot(x, y2, color2, linewidth = 3)
    plt.xticks(fontsize = 20)
    plt.yticks(fontsize = 20)
    plt.legend([Set1+' - '+param, Set2+' - '+param], loc = 'upper right', fontsize = 24)
    plt.title(r"Comparison of "+Set1+ " "+param+ " and "+Set2+ " "+ param, fontsize = 36)
    plt.xlabel(r"Number of Epochs", fontsize = 24)
    plt.ylabel(param, fontsize = 24)
    plt.savefig("Images/"+Set1+"_v_"+Set2+"_"+param+".png")
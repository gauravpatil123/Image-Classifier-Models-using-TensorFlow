"""
PlotCode:
    Defines a class Graph to save plot images of the metrics from the trained model
"""

import matplotlib.pyplot as plt

class Graph:

    """
    class to create plots from the input metrics
    """

    def __init__(self, y1, y2, x, param, Set1, Set2, color1, color2):
        """
        Input:
            y1: values set 1
            y2: values set 2
            x: length of the sets y1 and y2
            param: comparison parameter being plotted
            Set1: name of the set y1
            Set2: name of the set y2
            color1: line color for set y1
            color2: line color for set y2

        Action:
            Initilaizes the input variables to class
        """
        self.y1 = y1
        self.y2 = y2
        self.x = x
        self.param = param
        self.Set1 = Set1
        self.Set2 = Set2
        self.color1 = color1
        self.color2 = color2

    def __call__(self):
        """
        creates the plot and saves it in 'Images/' directory
        """
        y1, y2, x, param = self.y1, self.y2, self.x, self.param
        Set1, Set2, color1, color2 = self.Set1, self.Set2, self.color1, self.color2
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


'''
def plot(y1, y2, x, param, Set1, Set2, color1, color2):
    """
    Input:
        y1: values set 1
        y2: values set 2
        x: length of the sets y1 and y2
        param: comparison parameter being plotted
        Set1: name of the set y1
        Set2: name of the set y2
        color1: line color for set y1
        color2: line color for set y2
    """
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
'''
from tkinter import *
from tkinter import filedialog
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,  
NavigationToolbar2Tk) 
import Channel as ch

class GUI_Handler:
    def __init__(self):
        self.mesh:              ch.Channel = None
        self.window:            Tk = None
        self.plotFrame:         Frame = None
        self.resolution:        float = 1       # TODO: abstract it
        # Flags
        self.foundCentroids:    bool = False    # NOTE: could get this from mesh
        self.straightened:      bool = False
        
        # Plot
        self.axes               = None          # Holds the plot data
        self.fig                = None
        self.plotCanvas         = None
        self.toolbar            = None

        # Constants
        self.BUTTON_HEIGHT = 2
        self.BUTTON_WIDTH = 10

    def plot(self):
        # if self.fig == None:
        self.fig = fig = Figure(figsize= (4, 3), dpi= 100)
        # else:
        #     fig = self.fig

        if self.axes == None:
            self.axes = plot1 = fig.add_subplot(projection= '3d')
        else:
            plot1 = self.axes
            plot1.clear()

        if self.mesh == None:
            fig.suptitle("Default Plot")
            plot1.plot([0, 1], [0, 2], [0, 2])
            print("No mesh loaded, default plot")
        else:
            self.mesh.plotMesh(plot1, self.resolution)

        # create Tkinter canvas containing
        # matplotlib figure
        if self.plotCanvas == None:
            self.plotCanvas = canvas = FigureCanvasTkAgg(fig, master = self.plotFrame)
        else:
            canvas = self.plotCanvas

        if self.foundCentroids:
            self.mesh.plotCentroids(self.axes)
        canvas.draw()

        if self.toolbar == None:
            self.toolbar = toolbar = NavigationToolbar2Tk(canvas, self.plotFrame)
        else:
            toolbar = self.toolbar
        toolbar.update()
        canvas.get_tk_widget().pack(fill= BOTH, expand= True)
        
    # Reads vectors
    def getFile(self):
        file = filedialog.askopenfile(mode='r', filetypes=[("Mesh files", "*.wrl *.stl")])
        if file == None:
            print("File not found")
        else:
            print(f"Opening {file}")
            file.close()
            print(file.name)
            self.mesh = ch.Channel(file.name)
            self.mesh.readVectors()
    
    def findCentroids(self):
        self.checkMesh()
        if self.foundCentroids == True:
            return
        # Finds the centroids for the object
        self.mesh.findCentroids()
        self.foundCentroids = True
        self.mesh.plotCentroids(self.axes)
    
    def checkMesh(self) :
        if self.mesh == None:
            raise Exception("Mesh hasn't been loaded yet!")

    def straighten(self):
        self.checkMesh()
        # Straightens the mesh
        if self.straightened == True:
            return
        self.mesh.straighten(self.resolution, True)
        self.straightened = True

    def rotate(self):
        self.checkMesh()
        # Rotates the mesh by the amount in the text box
        self.mesh.rotate((x, y, z))
        self.straightened = False
        self.foundCentroids = False

    def plotDiameter(self):
        self.checkMesh()
        # call the plot diameter function of Channel
        # TODO: Add functionality to intake width
        self.mesh.plotChunkDiameter(specifiedWidth = 0)

    def reset(self):
        self.window.destroy()
        self.__init__()
        self.createGUI()

    def createGUI(self):
        self.window = Tk()
        # Add menubar
        menubar = Menu(self.window)
        self.window.config(menu = menubar)

        # Add file option in menubar
        filemenu = Menu(menubar, tearoff= 0)
        menubar.add_cascade(label= "File", menu= filemenu)
        filemenu.add_command(label= "Open", command= self.getFile)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.window.quit)
        # Add help menu
        helpmenu = Menu(menubar, tearoff= 0)
        menubar.add_cascade(label='Help', menu=helpmenu)
        helpmenu.add_command(label='About')

        self.window.title("Mesh Diameter Plotter")
        self.window.geometry("500x500")

        # Build framing
        topFrame = Frame(self.window, bg= "red", height= 100, width= 300)
        topFrame.pack(side= "top", fill= BOTH, expand= True)
        self.plotFrame = plotFrame = Frame(topFrame, highlightcolor= "blue", height= 100, width= 300)
        plotFrame.pack(side= "left", anchor= "nw", fill= BOTH, expand= True)
        topRightFrame = Frame(topFrame, bg= "green", height = 100, width = 100)
        topRightFrame.pack(side= "right", fill= BOTH, expand= True)
        # Buttons and options
        buttonFrame = Frame(topRightFrame, bg = "yellow", height = 100, width= 100)
        buttonFrame.pack(side = "right", expand= True, anchor= W)
        optionsFrame = Frame(topRightFrame, bg = "orange", height = 100, width = 100)
        optionsFrame.pack(side = "left", fill= BOTH, expand = True)
        # Frame each button
        p2b = 20
        plot_Frame = Frame(optionsFrame, bg= "brown", height = self.BUTTON_HEIGHT, width = self.BUTTON_WIDTH)
        straighten_Frame = Frame(optionsFrame, bg= "blue", height = self.BUTTON_HEIGHT, width = self.BUTTON_WIDTH)
        find_centroid_Frame = Frame(optionsFrame, bg= "brown", height = self.BUTTON_HEIGHT, width = self.BUTTON_WIDTH)
        rotate_Frame = Frame(optionsFrame, bg= "blue", height = self.BUTTON_HEIGHT, width = self.BUTTON_WIDTH)
        plot_diameter_Frame = Frame(optionsFrame, bg= "brown", height = self.BUTTON_HEIGHT, width = self.BUTTON_WIDTH)
        read_file_Frame = Frame(optionsFrame, bg= "blue", height = self.BUTTON_HEIGHT, width = self.BUTTON_WIDTH)
        reset_Frame = Frame(optionsFrame, bg= "brown", height = self.BUTTON_HEIGHT, width = self.BUTTON_WIDTH)

        # Bottom frame
        bottomFrame = Frame(self.window, bg= "brown", height = 150, width = 300)
        bottomFrame.pack(side = "bottom", fill= BOTH, expand=True)

        # Create button
        plot_button = Button(master = buttonFrame,
                            command = self.plot,
                            height = self.BUTTON_HEIGHT,
                            width = self.BUTTON_WIDTH,
                            text = "Plot")
        straighten_button = Button(master = buttonFrame,
                                command = self.straighten,
                                height = self.BUTTON_HEIGHT,
                                width = self.BUTTON_WIDTH,
                                text = "Straighten")
        find_centroid_button = Button(master = buttonFrame,
                                    command = self.findCentroids,
                                    height = self.BUTTON_HEIGHT,
                                    width = self.BUTTON_WIDTH,
                                    text = "Find Centroids")
        rotate_button = Button(master = buttonFrame,
                            command = self.rotate,
                            height = self.BUTTON_HEIGHT,
                            width = self.BUTTON_WIDTH,
                            text = "Rotate")
        plot_diameter_button = Button(master = buttonFrame,
                                    command = self.plotDiameter,
                                    height = self.BUTTON_HEIGHT,
                                    width = self.BUTTON_WIDTH,
                                    text = "Plot Diameter")
        read_file_button = Button(master = buttonFrame,
                                command = self.getFile,
                                height = self.BUTTON_HEIGHT,
                                width = self.BUTTON_WIDTH,
                                text = "Read File")
        reset_button = Button(master = buttonFrame,
                              command = self.reset,
                              height = self.BUTTON_HEIGHT,
                              width = self.BUTTON_WIDTH,
                              text = "Reset")
        
        # Text input options
        rotate_text = Text(rotate_Frame,
                           height = self.BUTTON_HEIGHT / 2,
                           width = self.BUTTON_WIDTH)
        #rotate_text_instructions = Label

        # read file
        # calculate centroids, datamean, etc
        # fitGeometry
        # straighten

        # Packing
        # Place button onto window
        read_file_button.pack()
        plot_button.pack()
        find_centroid_button.pack()
        straighten_button.pack()
        rotate_button.pack()
        plot_diameter_button.pack()
        reset_button.pack()
        # Options
        rotate_Frame.pack()
        # rotate_text.pack()
        # Start mainloop
        self.window.mainloop()

gui = GUI_Handler()
gui.createGUI()

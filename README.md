Computational Metabolic Phenotyping within a PyQT6 GUI (CMPp6)


  To install, download repository to desired folder.
  
  Current version uses python 3.14, backwards compatible with 3.13

  
  Use PowerShell or cmd prompt
  Navigate to saved location : cd c:\...\venv location
  Create and activate a venv : python -m venv CMP 
                               venv\scripts\activate
                               
  Install requirements       :  pip install -r requirements.txt

  Run program                : python main.py
  
This program is designed to view registered image stacks of silvered tissue representing metabolic probes and perform cluster analysis to segment and identify cell signatures.
Acceptable image formats are JPG, PNG, TIF, or BMP. 

For accurate clustering, images need to be white-on-black, probe-to-background. Typical image acquisition is black-on-white silhouette from a brightfield scope and needs to be inverted before processing. 

Program contains modest tools for non-destructive image modification, allowing for cropping, rotating, and filters (gaussian, median, etc).  These modifications are automatically applied to all images
  and can be undone using the Tools > Image Adjustments > Undo (Invert, Rotation, or Crop). Alternatively, you can delete the associated JSON files to reset images.

When program opens, there is a "Home" button in the upper left corner. Click and navigate to the folder containing your images. When selected, the program will create three folders for storing cluster masks, histograms, and JSON files.

Images will appear in a list on the left hand of the program. Clicking on an image will display it as well as mark it as "selected" for modifications, saving, or clustering. 

Right-clicking on an image will allow you to change color and view contrast stretched.  Color and contrast stretching is not applied to images for clustering or filtering and is only for visualization.
  If you want to save a view of the modified images with color and contrast Tools > Export Modified Images. This creates a snapshot of what is visible in the Image Viewer window.

You can add files to your project by clicking File > Import Image or File > Import Mask.  Copies will be saved to your home folder.

Crop, rotate and invert images to desired ROI.  If needed, add filter to smooth out image.
A note about rotating: clustering is orientation agnostic, however, there is an option to weight clustering based on horizontal or vertical distances. 

Typically, the first step is to isolate the foreground from the background using k-Means or Gaussian Mixture (GMM).

  Select all images, making them visible, and flagging them for clustering.
  
  Set K-means "Number of Clusters" to 4 will produce exactly four clusters.  
  
  Clusters will be listed in the Cluster Mask list and visible when clicked. 
  	Typically, three of the produced clusters will represent the foreground and can be combined to create a new mask.  You can then delete the original four. Be aware deleting is permanent and removes the file from the Home folder.
	
  If the masks k-Means produces is too restrictive, excluding fainter signal, you can use GMM and set "Number of Components" to 4.  Follow the same procedure for k-Means.
    You can also use the "Normalize (per image):" or "...(global):" to fine-tune results.  Only use clustering to create isolating masks, do not use for generating cell profiles. 

For clustering, begin by selecting only those probes you are interested in.

  To isolate clustering to just the foreground signal, select the mask created using k-Means or GMM.
  
  Click Cluster > ISODATA
  
In the parameters window use default settings for the first run. To isolate clustering to foreground, select a mask from the list in Cluster under Mask section. 
  After clustering a .csv file is produced showing the basic stats for each cluster as it is applied to each image used for clustering. Overall fitness scores are produced as well; silhouette, Davies-Bouldin, and Calinski-Harabasz

  Delete or merge masks as needed.

  To produce histograms, select One mask and any number of images. 
  
  Histograms will be listed in the Graphs section and a JSON saved in the Graphs folder.  
  
  You can select which histograms to view together and save a .csv and snapshot png.
  
  You can also produce a joint plot with up to three comparisons between two images under a specified mask. 

  Fine-tuning clustering can be accomplished by using spatial awareness in the ISODATA parameters window. 
  
  By checking "Include Coordinates" you can set the 'cost' of clustering in either the x or y dimenstion.
  
 If you have oriented your retina horizontally, then you can set the cost to favor clustering along layers by entering a near zero value in "X Weight:" and a higher value in "Y Weight:".  
    Adjust the x and y weights as needed according to your retina's orientation.  Higher value, higher cost of including signal; lower value, lower cost to include distant signal. 

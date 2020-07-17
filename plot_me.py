"""
Official Code Implementation of:
"A Gated and Bifurcated Stacked U-Net Module for Document Image Dewarping"
Authors:    Hmrishav Bandyopadhyay,
            Tanmoy Dasgupta,
            Nibaran Das,
            Mita Nasipuri

Code: Hmrishav Bandyopadhyay

"""


import matplotlib.pyplot as plt
import os



def plot(training_loss,validation_loss,plot_path="."):
	
	plt.title("Loss after epoch: {}".format(len(training_loss)))
	plt.xlabel("Epoch")
	plt.ylabel("Loss")

	
	plt.plot(list(range(len(training_loss))),training_loss,color="r",label="Training Loss")
	plt.plot(list(range(len(validation_loss))),validation_loss,color="b",label="Validation Loss")
	

	plt.savefig(plot_path+"loss_plot.png")
	


# -*- coding: utf-8 -*-
#
# caffeNetViewer
# Programmed by : Birol Kuyumcu ( bluekid70@gmail.com )
#				: https://tr.linkedin.com/in/birol-kuyumcu-53798771
# License : GPL
# Requirements :
#   PySide : https://github.com/PySide
#   Caffe : https://github.com/BVLC/caffe
#   openCv : https://github.com/opencv

import sys
import cv2
from PySide import QtCore, QtGui,QtWebKit
from caffeNetViewer_ui import Ui_Dialog
import caffe
import os
import numpy as np


class caffeNetViewerForm(QtGui.QWidget):
	def __init__(self, parent=None):
		QtGui.QWidget.__init__(self, parent)
		self.ui = Ui_Dialog()
		self.ui.setupUi(self)
		#
		self.dir = os.getcwd()
		self.fDialog = QtGui.QFileDialog(self)
		self.fname =''	
		self.cImg = None
		
		self.ui.tabWidget.setEnabled(False)
		
		self.isModelLoaded = False
		self.net = None
		self.load()
		
	def disableModel(self):
		self.ui.tabWidget.setEnabled(False)
		self.isModelLoaded = False		
		
		
	@QtCore.Slot()
	def on_pushButtonExec_clicked(self):
		if self.isModelLoaded == False:
			self.loadLayers()		
			
		self.runCaffeModel()
		self.ui.tabWidget.setEnabled(True)
		self.ui.comboBoxChannel.setEnabled(False)
		


	@QtCore.Slot()
	def on_toolButtonModel_clicked(self):
		Sgn = ('Caffe Net Model File ', "(*.prototxt)")
		fileName = self.fDialog.getOpenFileName(self,Sgn[0], self.dir, Sgn[1])
		if fileName[0] == u'':
			return
		self.dir = os.path.dirname(fileName[0])
		self.ui.comboBoxModel.addItem(fileName[0])	
		self.ui.comboBoxModel.setCurrentIndex(self.ui.comboBoxModel.count()-1)
		self.disableModel()
		
	@QtCore.Slot()
	def on_toolButtonWeights_clicked(self):
		Sgn = ('Caffe Trained Weights File ', "(*.caffemodel)")
		fileName = self.fDialog.getOpenFileName(self,Sgn[0], self.dir, Sgn[1])
		if fileName[0] == u'':
			return
		self.dir = os.path.dirname(fileName[0])
		self.ui.comboBoxWeights.addItem(fileName[0])
		self.ui.comboBoxWeights.setCurrentIndex(self.ui.comboBoxWeights.count()-1)
		self.disableModel()
		
	@QtCore.Slot()
	def on_toolButtonImage_clicked(self):
		print 'Image File'
		Sgn = ( 'Image File', "Image Files (*.png *.jpg *.bmp)")
		fileName = self.fDialog.getOpenFileName(self,Sgn[0], self.dir, Sgn[1])
		if fileName[0] == u'':
			return
		self.dir = os.path.dirname(fileName[0])
		self.ui.comboBoxImage.addItem(fileName[0])
		self.ui.comboBoxImage.setCurrentIndex(self.ui.comboBoxImage.count()-1)

	@QtCore.Slot()
	def on_comboBoxLayers_currentIndexChanged(self):	
		if self.cImg is None :
			return
		ix = self.ui.comboBoxLayers.currentIndex() 
		if self.layerList[ix] == 'data':
			self.ui.comboBoxChannel.setEnabled(False)
			self.ui.comboBoxChannel.clear()			
			self.showImg(self.ui.labelDisplay,self.cImg)			
		else:
			self.out_data = self.net.blobs[self.layerList[ix]].data[0]
			if len(self.out_data.shape) > 2:	
				ch_list = ["Full"]+[str(x) for x in range(self.out_data.shape[0])]
				self.ui.comboBoxChannel.clear()
				self.ui.comboBoxChannel.addItems(ch_list)
				self.ui.comboBoxChannel.setEnabled(True)
			else:
				print 'Not Implemented yet ... '
		
	@QtCore.Slot()
	def on_comboBoxChannel_currentIndexChanged(self):
		ix = self.ui.comboBoxChannel.currentIndex()
		if ix == 0 :
			self.vis_square(self.out_data)
		else:
			oimg = self.out_data[ix-1]
			oimg = 255*(oimg-np.min(oimg))/(np.max(oimg)-np.min(oimg))			
			oimg = oimg.astype(np.uint8)
			self.showImg(self.ui.labelDisplay,oimg)
		

	@QtCore.Slot()	
	def on_comboBoxOutType_currentIndexChanged(self):
		# Not Implemented
		pass

	def closeEvent(self,event):
		print 'Exiting...'
		self.save()
		
	def save(self):
		print 'Saving ConfFile'
		cFile = 'ConfFile.txt'
		f = open(cFile, 'w')
		f.write(self.ui.comboBoxModel.currentText()+'\n')
		f.write(self.ui.comboBoxWeights.currentText() + '\n')
		f.write(self.ui.comboBoxImage.currentText() + '\n')
		f.close()
		
	def load(self):
		print 'Loading ConfFile'	
		cFile = 'ConfFile.txt'
		if os.path.isfile(cFile) :
			f = file(cFile)
			lst = f.readlines()
			f.close()
			self.ui.comboBoxModel.addItem(lst[0].rstrip())
			self.ui.comboBoxWeights.addItem(lst[1].rstrip())
			self.ui.comboBoxImage.addItem(lst[2].rstrip())
			self.dir  = os.path.dirname(lst[2].rstrip())			

	def runCaffeModel(self):
		iname = str(self.ui.comboBoxImage.currentText())
		self.cImg = cv2.imread(iname)
		self.cImg = cv2.cvtColor(self.cImg, cv2.COLOR_BGR2RGB)
		self.ui.plainTextEdit.appendPlainText('Model Running ... ')
		self.ui.plainTextEdit.appendPlainText('  Image Name : '+iname)
		self.ui.plainTextEdit.appendPlainText("  Image Shape : " + str(self.cImg.shape))
		self.ui.plainTextEdit.appendPlainText("  Model Input Image Shape : " + str(self.net.blobs['data'].data.shape))	

		transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
		transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
		#transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
		transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
		transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR			

		image = caffe.io.load_image(iname)
		inData = transformer.preprocess('data', image)
		
		self.net.blobs['data'].data[...] = [inData]

		self.outClass = self.net.forward()
		self.on_comboBoxLayers_currentIndexChanged()
		
		
	def loadLayers(self):
		mname = str(self.ui.comboBoxModel.currentText())
		wname = str(self.ui.comboBoxWeights.currentText())
		self.net = caffe.Net(mname,wname , caffe.TEST)
		out = self.net.blobs
		self.layerList = out.keys()
		self.ui.comboBoxLayers.clear()
		for ln in self.layerList :
			self.ui.comboBoxLayers.addItem(ln)
			
		self.ui.plainTextEdit.clear()
		self.ui.plainTextEdit.appendPlainText('Caffe Model Loaded...')
		self.ui.plainTextEdit.appendPlainText('  Model Name : '+mname)
		self.ui.plainTextEdit.appendPlainText('  Weights Name : '+wname)
		
		self.ui.plainTextEdit.appendPlainText("Network Layers ...")
		
		for name, layer in zip(self.net._layer_names, self.net.layers):
			if not name in self.layerList :
				continue
			msg = "   "+name +" --> "+str(layer.type) +" --> "+ str((self.net.blobs[name].data[0]).shape)
			self.ui.plainTextEdit.appendPlainText(msg)		
		
		self.isModelLoaded = True
		
	def showImg(self,label,img):
		if len(img.shape) == 2:
			img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
			img = cv2.resize(img, (512, 512),cv2.INTER_AREA)
		height, width, byteValue = img.shape
		byteValue = byteValue * width
		timg = img.copy()
		image = QtGui.QImage(timg.data, width, height,byteValue, QtGui.QImage.Format_RGB888)
		label.setPixmap(QtGui.QPixmap(image).scaled(label.size(),aspectMode=QtCore.Qt.KeepAspectRatio))		
		
		
		""" visualize function from
		https://github.com/BVLC/caffe/blob/master/examples/00-classification.ipynb
		"""
	def vis_square(self, data):
		"""Take an array of shape (n, height, width) or (n, height, width, 3)
		   and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)"""

		print "Data Shape : ", data.shape

		# normalize data for display
		data = (data - data.min()) / (data.max() - data.min())

		# force the number of filters to be square
		n = int(np.ceil(np.sqrt(data.shape[0])))
		padding = (((0, n ** 2 - data.shape[0]),
					(0, 1), (0, 1))  # add some space between filters
				   + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
		data = np.pad(data, padding, mode='constant', constant_values=0)  # pad with ones (white)

		# tile the filters into an image
		data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
		data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
		
		# show at display
		#print 'Data shape : ', data.shape , len(data.shape)
		img = 255 * data
		img = cv2.resize(img, (512, 512))
		img = np.array(img, dtype='uint8')
		img_c = cv2.applyColorMap(img, cv2.COLORMAP_JET)
		height, width, byteValue = img_c.shape
		byteValue = byteValue * width
		self.image = QtGui.QImage(img_c.data, width, height, byteValue, QtGui.QImage.Format_RGB888)
		self.ui.labelDisplay.setPixmap(QtGui.QPixmap(self.image))			


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    monitor = caffeNetViewerForm()
    monitor.show()
    sys.exit(app.exec_())

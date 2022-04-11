#import jinja2
from fileinput import filename
from turtle import pos
import numpy as np
import argparse
import torch
import torch.nn as nn
import random
from PyQt5.QtWidgets import QApplication, QFileDialog, QMainWindow
import sys
import vtk
from PyQt5 import QtCore, QtGui, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from gui import Ui_MainWindow
import math
from sklearn.cluster import DBSCAN
import open3d as o3d
import os
from pipeline.model_rotation import DGCNN_semseg_rotate_conv
########### used to transfer the rgb to r g b###########3
from struct import pack,unpack

cam_to_base_transform = [[ 6.3758686e-02 ,9.2318553e-01,-3.7902945e-01 ,4.5398907e+01],
 [ 9.8811066e-01,-5.1557920e-03 ,1.5365793e-01,-7.5876160e+02],
 [ 1.3990058e-01,-3.8432005e-01,-9.1253817e-01 ,9.6543054e+02],
 [ 0.0000000e+00 ,0.0000000e+00 ,0.0000000e+00 ,1.0000000e+00]]

color_map={"back_ground":[0,0,128],
           "cover":[0,100,0],
           "gear_container":[0,255,0],
           "charger":[255,255,0],
           "bottom":[255,165,0],
           "bolts":[255,0,0]}

def Read_PCD(file_path):

    pcd = o3d.io.read_point_cloud(file_path)
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    return np.concatenate([points, colors], axis=-1)



class Mywindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super(Mywindow, self).__init__()
        # pcd=Read_PCD("/home/bi/study/thesis/pyqt/test.pcd")
        self.fileName=""
        self.points_to_model=[]
        self.setupUi(self)
        self.setWindowTitle('Visialization of result')

        self.frame = QtWidgets.QFrame()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)     #give a QVTK a Qt framework
        self.formLayout_1.addWidget(self.vtkWidget)                 #connect the vtkWidget with a formlayout.
        self.ren = vtk.vtkRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()
        self.iren.Initialize()
        self.ren.SetBackground(192/255,192/255,192/255)
        self.ren.ResetCamera() 

        self.frame__ = QtWidgets.QFrame()
        self.vtkWidget__ = QVTKRenderWindowInteractor(self.frame__)     #give a QVTK a Qt framework
        self.formLayout_2.addWidget(self.vtkWidget__)                 #connect the vtkWidget with a formlayout.
        self.ren__ = vtk.vtkRenderer()
        self.vtkWidget__.GetRenderWindow().AddRenderer(self.ren__)
        self.iren__ = self.vtkWidget__.GetRenderWindow().GetInteractor()
        self.iren__.Initialize()
        self.ren__.SetBackground(192/255,192/255,192/255)
        self.ren__.ResetCamera() 

        self.action_load.triggered.connect(self.load)
        self.action_display_cuboid_pc.triggered.connect(self.cuboid_display)
        self.action_display_original_pc.triggered.connect(self.display_original)
        self.action_cubiod_prediction.triggered.connect(self.predict)
        self.action_to_robote_coordinates.triggered.connect(self.transfer_to_borot_coordinate)
        self.action_bolts_position.triggered.connect(self.filter_bolts)
        self.action_motor_orientation.triggered.connect(self.estimate)
        self.pushButton.clicked.connect(self.in_one)
        self.show()



    def load(self):
        self.ren.Clear()
        self.ren.NewInstance()
        # Create source
        self.fileName_previous=self.fileName
        self.label.setText("loading the file")
        self.label.adjustSize()
        self.label.repaint()
        self.fileName = QFileDialog.getOpenFileName(self,caption="choose the file you want to predict",filter="*.pcd *.ply")[0]

        #cloud_=Read_PCD(fileName)
        if self.fileName_previous!=self.fileName:
          if self.fileName_previous!="":
            self.ren.RemoveActor(self.actor)
          self.cloud=Read_PCD(self.fileName)
          #self.cloud = pcl.load_XYZRGB(self.fileName)
          self.judge_cut_again="yes"
          self.judge_predict_again="yes"
          self.trans_again="yes"
          self.search_bolts_again="yes"
          self.estimate_again="yes"
        self.num_points=len(self.cloud)

        # create the geometry of a point
        self.points = vtk.vtkPoints()
        #points.SetNumberOfPoints(size)

        #create the topology of the point
        self.vertices=vtk.vtkCellArray()

        #Setup colors
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")


        ############display the points
        if  self.fileName_previous!=self.fileName:
          self.points_to_model=[]
          for i in range(self.num_points):
              dp = self.cloud[i]
              # if dp[2]<0 or dp[2]>800:
              #   continue
              #self.points_to_model.append([dp[0], dp[1], dp[2],dp[3]])
              id=self.points.InsertNextPoint(dp[0], dp[1], dp[2])
              self.vertices.InsertNextCell(1)
              self.vertices.InsertCellPoint(id)

              r=int(self.cloud[i][3]*255)
              g=int(self.cloud[i][4]*255)
              b=int(self.cloud[i][5]*255)
              self.points_to_model.append([dp[0], dp[1], dp[2],r,g,b])
              self.Colors.InsertNextTuple3(r, g, b)
        else:
          for i in range(self.num_points):
              id=self.points.InsertNextPoint(self.points_to_model[i][0], self.points_to_model[i][1], self.points_to_model[i][2])
              self.vertices.InsertNextCell(1)
              self.vertices.InsertCellPoint(id)
              self.Colors.InsertNextTuple3(self.points_to_model[i][3], self.points_to_model[i][4], self.points_to_model[i][5])

        num_points=str(self.num_points)
        self.label_3.setText(num_points)
        self.label_3.adjustSize()
        self.label.setText("original pc")
        self.label.adjustSize()

        ##VTK color representation
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetVerts(self.vertices)
        self.polydata.GetPointData().SetScalars(self.Colors)
        self.polydata.Modified()
        
 
        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()
 
        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())
 
        #Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()  
        


    def display_original(self):
        self.label.setText("loading original pc")
        self.label.adjustSize()
        self.label.repaint()
        self.ren.RemoveActor(self.actor)
        self.num_points=self.cloud.size

        # create the geometry of a point
        self.points = vtk.vtkPoints()
        #points.SetNumberOfPoints(size)

        #create the topology of the point
        self.vertices=vtk.vtkCellArray()

        # Setup colors
        self.Colors = vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        ############display the points
        for i in range(self.num_points):
            id=self.points.InsertNextPoint(self.points_to_model[i][0], self.points_to_model[i][1], self.points_to_model[i][2])
            self.vertices.InsertNextCell(1)
            self.vertices.InsertCellPoint(id)
            self.Colors.InsertNextTuple3(self.points_to_model[i][3], self.points_to_model[i][4], self.points_to_model[i][5])

        num_points=str(self.num_points)
        self.label_3.setText(num_points)
        self.label_3.adjustSize()
        self.label.setText("original pc")
        self.label.adjustSize()

        ##VTK color representation
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetVerts(self.vertices)
        self.polydata.GetPointData().SetScalars(self.Colors)
        self.polydata.Modified()
        
 
        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()
 
        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())
 
        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)
 
        self.ren.AddActor(self.actor)
        self.ren.ResetCamera() 



    def cut_cuboid(self):
        # self.label.setText("cutting the cuboid")
        # self.label.adjustSize()
        # ########### patch the cloud popints
        self.label.setText("start to cut cuboid")
        self.label.adjustSize()
        self.label.repaint()
        self.points_to_model=np.array(self.points_to_model)
        self.motor_scene,self.residual_scene=cut_motor(self.points_to_model)
        current_points_size=self.motor_scene.shape[0]
        if current_points_size % 2048 !=0:
          num_add_points=2048-(current_points_size % 2048)
          choice=np.random.choice(current_points_size,num_add_points,replace=True)
          add_points=self.motor_scene[choice,:]
          self.motor_points=np.vstack((self.motor_scene,add_points))
          np.random.shuffle(self.motor_points)
        else:
          self.motor_points=self.motor_scene
          np.random.shuffle(self.motor_points)
        self.judge_cut_again=self.fileName



    def cuboid_display(self):
        
        if self.judge_cut_again!=self.fileName:
          self.cut_cuboid()
        self.judge_cut_again=self.fileName
        self.ren.RemoveActor(self.actor)
        # create the geometry of a point
        self.points = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        #create the topology of the point
        self.vertices=vtk.vtkCellArray()

        # Setup colors
        self.Colors= vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        # for i in range(self.motor_points.shape[0]):
        #   dp = self.motor_points[i]
        #   id=self.points.InsertNextPoint(dp[0], dp[1], dp[2])
        #   self.vertices.InsertNextCell(1)
        #   self.vertices.InsertCellPoint(id)
        #   inter=unpack('i',pack('f',self.motor_points[i][3]))[0]
        #   r=(inter>>16) & 0x0000ff
        #   g=(inter>>8) & 0x0000ff
        #   b=(inter>>0) & 0x0000ff
        #   self.Colors.InsertNextTuple3(r, g, b)

        for i in range(self.motor_points.shape[0]):
          id=self.points.InsertNextPoint(self.motor_points[i][0], self.motor_points[i][1], self.motor_points[i][2])
          self.vertices.InsertNextCell(1)
          self.vertices.InsertCellPoint(id)
          self.Colors.InsertNextTuple3(self.motor_points[i][3], self.motor_points[i][4], self.motor_points[i][5])

        self.label.setText("cut cuboid")
        self.label.adjustSize()
        num_points=str(self.motor_points.shape[0])
        self.label_3.setText(num_points)
        self.label_3.adjustSize()

        ##VTK color representation
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetVerts(self.vertices)
        self.polydata.GetPointData().SetScalars(self.Colors)
        self.polydata.Modified()
        
 
        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()
 
        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())
 
        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()  



    def cuboid_display__(self):
        
        if self.judge_cut_again!=self.fileName:
          self.cut_cuboid()
        self.judge_cut_again=self.fileName
        self.ren.RemoveActor(self.actor)
        # create the geometry of a point
        self.points = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        #create the topology of the point
        self.vertices=vtk.vtkCellArray()

        # Setup colors
        self.Colors= vtk.vtkUnsignedCharArray()
        self.Colors.SetNumberOfComponents(3)
        self.Colors.SetName("Colors")

        # for i in range(self.motor_points.shape[0]):
        #   dp = self.motor_points[i]
        #   id=self.points.InsertNextPoint(dp[0], dp[1], dp[2])
        #   self.vertices.InsertNextCell(1)
        #   self.vertices.InsertCellPoint(id)
        #   inter=unpack('i',pack('f',self.motor_points[i][3]))[0]
        #   r=(inter>>16) & 0x0000ff
        #   g=(inter>>8) & 0x0000ff
        #   b=(inter>>0) & 0x0000ff
        #   self.Colors.InsertNextTuple3(r, g, b)

        for i in range(self.motor_points.shape[0]):
          id=self.points.InsertNextPoint(self.motor_points[i][0], self.motor_points[i][1], self.motor_points[i][2])
          self.vertices.InsertNextCell(1)
          self.vertices.InsertCellPoint(id)
          self.Colors.InsertNextTuple3(self.motor_points[i][3], self.motor_points[i][4], self.motor_points[i][5])

        self.num_points=str(self.motor_points.shape[0])

        ##VTK color representation
        self.polydata = vtk.vtkPolyData()
        self.polydata.SetPoints(self.points)
        self.polydata.SetVerts(self.vertices)
        self.polydata.GetPointData().SetScalars(self.Colors)
        self.polydata.Modified()
        
 
        self.glyphFilter = vtk.vtkVertexGlyphFilter()
        self.glyphFilter.SetInputData(self.polydata)
        self.glyphFilter.Update()
 
        self.dataMapper = vtk.vtkPolyDataMapper()
        self.dataMapper.SetInputConnection(self.glyphFilter.GetOutputPort())
 
        # Create an actor
        self.actor = vtk.vtkActor()
        self.actor.SetMapper(self.dataMapper)

        self.ren.AddActor(self.actor)
        self.ren.ResetCamera()  



    def predict__(self):
        if self.judge_predict_again!=self.fileName:
          self.motor_points_forecast=predict(self.motor_points[:,0:3])
        self.judge_predict_again=self.fileName
      


    def predict(self):
        if self.judge_cut_again!=self.fileName:
          self.cut_cuboid()
          self.judge_cut_again=self.fileName
          self.cuboid_display__()


        # # create the geometry of a point
        # self.points__ = vtk.vtkPoints()
        # # points.SetNumberOfPoints(size)

        # #create the topology of the point
        # self.vertices__=vtk.vtkCellArray()

        # # Setup colors
        # self.Colors__ = vtk.vtkUnsignedCharArray()
        # self.Colors__.SetNumberOfComponents(3)
        # self.Colors__.SetName("Colors__")

        if self.judge_predict_again!=self.fileName:
          self.label_2.setText("send to architecture to predict")
          self.label_2.adjustSize()
          self.label_2.repaint()
          self.motor_points_forecast=predict(self.motor_points[:,0:3])
          self.label_3.setText(self.num_points)
          self.label_3.adjustSize()
        self.judge_predict_again=self.fileName
        self.transfer_to_borot_coordinate()
        self.label.setText("cut cuboid")
        self.label.adjustSize()



    def transfer_to_borot_coordinate__(self):
        if self.trans_again!=self.fileName:
          self.motor_points_forecast_in_robot=np.random.rand(self.motor_points_forecast.shape[0],4)
          self.motor_points_forecast_in_robot[:,0:3]=np.array(camera_to_base(self.motor_points_forecast[:,0:3]))
          self.motor_points_forecast_in_robot[:,3]=np.array(self.motor_points_forecast[:,3])
        self.trans_again=self.fileName



    def transfer_to_borot_coordinate(self):
        self.label_2.setText("start to visiualise result")
        self.label_2.adjustSize()
        self.label_2.repaint()
        if self.fileName_previous!="":
            self.ren__.RemoveActor(self.actor__)
        self.fileName_previous="been precessed"
        # create the geometry of a point
        self.points__ = vtk.vtkPoints()
        # points.SetNumberOfPoints(size)

        #create the topology of the point
        self.vertices__=vtk.vtkCellArray()

        # Setup colors
        self.Colors__ = vtk.vtkUnsignedCharArray()
        self.Colors__.SetNumberOfComponents(3)
        self.Colors__.SetName("Colors__")
        if self.trans_again!=self.fileName:
          self.motor_points_forecast_in_robot=np.random.rand(self.motor_points_forecast.shape[0],4)
          self.motor_points_forecast_in_robot[:,0:3]=np.array(camera_to_base(self.motor_points_forecast[:,0:3]))
          self.motor_points_forecast_in_robot[:,3]=np.array(self.motor_points_forecast[:,3])
        self.trans_again=self.fileName
        self.filter_bolts_no_display()
        for i in range(self.motor_points_forecast_in_robot.shape[0]):
          dp = self.motor_points_forecast_in_robot[i]
          id=self.points__.InsertNextPoint(dp[0], dp[1], dp[2])
          self.vertices__.InsertNextCell(1)
          self.vertices__.InsertCellPoint(id)
          if dp[3]==0:
            r=color_map["back_ground"][0]
            g=color_map["back_ground"][1]
            b=color_map["back_ground"][2]
          elif dp[3]==1:
            r=color_map["cover"][0]
            g=color_map["cover"][1]
            b=color_map["cover"][2]
          elif dp[3]==2:
            r=color_map["gear_container"][0]
            g=color_map["gear_container"][1]
            b=color_map["gear_container"][2]
          elif dp[3]==3:
            r=color_map["charger"][0]
            g=color_map["charger"][1]
            b=color_map["charger"][2]
          elif dp[3]==4:
            r=color_map["bottom"][0]
            g=color_map["bottom"][1]
            b=color_map["bottom"][2]
          else:
            r=color_map["bolts"][0]
            g=color_map["bolts"][1]
            b=color_map["bolts"][2]
          self.Colors__.InsertNextTuple3(r, g, b)
        ## VTK color representation   22222222
        polydata__ = vtk.vtkPolyData()
        polydata__.SetPoints(self.points__)
        polydata__.SetVerts(self.vertices__)
        polydata__.GetPointData().SetScalars(self.Colors__)
        polydata__.Modified()
        

        glyphFilter__ = vtk.vtkVertexGlyphFilter()
        glyphFilter__.SetInputData(polydata__)
        glyphFilter__.Update()

        dataMapper__ = vtk.vtkPolyDataMapper()
        dataMapper__.SetInputConnection(glyphFilter__.GetOutputPort())

        # Create an actor
        self.actor__ = vtk.vtkActor()
        self.actor__.SetMapper(dataMapper__)

        self.ren__.AddActor(self.actor__)
        self.ren__.ResetCamera()

        
        self.label_2.setText("predicted result for cuboid")
        self.label_2.adjustSize()



    def estimate__(self):

      if self.estimate_again!=self.fileName:
        self.covers,self.normal=find_covers(self.motor_points_forecast_in_robot)
      self.estimate_again=self.fileName



    def estimate(self):
      self.ren__.RemoveActor(self.actor__)
      self.label_6.setText("")
      self.label_6.adjustSize()
      self.label_11.setText("")
      self.label_11.adjustSize()
      self.label_5.setText("")
      self.label_5.adjustSize()
      self.label_12.setText("")
      self.label_12.adjustSize()
      self.label_7.setText("")
      self.label_7.adjustSize()
      self.label_13.setText("")
      self.label_13.adjustSize()
      self.label_8.setText("")
      self.label_8.adjustSize()
      self.label_14.setText("")
      self.label_14.adjustSize()
      self.label_9.setText("")
      self.label_9.adjustSize()
      self.label_15.setText("")
      self.label_15.adjustSize()
      self.label_10.setText("")
      self.label_10.adjustSize()
      self.label_16.setText("")
      self.label_16.adjustSize()
      #self.covers,self.Rx_Ry_Rz=find_covers(self.motor_points_forecast)
      if self.estimate_again!=self.fileName:
        self.covers,self.normal=find_covers(self.motor_points_forecast_in_robot)
      self.estimate_again==self.fileName
      self.label_2.setText("estimated orientation of bolts")
      self.label_2.adjustSize()
      self.label_6.setText("Rx_Ry_Rz:")
      self.label_6.adjustSize()
      self.label_11.setText(str(self.normal))
      self.label_11.adjustSize()
      # create the geometry of a point
      self.points__ = vtk.vtkPoints()
      # points.SetNumberOfPoints(size)

      #create the topology of the point
      self.vertices__=vtk.vtkCellArray()

      # Setup colors
      self.Colors__ = vtk.vtkUnsignedCharArray()
      self.Colors__.SetNumberOfComponents(3)
      self.Colors__.SetName("Colors__")
      for i in range(self.covers.shape[0]):
        dp = self.covers[i]
        id=self.points__.InsertNextPoint(dp[0], dp[1], dp[2])
        self.vertices__.InsertNextCell(1)
        self.vertices__.InsertCellPoint(id)
        self.Colors__.InsertNextTuple3(255, 178, 102)

      ## VTK color representation   22222222
      polydata__ = vtk.vtkPolyData()
      polydata__.SetPoints(self.points__)
      polydata__.SetVerts(self.vertices__)
      polydata__.GetPointData().SetScalars(self.Colors__)
      polydata__.Modified()
      

      glyphFilter__ = vtk.vtkVertexGlyphFilter()
      glyphFilter__.SetInputData(polydata__)
      glyphFilter__.Update()

      dataMapper__ = vtk.vtkPolyDataMapper()
      dataMapper__.SetInputConnection(glyphFilter__.GetOutputPort())

      # Create an actor
      self.actor__ = vtk.vtkActor()
      self.actor__.SetMapper(dataMapper__)

      self.ren__.AddActor(self.actor__)
      self.ren__.ResetCamera()



    def filter_bolts(self):
      self.ren__.RemoveActor(self.actor__)
      if self.search_bolts_again!=self.fileName:
        self.positions_bolts,self.num_bolts,self.bolts=find_bolts(self.motor_points_forecast_in_robot, eps=2.5, min_points=50)
      self.search_bolts_again=self.fileName

      self.label_2.setText("estimated positions of bolts")
      self.label_2.adjustSize()
      self.label_6.setText("num_bolts:")
      self.label_6.adjustSize()
      self.label_11.setText(str(self.num_bolts))
      self.label_11.adjustSize()
      self.label_5.setText("")
      self.label_5.adjustSize()
      self.label_12.setText("")
      self.label_12.adjustSize()
      self.label_7.setText("")
      self.label_7.adjustSize()
      self.label_13.setText("")
      self.label_13.adjustSize()
      self.label_8.setText("")
      self.label_8.adjustSize()
      self.label_14.setText("")
      self.label_14.adjustSize()
      self.label_9.setText("")
      self.label_9.adjustSize()
      self.label_15.setText("")
      self.label_15.adjustSize()
      self.label_10.setText("")
      self.label_10.adjustSize()
      self.label_16.setText("")
      self.label_16.adjustSize()
      if self.num_bolts==1:
        self.label_5.setText("bolts 1:")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
      elif self.num_bolts==2:
        self.label_5.setText("bolts 1:")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
        self.label_7.setText("bolts 2:")
        self.label_7.adjustSize()
        self.label_13.setText(str(self.positions_bolts[1]))
        self.label_13.adjustSize()
      elif self.num_bolts==3:
        self.label_5.setText("bolts 1:")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
        self.label_7.setText("bolts 2:")
        self.label_7.adjustSize()
        self.label_13.setText(str(self.positions_bolts[1]))
        self.label_13.adjustSize()
        self.label_8.setText("bolts 3:")
        self.label_8.adjustSize()
        self.label_14.setText(str(self.positions_bolts[2]))
        self.label_14.adjustSize()
      elif self.num_bolts==4:
        self.label_5.setText("bolts 1(side):")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
        self.label_7.setText("bolts 2:")
        self.label_7.adjustSize()
        self.label_13.setText(str(self.positions_bolts[1]))
        self.label_13.adjustSize()
        self.label_8.setText("bolts 3:")
        self.label_8.adjustSize()
        self.label_14.setText(str(self.positions_bolts[2]))
        self.label_14.adjustSize()
        self.label_9.setText("bolts 4:")
        self.label_9.adjustSize()
        self.label_15.setText(str(self.positions_bolts[3]))
        self.label_15.adjustSize()
      else:
        self.label_5.setText("bolts 1(side):")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
        self.label_7.setText("bolts 2:")
        self.label_7.adjustSize()
        self.label_13.setText(str(self.positions_bolts[1]))
        self.label_13.adjustSize()
        self.label_8.setText("bolts 3:")
        self.label_8.adjustSize()
        self.label_14.setText(str(self.positions_bolts[2]))
        self.label_14.adjustSize()
        self.label_9.setText("bolts 4:")
        self.label_9.adjustSize()
        self.label_15.setText(str(self.positions_bolts[3]))
        self.label_15.adjustSize()
        self.label_10.setText("bolts 5:")
        self.label_10.adjustSize()
        self.label_16.setText(str(self.positions_bolts[4]))
        self.label_16.adjustSize()
      # create the geometry of a point
      self.points__ = vtk.vtkPoints()
      # points.SetNumberOfPoints(size)

      #create the topology of the point
      self.vertices__=vtk.vtkCellArray()

      # Setup colors
      self.Colors__ = vtk.vtkUnsignedCharArray()
      self.Colors__.SetNumberOfComponents(3)
      self.Colors__.SetName("Colors__")
      for i in range(self.bolts.shape[0]):
        dp = self.bolts[i]
        id=self.points__.InsertNextPoint(dp[0], dp[1], dp[2])
        self.vertices__.InsertNextCell(1)
        self.vertices__.InsertCellPoint(id)
        self.Colors__.InsertNextTuple3(255, 0, 0)

      ## VTK color representation   22222222
      polydata__ = vtk.vtkPolyData()
      polydata__.SetPoints(self.points__)
      polydata__.SetVerts(self.vertices__)
      polydata__.GetPointData().SetScalars(self.Colors__)
      polydata__.Modified()
      

      glyphFilter__ = vtk.vtkVertexGlyphFilter()
      glyphFilter__.SetInputData(polydata__)
      glyphFilter__.Update()

      dataMapper__ = vtk.vtkPolyDataMapper()
      dataMapper__.SetInputConnection(glyphFilter__.GetOutputPort())

      # Create an actor
      self.actor__ = vtk.vtkActor()
      self.actor__.SetMapper(dataMapper__)

      self.ren__.AddActor(self.actor__)
      self.ren__.ResetCamera()



    def filter_bolts_no_display(self):
      if self.search_bolts_again!=self.fileName:
        self.positions_bolts,self.num_bolts,self.bolts=find_bolts(self.motor_points_forecast_in_robot, eps=2.5, min_points=50)
      self.search_bolts_again=self.fileName

      self.label_2.setText("estimated positions of bolts")
      self.label_2.adjustSize()
      self.label_6.setText("num_bolts:")
      self.label_6.adjustSize()
      self.label_11.setText(str(self.num_bolts))
      self.label_11.adjustSize()
      self.label_5.setText("")
      self.label_5.adjustSize()
      self.label_12.setText("")
      self.label_12.adjustSize()
      self.label_7.setText("")
      self.label_7.adjustSize()
      self.label_13.setText("")
      self.label_13.adjustSize()
      self.label_8.setText("")
      self.label_8.adjustSize()
      self.label_14.setText("")
      self.label_14.adjustSize()
      self.label_9.setText("")
      self.label_9.adjustSize()
      self.label_15.setText("")
      self.label_15.adjustSize()
      self.label_10.setText("")
      self.label_10.adjustSize()
      self.label_16.setText("")
      self.label_16.adjustSize()
      if self.num_bolts==1:
        self.label_5.setText("bolts 1:")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
      elif self.num_bolts==2:
        self.label_5.setText("bolts 1:")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
        self.label_7.setText("bolts 2:")
        self.label_7.adjustSize()
        self.label_13.setText(str(self.positions_bolts[1]))
        self.label_13.adjustSize()
      elif self.num_bolts==3:
        self.label_5.setText("bolts 1:")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
        self.label_7.setText("bolts 2:")
        self.label_7.adjustSize()
        self.label_13.setText(str(self.positions_bolts[1]))
        self.label_13.adjustSize()
        self.label_8.setText("bolts 3:")
        self.label_8.adjustSize()
        self.label_14.setText(str(self.positions_bolts[2]))
        self.label_14.adjustSize()
      elif self.num_bolts==4:
        self.label_5.setText("bolts 1(side):")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
        self.label_7.setText("bolts 2:")
        self.label_7.adjustSize()
        self.label_13.setText(str(self.positions_bolts[1]))
        self.label_13.adjustSize()
        self.label_8.setText("bolts 3:")
        self.label_8.adjustSize()
        self.label_14.setText(str(self.positions_bolts[2]))
        self.label_14.adjustSize()
        self.label_9.setText("bolts 4:")
        self.label_9.adjustSize()
        self.label_15.setText(str(self.positions_bolts[3]))
        self.label_15.adjustSize()
      else:
        self.label_5.setText("bolts 1(side):")
        self.label_5.adjustSize()
        self.label_12.setText(str(self.positions_bolts[0]))
        self.label_12.adjustSize()
        self.label_7.setText("bolts 2:")
        self.label_7.adjustSize()
        self.label_13.setText(str(self.positions_bolts[1]))
        self.label_13.adjustSize()
        self.label_8.setText("bolts 3:")
        self.label_8.adjustSize()
        self.label_14.setText(str(self.positions_bolts[2]))
        self.label_14.adjustSize()
        self.label_9.setText("bolts 4:")
        self.label_9.adjustSize()
        self.label_15.setText(str(self.positions_bolts[3]))
        self.label_15.adjustSize()
        self.label_10.setText("bolts 5:")
        self.label_10.adjustSize()
        self.label_16.setText(str(self.positions_bolts[4]))
        self.label_16.adjustSize()


    def in_one(self):
      self.predict()
      self.estimate__()



def predict(points):
  parser = argparse.ArgumentParser(description='Point Cloud Semantic Segmentation')
  parser.add_argument('--dropout', type=float, default=0.5,
                      help='dropout rate')
  parser.add_argument('--emb_dims', type=int, default=1024, metavar='N',
                      help='Dimension of embeddings')
  parser.add_argument('--k', type=int, default=20, metavar='N',
                      help='Num of nearest neighbors to use')
  args = parser.parse_args()
  device = torch.device("cuda")
  model = DGCNN_semseg_rotate_conv(args).to(device)
  model = nn.DataParallel(model)
  filename=os. getcwd()
  filename=filename+"/pipeline/best.pth"
  loaded_model = torch.load(filename)
  model.load_state_dict(loaded_model['model_state_dict'])
  n_points=2048
  test_batchsize=14
  num_points_size=points.shape[0]
  result=np.zeros((num_points_size,4),dtype=float)
  with torch.no_grad(): 
    if num_points_size / n_points % test_batchsize ==0:
      num_batch=num_points_size / n_points / test_batchsize
      for i in range(num_batch):
        sequence=np.arange(n_points*test_batchsize)
        sequence=i*n_points*test_batchsize+sequence
        current_points_batch=points[sequence,:]
        current_points_batch=current_points_batch.reshape(test_batchsize,n_points,-1)
        current_points_batch=torch.tensor(current_points_batch)
        data= current_points_batch.to(device)
        data=normalize_data(data)
        data,GT=rotate_per_batch(data)
        data = data.permute(0, 2, 1)
        for_alignment=1
        seg_pred,_= model(data,for_alignment)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        seg_pred = seg_pred.contiguous().view(-1, 6)   # (batch_size*num_points , num_class)
        pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  #array(batch_size*num_points)
        result[sequence,0:3]=points[sequence,:]
        result[sequence,3]=pred_choice[:,]
    else:
      num_batch=math.floor(num_points_size / n_points / test_batchsize)
      for i in range(num_batch):
        sequence=np.arange(n_points*test_batchsize)
        sequence=i*n_points*test_batchsize+sequence
        current_points_batch=points[sequence,:]
        current_points_batch=current_points_batch.reshape(test_batchsize,n_points,-1)
        current_points_batch=torch.tensor(current_points_batch)
        data= current_points_batch.to(device)
        data=normalize_data(data)
        data,GT=rotate_per_batch(data)
        data = data.permute(0, 2, 1)
        for_alignment=1
        seg_pred,_= model(data,for_alignment)
        seg_pred = seg_pred.permute(0, 2, 1).contiguous()
        seg_pred = seg_pred.contiguous().view(-1, 6)   # (batch_size*num_points , num_class)
        pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  #array(batch_size*num_points)
        result[sequence,0:3]=points[sequence,:]
        result[sequence,3]=pred_choice[:,]
      num_batch_residuel=int(num_points_size / n_points % test_batchsize)
      sequence=np.arange(n_points*num_batch_residuel)+int(num_batch*n_points*test_batchsize)
      current_points_batch=points[sequence,:]
      current_points_batch=current_points_batch.reshape(num_batch_residuel,n_points,-1)
      current_points_batch=torch.tensor(current_points_batch)
      data= current_points_batch.to(device)
      data=normalize_data(data)
      data,GT=rotate_per_batch(data)
      data = data.permute(0, 2, 1)
      for_alignment=1
      seg_pred,_= model(data,for_alignment)
      seg_pred = seg_pred.permute(0, 2, 1).contiguous()
      seg_pred = seg_pred.contiguous().view(-1, 6)   # (batch_size*num_points , num_class)
      pred_choice = seg_pred.cpu().data.max(1)[1].numpy()  #array(batch_size*num_points)
      result[sequence,0:3]=points[sequence,:]
      result[sequence,3]=pred_choice[:,]
    return result



def cut_motor(whole_scene):
    x_far=-360
    x_close=-210
    y_far=-940
    y_close=-640
    z_down=0
    z_up=250
    Corners = [(x_close,y_far,z_up), (x_close,y_close,z_up), (x_far,y_close,z_up), (x_far,y_far,z_up), (x_close,y_far,z_down), (x_close,y_close,z_down), (x_far,y_close,z_down), (x_far,y_far,z_down)]
    #Corners = [(35,880,300), (35,1150,300), (-150,1150,300), (-150,880,300), (35,880,50), (35,1150,50), (-150,1150,50), (-150,880,50)]
    cor_inCam = []
    for corner in Corners:
        cor_inCam_point = base_to_camera(np.array(corner))
        cor_inCam.append(np.squeeze(np.array(cor_inCam_point)))

    panel_1 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[2])
    panel_2 = get_panel(cor_inCam[5], cor_inCam[6], cor_inCam[4])
    panel_3 = get_panel(cor_inCam[0], cor_inCam[3], cor_inCam[4])
    panel_4 = get_panel(cor_inCam[1], cor_inCam[2], cor_inCam[5])
    panel_5 = get_panel(cor_inCam[0], cor_inCam[1], cor_inCam[4])
    panel_6 = get_panel(cor_inCam[2], cor_inCam[3], cor_inCam[6])
    panel_list = {'panel_up':panel_1, 'panel_bot':panel_2, 'panel_front':panel_3, 'panel_behind':panel_4, 'panel_right':panel_5, 'panel_left':panel_6}

    patch_motor = []
    residual_scene=[]
    for point in whole_scene:
        point_cor = (point[0], point[1], point[2])
        if set_Boundingbox(panel_list, point_cor):
            patch_motor.append(point)
        else:
            residual_scene.append(point)
    return np.array(patch_motor),np.array(residual_scene)



def get_panel(point_1, point_2, point_3):

    x1 = point_1[0]
    y1 = point_1[1]
    z1 = point_1[2]

    x2 = point_2[0]
    y2 = point_2[1]
    z2 = point_2[2] 

    x3 = point_3[0]
    y3 = point_3[1]
    z3 = point_3[2]
    
    a = (y2-y1)*(z3-z1) - (y3-y1)*(z2-z1)
    b = (z2-z1)*(x3-x1) - (z3-z1)*(x2-x1)
    c = (x2-x1)*(y3-y1) - (x3-x1)*(y2-y1)
    d = 0 - (a*x1 + b*y1 + c*z1)

    return (a, b, c, d)



def set_Boundingbox(panel_list, point_cor):

    if panel_list['panel_up'][0]*point_cor[0] + panel_list['panel_up'][1]*point_cor[1] + panel_list['panel_up'][2]*point_cor[2] + panel_list['panel_up'][3] <= 0 :   # panel 1
        if panel_list['panel_bot'][0]*point_cor[0] + panel_list['panel_bot'][1]*point_cor[1] + panel_list['panel_bot'][2]*point_cor[2] + panel_list['panel_bot'][3] >= 0 : # panel 2
            if panel_list['panel_front'][0]*point_cor[0] + panel_list['panel_front'][1]*point_cor[1] + panel_list['panel_front'][2]*point_cor[2] + panel_list['panel_front'][3] <= 0 : # panel 3
                if panel_list['panel_behind'][0]*point_cor[0] + panel_list['panel_behind'][1]*point_cor[1] + panel_list['panel_behind'][2]*point_cor[2] + panel_list['panel_behind'][3] >= 0 : # panel 4
                    if panel_list['panel_right'][0]*point_cor[0] + panel_list['panel_right'][1]*point_cor[1] + panel_list['panel_right'][2]*point_cor[2] + panel_list['panel_right'][3] >= 0 : #panel 5
                        if panel_list['panel_left'][0]*point_cor[0] + panel_list['panel_left'][1]*point_cor[1] + panel_list['panel_left'][2]*point_cor[2] + panel_list['panel_left'][3] >= 0 : # panel 6

                            return True
    return False



def base_to_camera(xyz, calc_angle=False):
    '''
    now do the base to camera transform
    '''

        # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

        # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]

    cam_to_base_transform_ = np.matrix(cam_to_base_transform)
    base_to_cam_transform = cam_to_base_transform_.I
    xyz_transformed2 = np.matmul(base_to_cam_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]



def camera_to_base(xyz, calc_angle=False):
    '''
    '''
        # squeeze the first two dimensions
    xyz_transformed2 = xyz.reshape(-1, 3)  # [N=X*Y, 3]

        # homogeneous transformation
    if calc_angle:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.zeros((xyz_transformed2.shape[0], 1))))  # [N, 4]
    else:
        xyz_transformed2 = np.hstack((xyz_transformed2, np.ones((xyz_transformed2.shape[0], 1))))  # [N, 4]


    xyz_transformed2 = np.matmul(cam_to_base_transform, xyz_transformed2.T).T  # [N, 4]

    return xyz_transformed2[:, :-1].reshape(xyz.shape)  # [X, Y, 3]


    
def normalize_data(batch_data):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    B, N, C = batch_data.shape
    for b in range(B):
        pc = batch_data[b]
        centroid = torch.mean(pc, dim=0,keepdim=True)
        pc = pc - centroid
        m = torch.max(torch.sqrt(torch.sum(pc ** 2, dim=1,keepdim=True)))
        pc = pc / m
        batch_data[b] = pc
    return batch_data



def rotate_per_batch(data,angle_clip=np.pi*1):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BXNx6 array, original batch of point clouds and point normals
        Return:
          BXNx3 array, rotated batch of point clouds
    """
    data=data.float()
    rotated_data = torch.zeros(data.shape, dtype=torch.float32)
    rotated_data = rotated_data.cuda()
    batch_size=data.shape[0]
    rotation_matrix=torch.zeros((batch_size,3,3),dtype=torch.float32).cuda()
    for k in range(data.shape[0]):
        angles=[]
        for i in range(3): 
            angles.append(random.uniform(-angle_clip,angle_clip))
        angles=np.array(angles)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        R=torch.from_numpy(R).float().cuda()
        rotated_data[k,:,:] = torch.matmul(data[k,:,:], R)
        rotation_matrix[k,:,:]=R
    return rotated_data,rotation_matrix



def find_bolts(seg_motor, eps, min_points):
    bolts = []
    for point in seg_motor:
        if point[3]==5  : bolts.append(point[0:3])
    bolts = np.asarray(bolts)
    model = DBSCAN(eps=eps, min_samples=min_points)
    yhat = model.fit_predict(bolts)  # genalize label based on index
    clusters = np.unique(yhat)
    noise = []
    clusters_new = []
    positions = []
    for i in clusters:
        noise.append(i) if np.sum(i == yhat) < 150 or i == -1 else clusters_new.append(i)
    flag=0
    bolts__=1
    for clu in clusters_new :
        row_ix = np.where(yhat == clu)
        if flag==0:
          bolts__=np.squeeze(np.array(bolts[row_ix, :3]))
          flag=1
        else:
          inter=np.squeeze(np.array(bolts[row_ix, :3]))
          bolts__=np.concatenate((bolts__,inter),axis=0)
        np.set_printoptions(precision=2)
        position = np.squeeze(np.mean(bolts[row_ix, :3], axis=1))
        positions.append(position)
    positions=np.array(positions)
    indexs=np.argsort(positions[:,1])
    positions=positions[indexs,:]
    
    return positions, len(clusters_new),bolts__



def find_covers(seg_motor):
    bottom = []
    for point in seg_motor:
        if point[3]==1  : bottom.append(point[0:3])
    filename=os. getcwd()
    filename=filename+"/cover.pcd"
    open3d_save_pcd(bottom,filename)
    pcd = o3d.io.read_point_cloud(filename)
    downpcd = pcd.voxel_down_sample(voxel_size=0.002)  # 下采样滤波，体素边长为0.002m
    downpcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamKNN(knn=20))  # 计算法线，只考虑邻域内的20个点
    nor=downpcd.normals
    normal=[]
    for ele in nor:
        normal.append(ele)
    normal=np.array(normal)
    model = DBSCAN(eps=0.02, min_samples=100)
    yhat = model.fit_predict(normal)  # genalize label based on index
    clusters = np.unique(yhat)
    noise = []
    clusters_new = []
    for i in clusters:
        noise.append(i) if np.sum(i == yhat) < 300 or i == -1 else clusters_new.append(i)
    for clu in clusters_new :
        row_ix = np.where(yhat == clu)
        normal = np.squeeze(np.mean(normal[row_ix, :3], axis=1))
    return np.array(bottom),normal



def open3d_save_pcd(pc,filename):
    sampled = np.asarray(pc)
    PointCloud_koordinate = sampled[:, 0:3]

    #visuell the point cloud
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(PointCloud_koordinate)
    o3d.io.write_point_cloud(filename, point_cloud, write_ascii=True)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = Mywindow()
    window.show()
    sys.exit(app.exec_())
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 09:02:45 2019

@author: Bruce Wu
"""

import os
import shutil
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray
import numpy as np
from scipy.spatial import distance_matrix
import sys
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import math


def GetCellIdsPointsFromCells(new_cells):
    '''
    new_cell_ids, new_points = GetCellIdsPointsFromCells(new_cells)
    new_cells: [n, 9] array, where n is the number of new cells
    new_cell_ids: [n, 3] array, where n is the number of new cells
    new_points: [m, 3] array, where m is the number of new points
    '''
    new_points = new_cells.reshape([int(new_cells.shape[0]*3), 3])
    new_points = np.unique(new_points, axis=0)
    new_cell_ids = np.zeros([new_cells.shape[0], 3], dtype='int64')
    
    for i_count in range(new_cells.shape[0]):
        counts0 = np.bincount(np.where(new_points==new_cells[i_count, 0:3])[0])
        counts1 = np.bincount(np.where(new_points==new_cells[i_count, 3:6])[0])
        counts2 = np.bincount(np.where(new_points==new_cells[i_count, 6:9])[0])
        new_cell_ids[i_count, 0] = np.argmax(counts0)
        new_cell_ids[i_count, 1] = np.argmax(counts1)
        new_cell_ids[i_count, 2] = np.argmax(counts2)
        
    return new_cell_ids, new_points


def GetEdgeValue(cell_ids, points):
    '''
    edges = GetEdgeValue(cell_ids, points)
    cell_ids: [n, 3] array, where n is the number of new cells
    points: [m, 3] array, where m is the number of new points
    '''
    edges = np.zeros([cell_ids.shape[0], 3])
    
    for i_count in range(cell_ids.shape[0]):
        v1 = points[cell_ids[i_count, 0], :] - points[cell_ids[i_count, 1], :]
        v2 = points[cell_ids[i_count, 1], :] - points[cell_ids[i_count, 2], :]
        v3 = points[cell_ids[i_count, 0], :] - points[cell_ids[i_count, 2], :]
        edges[i_count, 0] = np.linalg.norm(v1)
        edges[i_count, 1] = np.linalg.norm(v2)
        edges[i_count, 2] = np.linalg.norm(v3)
        
    return edges
    

def GetCellsFromOBJ(filename):
    '''
    cells, cell_ids, normals, points = GetCellsFromOBJ(stl_filename)
    cells: [n, 9] array, where n is the number of cells
    cell_ids: [n, 3] array, where n is the number of cells
    normals: [n, 3] array, where h is the number of cells
    points: [m, 3] array, where m is the number of points
    '''
    
    #vtk reader for cells and points
    reader = vtk.vtkOBJReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
        
    n_triangles = data.GetNumberOfCells()
    n_points = data.GetNumberOfPoints()
    mesh_triangles = np.zeros([n_triangles, 9], dtype='float32')
    mesh_triangle_ids = np.zeros([n_triangles, 3], dtype='int32')
    mesh_points = np.zeros([n_points, 3], dtype='float32')
    
    for i in range(n_triangles):
        mesh_triangles[i][0], mesh_triangles[i][1], mesh_triangles[i][2] = data.GetPoint(data.GetCell(i).GetPointId(0))
        mesh_triangles[i][3], mesh_triangles[i][4], mesh_triangles[i][5] = data.GetPoint(data.GetCell(i).GetPointId(1))
        mesh_triangles[i][6], mesh_triangles[i][7], mesh_triangles[i][8] = data.GetPoint(data.GetCell(i).GetPointId(2))
        mesh_triangle_ids[i][0] = data.GetCell(i).GetPointId(0)
        mesh_triangle_ids[i][1] = data.GetCell(i).GetPointId(1)
        mesh_triangle_ids[i][2] = data.GetCell(i).GetPointId(2)
        
    
    for i in range(n_points):
        mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)
        
    #normal
    v1 = np.zeros([n_triangles, 3], dtype='float32')
    v2 = np.zeros([n_triangles, 3], dtype='float32')
    v1[:, 0] = mesh_triangles[:, 0]-mesh_triangles[:, 3]
    v1[:, 1] = mesh_triangles[:, 1]-mesh_triangles[:, 4]
    v1[:, 2] = mesh_triangles[:, 2]-mesh_triangles[:, 5]
    v2[:, 0] = mesh_triangles[:, 3]-mesh_triangles[:, 6]
    v2[:, 1] = mesh_triangles[:, 4]-mesh_triangles[:, 7]
    v2[:, 2] = mesh_triangles[:, 5]-mesh_triangles[:, 8]
    mesh_normals = np.cross(v1, v2)
        
    return mesh_triangles, mesh_triangle_ids, mesh_normals, mesh_points


def GetCellsFromSTL(filename):
    '''
    cells, cell_ids, normals, points = GetCellsFromSTL(stl_filename)
    cells: [n, 9] array, where n is the number of cells
    cell_ids: [n, 3] array, where n is the number of cells
    normals: [n, 3] array, where h is the number of cells
    points: [m, 3] array, where m is the number of points
    '''
    
    #vtk reader for cells and points
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()
        
    n_triangles = data.GetNumberOfCells()
    n_points = data.GetNumberOfPoints()
    mesh_triangles = np.zeros([n_triangles, 9], dtype='float32')
    mesh_triangle_ids = np.zeros([n_triangles, 3], dtype='int32')
    mesh_points = np.zeros([n_points, 3], dtype='float32')
    
    for i in range(n_triangles):
        mesh_triangles[i][0], mesh_triangles[i][1], mesh_triangles[i][2] = data.GetPoint(data.GetCell(i).GetPointId(0))
        mesh_triangles[i][3], mesh_triangles[i][4], mesh_triangles[i][5] = data.GetPoint(data.GetCell(i).GetPointId(1))
        mesh_triangles[i][6], mesh_triangles[i][7], mesh_triangles[i][8] = data.GetPoint(data.GetCell(i).GetPointId(2))
        mesh_triangle_ids[i][0] = data.GetCell(i).GetPointId(0)
        mesh_triangle_ids[i][1] = data.GetCell(i).GetPointId(1)
        mesh_triangle_ids[i][2] = data.GetCell(i).GetPointId(2)
        
    
    for i in range(n_points):
        mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)
        
    #normal
    v1 = np.zeros([n_triangles, 3], dtype='float32')
    v2 = np.zeros([n_triangles, 3], dtype='float32')
    v1[:, 0] = mesh_triangles[:, 0]-mesh_triangles[:, 3]
    v1[:, 1] = mesh_triangles[:, 1]-mesh_triangles[:, 4]
    v1[:, 2] = mesh_triangles[:, 2]-mesh_triangles[:, 5]
    v2[:, 0] = mesh_triangles[:, 3]-mesh_triangles[:, 6]
    v2[:, 1] = mesh_triangles[:, 4]-mesh_triangles[:, 7]
    v2[:, 2] = mesh_triangles[:, 5]-mesh_triangles[:, 8]
    mesh_normals = np.cross(v1, v2)
        
    return mesh_triangles, mesh_triangle_ids, mesh_normals, mesh_points


def GetCellLabelFromSTL(main_cells, label_file_list, tol=0.01):
    '''
    input:
        main_cells: [n, 9] array
        label_file_list: a list containing STL file names of labels
    return:
        cell_labels: [n, 1] array
    '''
    cell_labels = np.zeros([len(main_cells), 1], dtype='int32')
    #initialize all values 
    cell_labels += len(label_file_list)
    
    cell_centers = (main_cells[:, 0:3] + main_cells[:, 3:6] + main_cells[:, 6:9]) / 3.0
    
    for i_label in range(len(label_file_list)):
        label_cells, label_cell_ids, label_normals, label_points = GetCellsFromSTL(label_file_list[i_label])
        
        label_cell_centers = (label_cells[:, 0:3] + label_cells[:, 3:6] + label_cells[:, 6:9]) / 3.0
        D = distance_matrix(cell_centers, label_cell_centers)
        
        if len(np.argwhere(D<=tol)) > label_cell_centers.shape[0]:
            sys.exit('tolerance ({0}) is too large, please adjust.'.format(tol))
        elif len(np.argwhere(D<=tol)) < label_cell_centers.shape[0]:
            sys.exit('tolerance ({0}) is too small, please adjust.'.format(tol))
        else:
            for i in range(label_cell_centers.shape[0]):
                label_id = np.argwhere(D<=tol)[i][0]
                cell_labels[label_id, 0] = i_label
    
    #Make gingival == 0
    cell_labels += 1
    cell_labels[cell_labels==(len(label_file_list)+1)] = 0
                
    return cell_labels


def OutputVTPFromToothLabels(file_name, label_list, output_path_name):
    '''
    inputs:
        file_name: vtp filename
        label_list: list including teeth names
        output_path_name: output_path + output_name, e.g., './tmp/Sample_01'
    '''
    cells, cell_ids, normals, points, labels = GetCellsFromVTP(file_name)
    for i_label in range(len(label_list)):
        selected_idx = np.where(labels==(i_label+1))[0]
        tooth_cells = cells[selected_idx, :]
        tooth_labels = labels[selected_idx, :]
        tooth_cell_ids, tooth_points = GetCellIdsPointsFromCells(tooth_cells)
        
        vtk_poly = GetVTKPolyDataFromCellsWithLabels(tooth_cell_ids, tooth_labels, tooth_points)
        OutputVTPFile(vtk_poly, output_path_name+label_list[i_label])

def GetVTKPolyDataFromCells(cell_ids, points):
    '''
    inputs:
        cell_ids
        points
    output:
        vtk_polyData
    '''
    #setup points and cells
    Polydata = vtk.vtkPolyData()
    Points = vtk.vtkPoints()
    Cells = vtk.vtkCellArray()
    
    Points.SetData(numpy_to_vtk(points))
    Cells.SetCells(len(cell_ids),
                   numpy_to_vtkIdTypeArray(np.hstack((np.ones(len(cell_ids))[:, None] * 3,
                                                      cell_ids)).astype(np.int64).ravel(),
                                           deep=1))
    
    Polydata.SetPoints(Points)
    Polydata.SetPolys(Cells)
    
    Polydata.Modified()
    
    return Polydata


def GetVTKPolyDataFromCellsWithLabels(cell_ids, cell_labels, points):
    '''
    inputs:
        cell_ids
        cell_labels
        points
    output:
        vtk_polyData
    '''
    #setup points and cells
    Polydata = vtk.vtkPolyData()
    Points = vtk.vtkPoints()
    Cells = vtk.vtkCellArray()
    
    Points.SetData(numpy_to_vtk(points))
    Cells.SetCells(len(cell_ids),
                   numpy_to_vtkIdTypeArray(np.hstack((np.ones(len(cell_ids))[:, None] * 3,
                                                      cell_ids)).astype(np.int64).ravel(),
                                           deep=1))
    
    #setup label
    Labels = vtk.vtkUnsignedCharArray();
    Labels.SetNumberOfComponents(1);
    Labels.SetName("Label");
    for i_label in cell_labels:
        Labels.InsertNextTuple1(i_label)    
    
    Polydata.SetPoints(Points)
    Polydata.SetPolys(Cells)
    Polydata.GetCellData().SetScalars(Labels);
    
    Polydata.Modified()
    
    return Polydata
    

def OutputVTPFile(Polydata, output_name):
    '''
    input:
        vtk_polyData
    output:
        vtp_file
    '''
    if vtk.VTK_MAJOR_VERSION <= 5:
        Polydata.Update()
     
    writer = vtk.vtkXMLPolyDataWriter();
    writer.SetFileName("{0}.vtp".format(output_name));
    if vtk.VTK_MAJOR_VERSION <= 5:
        writer.SetInput(Polydata)
    else:
        writer.SetInputData(Polydata)
    writer.Write()
    
    
def GetCellsFromVTP(vtk_filename):
    '''
    cells, cell_ids, normals, points, labels = GetCellsFromVTP(vtp_filename)
    cells: [n, 9] array, where n is the number of cells
    cell_ids: [n, 3] array, where n is the number of cells
    normals: [n, 3] array, where n is the number of cells
    points: [m, 3] array, where m is the number of points
    labels: [n, 1] array, where n is the number of cells
    '''
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtk_filename)
    reader.Update()
    
    data = reader.GetOutput()
    
    n_triangles = data.GetNumberOfCells()
    n_points = data.GetNumberOfPoints()
    mesh_triangles = np.zeros([n_triangles, 9], dtype='float32')
    mesh_triangle_ids = np.zeros([n_triangles, 3], dtype='int32')
    mesh_points = np.zeros([n_points, 3], dtype='float32')
    
    for i in range(n_points):
        mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)
        
    for i in range(n_triangles):
        mesh_triangles[i][0], mesh_triangles[i][1], mesh_triangles[i][2] = data.GetPoint(data.GetCell(i).GetPointId(0))
        mesh_triangles[i][3], mesh_triangles[i][4], mesh_triangles[i][5] = data.GetPoint(data.GetCell(i).GetPointId(1))
        mesh_triangles[i][6], mesh_triangles[i][7], mesh_triangles[i][8] = data.GetPoint(data.GetCell(i).GetPointId(2))
        mesh_triangle_ids[i][0] = data.GetCell(i).GetPointId(0)
        mesh_triangle_ids[i][1] = data.GetCell(i).GetPointId(1)
        mesh_triangle_ids[i][2] = data.GetCell(i).GetPointId(2)
        
    #normal
    v1 = np.zeros([n_triangles, 3], dtype='float32')
    v2 = np.zeros([n_triangles, 3], dtype='float32')
    v1[:, 0] = mesh_triangles[:, 0]-mesh_triangles[:, 3]
    v1[:, 1] = mesh_triangles[:, 1]-mesh_triangles[:, 4]
    v1[:, 2] = mesh_triangles[:, 2]-mesh_triangles[:, 5]
    v2[:, 0] = mesh_triangles[:, 3]-mesh_triangles[:, 6]
    v2[:, 1] = mesh_triangles[:, 4]-mesh_triangles[:, 7]
    v2[:, 2] = mesh_triangles[:, 5]-mesh_triangles[:, 8]
    mesh_normals = np.cross(v1, v2)
    
    #cell labels
    mesh_labels = np.zeros([n_triangles, 1], dtype='int32')
    for i in range(n_triangles):
        mesh_labels[i] = data.GetCellData().GetArray('Label').GetValue(i)
    
    
    return mesh_triangles, mesh_triangle_ids, mesh_normals, mesh_points, mesh_labels


def GetCellsFromVTPWithoutLabels(vtk_filename):
    '''
    cells, cell_ids, normals, points = GetCellsFromVTPWithoutLabels(vtp_filename)
    cells: [n, 9] array, where n is the number of cells
    cell_ids: [n, 3] array, where n is the number of cells
    normals: [n, 3] array, where n is the number of cells
    points: [m, 3] array, where m is the number of points
    '''
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtk_filename)
    reader.Update()
    
    data = reader.GetOutput()
    
    n_triangles = data.GetNumberOfCells()
    n_points = data.GetNumberOfPoints()
    mesh_triangles = np.zeros([n_triangles, 9], dtype='float32')
    mesh_triangle_ids = np.zeros([n_triangles, 3], dtype='int32')
    mesh_points = np.zeros([n_points, 3], dtype='float32')
    
    for i in range(n_points):
        mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)
        
    for i in range(n_triangles):
        mesh_triangles[i][0], mesh_triangles[i][1], mesh_triangles[i][2] = data.GetPoint(data.GetCell(i).GetPointId(0))
        mesh_triangles[i][3], mesh_triangles[i][4], mesh_triangles[i][5] = data.GetPoint(data.GetCell(i).GetPointId(1))
        mesh_triangles[i][6], mesh_triangles[i][7], mesh_triangles[i][8] = data.GetPoint(data.GetCell(i).GetPointId(2))
        mesh_triangle_ids[i][0] = data.GetCell(i).GetPointId(0)
        mesh_triangle_ids[i][1] = data.GetCell(i).GetPointId(1)
        mesh_triangle_ids[i][2] = data.GetCell(i).GetPointId(2)
        
    #normal
    v1 = np.zeros([n_triangles, 3], dtype='float32')
    v2 = np.zeros([n_triangles, 3], dtype='float32')
    v1[:, 0] = mesh_triangles[:, 0]-mesh_triangles[:, 3]
    v1[:, 1] = mesh_triangles[:, 1]-mesh_triangles[:, 4]
    v1[:, 2] = mesh_triangles[:, 2]-mesh_triangles[:, 5]
    v2[:, 0] = mesh_triangles[:, 3]-mesh_triangles[:, 6]
    v2[:, 1] = mesh_triangles[:, 4]-mesh_triangles[:, 7]
    v2[:, 2] = mesh_triangles[:, 5]-mesh_triangles[:, 8]
    mesh_normals = np.cross(v1, v2)
    
    return mesh_triangles, mesh_triangle_ids, mesh_normals, mesh_points


def GetCellsFromVTPWithCellAttribute(vtk_filename, attribute_name, attribute_dim):
    '''
    cells, cell_ids, normals, points, attribute = GetCellsFromVTPWithCellAttribute(vtp_filename, attribute_name, attribute_dim)
    inputs:
        vtk_filename
        attribute_name
        attribute_dim: ususally 1, 2 or 3
    outputs:
        cells: [n, 9] array, where n is the number of cells
        cell_ids: [n, 3] array, where n is the number of cells
        normals: [n, 3] array, where n is the number of cells
        points: [m, 3] array, where m is the number of points
        attribute: [n, h] array, where n is the number of cells and h is dimension
    '''
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtk_filename)
    reader.Update()
    
    data = reader.GetOutput()
    
    n_triangles = data.GetNumberOfCells()
    n_points = data.GetNumberOfPoints()
    mesh_triangles = np.zeros([n_triangles, 9], dtype='float32')
    mesh_triangle_ids = np.zeros([n_triangles, 3], dtype='int32')
    mesh_points = np.zeros([n_points, 3], dtype='float32')
    
    for i in range(n_points):
        mesh_points[i][0], mesh_points[i][1], mesh_points[i][2] = data.GetPoint(i)
        
    for i in range(n_triangles):
        mesh_triangles[i][0], mesh_triangles[i][1], mesh_triangles[i][2] = data.GetPoint(data.GetCell(i).GetPointId(0))
        mesh_triangles[i][3], mesh_triangles[i][4], mesh_triangles[i][5] = data.GetPoint(data.GetCell(i).GetPointId(1))
        mesh_triangles[i][6], mesh_triangles[i][7], mesh_triangles[i][8] = data.GetPoint(data.GetCell(i).GetPointId(2))
        mesh_triangle_ids[i][0] = data.GetCell(i).GetPointId(0)
        mesh_triangle_ids[i][1] = data.GetCell(i).GetPointId(1)
        mesh_triangle_ids[i][2] = data.GetCell(i).GetPointId(2)
        
    #normal
    v1 = np.zeros([n_triangles, 3], dtype='float32')
    v2 = np.zeros([n_triangles, 3], dtype='float32')
    v1[:, 0] = mesh_triangles[:, 0]-mesh_triangles[:, 3]
    v1[:, 1] = mesh_triangles[:, 1]-mesh_triangles[:, 4]
    v1[:, 2] = mesh_triangles[:, 2]-mesh_triangles[:, 5]
    v2[:, 0] = mesh_triangles[:, 3]-mesh_triangles[:, 6]
    v2[:, 1] = mesh_triangles[:, 4]-mesh_triangles[:, 7]
    v2[:, 2] = mesh_triangles[:, 5]-mesh_triangles[:, 8]
    mesh_normals = np.cross(v1, v2)
    
    #cell attribute
    cell_attribute = np.zeros([n_triangles, attribute_dim], dtype='float32')
    for i in range(n_triangles):
        cell_attribute[i] = data.GetCellData().GetArray(attribute_name).GetValue(i)
    
    return mesh_triangles, mesh_triangle_ids, mesh_normals, mesh_points, cell_attribute


def GetVTKPolyDataFromVTP(vtk_filename):
    '''
    vtk_poly = GetVTKPolyDataFromVTP(vtp_filename)
    '''
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtk_filename)
    reader.Update()
    
    data = reader.GetOutput()
    
    return data


def GetVTKTransformationMatrix(rotate_X=[-180, 180], rotate_Y=[-180, 180], rotate_Z=[-180, 180],
                               translate_X=[-10, 10], translate_Y=[-10, 10], translate_Z=[-10, 10],
                               scale_X=[0.8, 1.2], scale_Y=[0.8, 1.2], scale_Z=[0.8, 1.2]):
    '''
    get transformation matrix (4*4)
    
    return: vtkMatrix4x4
    '''
    Trans = vtk.vtkTransform()
    
    ry_flag = np.random.randint(0,2) #if 0, no rotate
    rx_flag = np.random.randint(0,2) #if 0, no rotate
    rz_flag = np.random.randint(0,2) #if 0, no rotate
    if ry_flag == 1:
        # rotate along Yth axis
        Trans.RotateY(np.random.uniform(rotate_Y[0], rotate_Y[1]))
    if rx_flag == 1:
        # rotate along Xth axis
        Trans.RotateX(np.random.uniform(rotate_X[0], rotate_X[1]))
    if rz_flag == 1:
        # rotate along Zth axis
        Trans.RotateZ(np.random.uniform(rotate_Z[0], rotate_Z[1]))

    trans_flag = np.random.randint(0,2) #if 0, no translate
    if trans_flag == 1:
        Trans.Translate([np.random.uniform(translate_X[0], translate_X[1]),
                         np.random.uniform(translate_Y[0], translate_Y[1]),
                         np.random.uniform(translate_Z[0], translate_Z[1])])

    scale_flag = np.random.randint(0,2)
    if scale_flag == 1:
        Trans.Scale([np.random.uniform(scale_X[0], scale_X[1]),
                     np.random.uniform(scale_Y[0], scale_Y[1]),
                     np.random.uniform(scale_Z[0], scale_Z[1])])
    
    matrix = Trans.GetMatrix()
    
    return matrix


def TransformSTL(filename, output_name, vtk_matrix):
    '''
    transform STL file, for augmentation
    '''
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    
    Trans = vtk.vtkTransform()
    Trans.SetMatrix(vtk_matrix)
    
    TransFilter = vtk.vtkTransformPolyDataFilter()
    TransFilter.SetTransform(Trans)
    TransFilter.SetInputConnection(reader.GetOutputPort())
    TransFilter.Update()
    
    #write STL file
    StlWriter = vtk.vtkSTLWriter()
    StlWriter.SetFileName(output_name)
    StlWriter.SetInputConnection(TransFilter.GetOutputPort())
    StlWriter.Write()


def TransformVTP(filename, output_name, vtk_matrix):
    '''
    transform vtp file, for augmentation
    '''
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(filename)
    reader.Update()
    
    Trans = vtk.vtkTransform()
    Trans.SetMatrix(vtk_matrix)
    
    TransFilter = vtk.vtkTransformPolyDataFilter()
    TransFilter.SetTransform(Trans)
    TransFilter.SetInputConnection(reader.GetOutputPort())
    TransFilter.Update()
    
    #write VTP file
    OutputVTPFile(TransFilter.GetOutput(), output_name)


def DecimateSTL(filename, reduction_rate):
    '''
    inputs:
        filename
        reduction_rate (e.g., 0.1)
    outputs:
        vtk_obj, vtk_polyData
    '''
    reader = vtk.vtkSTLReader()
    reader.SetFileName(filename)
    reader.Update()
    data = reader.GetOutput()

    print("Before decimation\n"
          "-----------------\n"
          "There are " + str(data.GetNumberOfPoints()) + "points.\n"
          "There are " + str(data.GetNumberOfPolys()) + "polygons.\n")

    decimate_reader = vtk.vtkQuadricDecimation()
    decimate_reader.SetInputData(data)
    decimate_reader.SetTargetReduction(reduction_rate)
    decimate_reader.Update()
    decimate_data = decimate_reader.GetOutput()

    print("After decimation \n"
          "-----------------\n"
          "There are " + str(decimate_data.GetNumberOfPoints()) + "points.\n"
          "There are " + str(decimate_data.GetNumberOfPolys()) + "polygons.\n")

    return decimate_reader, decimate_data


def OutputSTLFromVTK(vtk_obj, filename):
    '''
    inputs:
        vtk_obj
        filename
    '''
    # Write the stl file to disk
    stlWriter = vtk.vtkSTLWriter()
    stlWriter.SetFileName(filename)
    stlWriter.SetInputConnection(vtk_obj.GetOutputPort())
    stlWriter.Write()
    
    
def OutputSTLFromVTP(vtp_filename, filename):
    '''
    inputs:
        vtk_obj
        filename
    '''
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtp_filename)
    reader.Update()
    OutputSTLFromVTK(reader, filename)
    
    
def ReflectCells(cells, points):
    '''
    inputs:
        cells
        points
    '''
    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    zmin = np.min(points[:, 2])
    zmax = np.max(points[:, 2])
    center = np.array([np.mean(points[:, 0]), np.mean(points[:, 1]), np.mean(points[:, 2])])
    
    point1 = [xmin, ymin, zmin]
    point2 = [xmin, ymax, zmin]
    point3 = [xmin, ymin, zmax]
    
    #get equation of the plane by three points
    v1 = np.zeros([3,])
    v2 = np.zeros([3,])
    
    for i in range(3):
        v1[i] = point1[i] - point2[i]
        v2[i] = point1[i] - point3[i]
    
    normal_vec = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))
    
    flipped_cells = np.copy(cells)
    flipped_points = np.copy(points)
    
    #flip cells
    for idx in range(len(cells)):
        tmp_p1 = cells[idx, 0:3]
        tmp_p2 = cells[idx, 3:6]
        tmp_p3 = cells[idx, 6:9]
        
        tmp_v1 = tmp_p1 - point1
        dis_v1 = np.dot(tmp_v1, normal_vec)*normal_vec
        
        tmp_v2 = tmp_p2 - point1
        dis_v2 = np.dot(tmp_v2, normal_vec)*normal_vec
        
        tmp_v3 = tmp_p3 - point1
        dis_v3 = np.dot(tmp_v3, normal_vec)*normal_vec
        
        flipped_p1 = tmp_p1 - 2*dis_v1
        flipped_p2 = tmp_p2 - 2*dis_v2
        flipped_p3 = tmp_p3 - 2*dis_v3

        flipped_cells[idx, 0:3] = flipped_p1
        flipped_cells[idx, 3:6] = flipped_p3 #change order p3 and p2
        flipped_cells[idx, 6:9] = flipped_p2 #change order p3 and p2
        
    #flip points
    for idx in range(len(points)):
        tmp_p1 = points[idx, 0:3]
        
        tmp_v1 = tmp_p1 - point1
        dis_v1 = np.dot(tmp_v1, normal_vec)*normal_vec
                
        flipped_p1 = tmp_p1 - 2*dis_v1

        flipped_points[idx, 0:3] = flipped_p1
    
    #move flipped_cells and flipped_points back to the center
    flipped_center = np.array([np.mean(flipped_points[:, 0]), np.mean(flipped_points[:, 1]), np.mean(flipped_points[:, 2])])
    displacement = center - flipped_center
    
    flipped_cells[:, 0:3] += displacement
    flipped_cells[:, 3:6] += displacement
    flipped_cells[:, 6:9] += displacement
    flipped_points[:, 0:3] += displacement
        
    return flipped_cells, flipped_points


def ReflectLandmarks(landmarks, points):
    '''
    inputs:
        landmarks
        points (whole points as a reference)
    '''
    xmin = np.min(points[:, 0])
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1])
    zmin = np.min(points[:, 2])
    zmax = np.max(points[:, 2])
    center = np.array([np.mean(points[:, 0]), np.mean(points[:, 1]), np.mean(points[:, 2])])
    
    point1 = [xmin, ymin, zmin]
    point2 = [xmin, ymax, zmin]
    point3 = [xmin, ymin, zmax]
    
    #get equation of the plane by three points
    v1 = np.zeros([3,])
    v2 = np.zeros([3,])
    
    for i in range(3):
        v1[i] = point1[i] - point2[i]
        v2[i] = point1[i] - point3[i]
    
    normal_vec = np.cross(v1, v2)/np.linalg.norm(np.cross(v1, v2))
    
    flipped_landmarks = np.copy(landmarks)
    flipped_points = np.copy(points)
            
    #flip landmarks
    for idx in range(len(landmarks)):
        tmp_p1 = landmarks[idx, 0:3]
        
        tmp_v1 = tmp_p1 - point1
        dis_v1 = np.dot(tmp_v1, normal_vec)*normal_vec
                
        flipped_p1 = tmp_p1 - 2*dis_v1

        flipped_landmarks[idx, 0:3] = flipped_p1
        
    #flip points
    for idx in range(len(points)):
        tmp_p1 = points[idx, 0:3]
        
        tmp_v1 = tmp_p1 - point1
        dis_v1 = np.dot(tmp_v1, normal_vec)*normal_vec
                
        flipped_p1 = tmp_p1 - 2*dis_v1

        flipped_points[idx, 0:3] = flipped_p1
    
    #move flipped_cells and flipped_points back to the center
    flipped_center = np.array([np.mean(flipped_points[:, 0]), np.mean(flipped_points[:, 1]), np.mean(flipped_points[:, 2])])
    displacement = center - flipped_center
    
    flipped_landmarks[:, 0:3] += displacement
    flipped_points[:, 0:3] += displacement
        
    return flipped_landmarks, flipped_points


def RunICP(moving_poly, fixed_poly, type='rigid', max_iter=2000, max_mdis=0.001, max_num_landmarks=200):
    '''
    perform Iterative Closest Points (ICP) algorithm on vtk_polys
    inputs:
        moving_poly
        fixed_poly
        type: 'rigid' or 'affine'
        max_iter: 2000 (default)
        max_mdis: 0.001 (default)
        max_num_landmarks: 200 (default)
    output
        output_vtk_matrix: 4x4 vtk matrix
        output_np_matrix: 4x4 numpy matrix
    '''
    icp = vtk.vtkIterativeClosestPointTransform()
    icp.SetSource(moving_poly)
    icp.SetTarget(fixed_poly)
    
    if type == 'rigid':
        icp.GetLandmarkTransform().SetModeToRigidBody()
    elif type == 'affine':
        icp.GetLandmarkTransform().SetModeToAffine()
    elif type == 'Similarity':
        icp.GetLandmarkTransform().SetModeToSimilarity()
    else:
        print('Wrong transform type, use default setting (rigid')
        icp.GetLandmarkTransform().SetModeToRigidBody()
        
    icp.SetMeanDistanceModeToRMS()
#    icp.SetMeanDistanceModeToAbsoluteValue()
    icp.SetMaximumNumberOfIterations(max_iter)
    icp.SetMaximumMeanDistance(max_mdis)
    icp.SetMaximumNumberOfLandmarks(max_num_landmarks)
    
    icp.Modified()
    icp.Update()
    output_vtk_Matrix = vtk.vtkMatrix4x4()
    icp.GetMatrix(output_vtk_Matrix)
    
    output_np_matrix = np.zeros([4, 4])
    for i in range(4):
        for j in range(4):
            output_np_matrix[i, j] = output_vtk_Matrix.GetElement(i, j)
    
    return output_vtk_Matrix, output_np_matrix


def ComputePointCurvature(cell_ids, points, type='mean'):
    '''
    inputs:
        cell_ids
        points
        type: ['mean'/'max'/'min'/'Gaussian'] default: 'mean'
    output:
        curvature: nx1 numpy array, where n is number of points
    '''
    Polydata = vtk.vtkPolyData()
    Points = vtk.vtkPoints()
    Cells = vtk.vtkCellArray()
    Points.SetData(numpy_to_vtk(points))
    Cells.SetCells(len(cell_ids),
                   numpy_to_vtkIdTypeArray(np.hstack((np.ones(len(cell_ids))[:, None] * 3,
                                                      cell_ids)).astype(np.int64).ravel(),
                   deep=1))
    Polydata.SetPoints(Points)
    Polydata.SetPolys(Cells)
    Polydata.Modified()
    
    curv = vtk.vtkCurvatures()
    curv.SetInputData(Polydata)
    if type == 'mean':
        curv.SetCurvatureTypeToMean()
    elif type == 'max':
        curv.SetCurvatureTypeToMaximum()
    elif type == 'min':
        curv.SetCurvatureTypeToMinimum()
    elif type == 'Gaussian':
        curv.SetCurvatureTypeToGaussian()
    else:
        curv.SetCurvatureTypeToMean()
        
    curv.Update()
    
    n_points = Polydata.GetNumberOfPoints()
    pc = np.zeros([n_points, 1], dtype='float32')
    for i in range(n_points):
        pc[i] = curv.GetOutput().GetPointData().GetArray(0).GetValue(i)
        
    return pc


def AddPointAttribute(vtk_poly, np_attribute, attribute_name='New Attribute'):
    '''
    inputs:
        vtk_poly: vtkPolyData
        np_attribue: numpy array
        attribute_name: e.g., 'curvature'
    '''
    new_attribute = vtk.vtkDoubleArray();
    new_attribute.SetNumberOfComponents(np_attribute.shape[1]);
    new_attribute.SetName(attribute_name);
    
#    n_points = vtk_poly.GetNumberOfPoints()
    for i_point in np_attribute:
        if np_attribute.shape[1] == 1:
            new_attribute.InsertNextTuple1(i_point)
        elif np_attribute.shape[1] == 2:
            new_attribute.InsertNextTuple2(i_point[0], i_point[1])
        elif np_attribute.shape[1] == 3:
            new_attribute.InsertNextTuple3(i_point[0], i_point[1], i_point[2])
        else:
            print('Check np_attribute dimension, only support 1D, 2D, and 3D now')
    
    vtk_poly.GetPointData().AddArray(new_attribute)
    # vtk_poly.GetPointData().SetVectors(new_attribute)
    
    return vtk_poly


def AddCellAttribute(vtk_poly, np_attribute, attribute_name='New Attribute'):
    '''
    inputs:
        vtk_poly: vtkPolyData
        np_attribue: numpy array
        attribute_name: e.g., 'normal'
    '''
    new_attribute = vtk.vtkDoubleArray();
    new_attribute.SetNumberOfComponents(np_attribute.shape[1]);
    new_attribute.SetName(attribute_name);
    
#    n_cells = vtk_poly.GetNumberOfCells()
    for i_cell in np_attribute:
        if np_attribute.shape[1] == 1:
            new_attribute.InsertNextTuple1(i_cell)
        elif np_attribute.shape[1] == 2:
            new_attribute.InsertNextTuple2(i_cell[0], i_cell[1])
        elif np_attribute.shape[1] == 3:
            new_attribute.InsertNextTuple3(i_cell[0], i_cell[1], i_cell[2])
        else:
            print('Check np_attribute dimension, only support 1D, 2D, and 3D now')
    
    vtk_poly.GetCellData().AddArray(new_attribute)
    # vtk_poly.GetCellData().SetVectors(new_attribute)
    
    return vtk_poly


def GetLandmarksFromGMDCSV(file_name, landmark_name):
    '''
    Get landmarks by loading Geomagic Design CSV file
    '''
    df = pd.read_csv(file_name)
    landmarks = np.zeros([df.shape[0], 3])
    
    for i in range(df.shape[0]):
        if df['Name'][i] == landmark_name[i]:
            x = df['Position X'][i]
            y = df['Position Y'][i]
            z = df['Position Z'][i]
            landmarks[i, 0] = x
            landmarks[i, 1] = y
            landmarks[i, 2] = z
        else:
            print('Wrong order in csv file', file_name)
            break
    
    return landmarks


def GetLandmarksFromGMDCSVNoCheck(file_name):
    '''
    Get landmarks by loading Geomagic Design CSV file
    '''
    df = pd.read_csv(file_name)
    landmarks = np.zeros([df.shape[0], 3])
    
    for i in range(df.shape[0]):
        x = df['Position X'][i]
        y = df['Position Y'][i]
        z = df['Position Z'][i]
        landmarks[i, 0] = x
        landmarks[i, 1] = y
        landmarks[i, 2] = z
    
    return landmarks


def GetVTKPolyDataFromLandmarks(landmarks):
    '''
    inputs:
        cell_ids
        points
    output:
        vtk_polyData
    '''
    #setup points and cells
    Polydata = vtk.vtkPolyData()
    Points = vtk.vtkPoints()
    
    Points.SetData(numpy_to_vtk(landmarks))
    
    Polydata.SetPoints(Points)
    
    Polydata.Modified()
    
    return Polydata


def GetLandmarksFromVTP(vtk_filename):
    '''
    '''
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(vtk_filename)
    reader.Update()
    
    data = reader.GetOutput()
    
    n_points = data.GetNumberOfPoints()
    landmarks = np.zeros([n_points, 3], dtype='float32')
    
    for i in range(n_points):
        landmarks[i][0], landmarks[i][1], landmarks[i][2] = data.GetPoint(i)
    
    return landmarks


def ComputeGuassianHeatmap(landmark, cells, cell_ids, points, sigma = 10.0, height = 1.0):
    '''
    inputs:
        landmark: np.array [1, 3]
        cells
        cell_ids,
        points,
        sigma (default=10.0)
        height (default=1.0)
    output:
        vtk_poly
    '''
    cell_centers = (cells[:, 0:3] + cells[:, 3:6] + cells[:, 6:9]) / 3.0
    heatmap = np.zeros([cell_centers.shape[0], 1])
    
    for i_cell in range(len(cell_centers)):
        delx = cell_centers[i_cell, 0] - landmark[0]
        dely = cell_centers[i_cell, 1] - landmark[1]
        delz = cell_centers[i_cell, 2] - landmark[2]
        heatmap[i_cell, 0] = height*math.exp(-1*(delx*delx+dely*dely+delz*delz)/2.0/sigma/sigma)
            
    vtk_poly = GetVTKPolyDataFromCells(cell_ids, points)
    vtk_poly = AddCellAttribute(vtk_poly, heatmap, attribute_name='heatmap')
    
    return vtk_poly


def ComputeMeshDeviations(vtk_filename1, vtk_filename2, output_filename):
    '''
    inputs:
        vtk_filename1 (with .vtp)
        vtk_filename2 (with .vtp)
        output_filename (without extension)
    '''
    m_vtk_poly = GetVTKPolyDataFromVTP(vtk_filename1)
    f_vtk_poly = GetVTKPolyDataFromVTP(vtk_filename2)
        
    df = vtk.vtkDistancePolyDataFilter()
    df.SignedDistanceOn()
    df.SetInputData(0, m_vtk_poly)
    df.SetInputData(1, f_vtk_poly)
    df.Update()
        
    vtk_poly = df.GetOutput()
        
    OutputVTPFile(vtk_poly, output_filename)
    
    
def LRBC(tooth_vtp_filename):
    '''
    inputs:
        tooth_vtp_filename: (with .vtp)
    
    outputs:
        point0(center), point1
    '''
    tmp_path = './.tmp_LRBC/'
    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)
        
    cells, cell_ids, normals, points = GetCellsFromVTPWithoutLabels(tooth_vtp_filename)
    vtk_poly = GetVTKPolyDataFromVTP(tooth_vtp_filename)
    OutputVTPFile(vtk_poly, tmp_path+'tmp')
    
    reader = vtk.vtkXMLPolyDataReader()
    reader.SetFileName(tmp_path+'tmp.vtp')
    reader.Update()
    
    featureEdges = vtk.vtkFeatureEdges()
    featureEdges.SetInputConnection(reader.GetOutputPort())
    featureEdges.BoundaryEdgesOn()
    featureEdges.FeatureEdgesOff()
    featureEdges.ManifoldEdgesOff()
    featureEdges.NonManifoldEdgesOff()
    featureEdges.Update()
    
    vtk_poly = featureEdges.GetOutput()
    num_points = vtk_poly.GetNumberOfPoints()
    boundary_points = np.zeros([num_points, 3], dtype='float32')
    
    for i in range(num_points):
        boundary_points[i][0], boundary_points[i][1], boundary_points[i][2] = vtk_poly.GetPoint(i)     
    
    all_bp_center = []
    all_bp_points = []
    
    i_count = 0
    while True:
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(tmp_path+'tmp.vtp')
        reader.Update()
        
        featureEdges = vtk.vtkFeatureEdges()
        featureEdges.SetInputConnection(reader.GetOutputPort())
        featureEdges.BoundaryEdgesOn()
        featureEdges.FeatureEdgesOff()
        featureEdges.ManifoldEdgesOff()
        featureEdges.NonManifoldEdgesOff()
        featureEdges.Update()
        
        # extract boundary points
        vtk_poly = featureEdges.GetOutput()
        num_points = vtk_poly.GetNumberOfPoints()
        boundary_points = np.zeros([num_points, 3], dtype='float32')
        
        for i in range(num_points):
            boundary_points[i][0], boundary_points[i][1], boundary_points[i][2] = vtk_poly.GetPoint(i)
        all_bp_points.append(boundary_points)
            
        bp_center = np.mean(boundary_points[:, 0:3], axis=0)
        all_bp_center.append(bp_center)
        
        # eliminate boundary points and generate a new mesh
        all_idx = np.array([], dtype='int32')
        for i_bp in boundary_points:
            counts0 = np.bincount(np.where(points[:, 0:3]==i_bp)[0])
            bp_idx = np.argmax(counts0)
            
            bp_cell_idx = np.where(cell_ids[:, 0:3]==bp_idx)
            all_idx = np.concatenate((all_idx, bp_cell_idx[0]), axis=None)
            
            all_idx = np.unique(all_idx)
    
        cells = np.delete(cells, all_idx, 0)
        try:
            cell_ids, points = GetCellIdsPointsFromCells(cells)
            vtk_poly = GetVTKPolyDataFromCells(cell_ids, points)
            OutputVTPFile(vtk_poly, tmp_path+'tmp')
            i_count += 1
#            print(i_count)
        except:
            break
        
    alpha = 0.5
    
    all_bp_center = np.asarray(all_bp_center)
    c_e = all_bp_center[0:int(i_count*alpha), :]
    c_m_points = all_bp_points[int(i_count*alpha)]
    for i in range(int(i_count*alpha)+1, i_count+1):
        c_m_points = np.concatenate((c_m_points, all_bp_points[i]))
    c_m = np.mean(c_m_points, axis=0)
    c_m = c_m.reshape([1, 3])
    final_bp_centers = np.append(c_e, c_m, axis=0)
    final_bp_centers_mean = np.mean(final_bp_centers, axis=0)
    
    #SVD
    uu, dd, vv = np.linalg.svd(final_bp_centers - final_bp_centers_mean)
    
    cells, cell_ids, normals, points = GetCellsFromVTPWithoutLabels(tooth_vtp_filename)
    center = np.array([points[:, 0].mean(), points[:, 1].mean(), points[:, 2].mean()])
    
    #remove tmp_path
    shutil.rmtree(tmp_path)
    
    return (center, center+vv[0])
    

################################################################
##          From GitHub
################################################################
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)

    prev_error = 0

    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        src = np.dot(T, src)

        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)

    return T, distances, i
    
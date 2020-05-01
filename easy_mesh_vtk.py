import numpy as np
import vtk
from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

class Easy_Mesh(object):
    def __init__(self):
        #initialize
        self.filename = None
        self.reader = None
        self.vtkPolyData = None
        self.cells = np.array([])
        self.cell_ids = np.array([])
        self.points = np.array([])
        self.cell_normals = np.array([])
        self.cell_labels = np.array([])
        self.cell_curvature = np.array([])
        self.point_curvature = np.array([])
        self.edges = np.array([])
        
    def read_stl(self, stl_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_normals
            self.cell_labels
        '''
        self.filename = stl_filename
        reader = vtk.vtkSTLReader()
        reader.SetFileName(stl_filename)
        reader.Update()
        self.reader = reader
        
        data = reader.GetOutput()
        self.vtkPolyData = data
        
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
        
        self.cells = mesh_triangles
        self.cell_ids = mesh_triangle_ids
        self.points = mesh_points
        self.cell_normals = mesh_normals
        
        
    def read_obj(self, obj_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_normals
        '''
        self.filename = obj_filename
        reader = vtk.vtkOBJReader()
        reader.SetFileName(obj_filename)
        reader.Update()
        self.reader = reader
        
        data = reader.GetOutput()
        self.vtkPolyData = data
        
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
            
        self.cells = mesh_triangles
        self.cell_ids = mesh_triangle_ids
        self.points = mesh_points
        self.cell_normals = mesh_normals
    
    
    def read_vtp(self, vtp_filename):
        '''
        update
            self.filename
            self.reader
            self.vtkPolyData
            self.cells
            self.cell_ids
            self.points
            self.cell_normals
        '''
        self.filename = vtp_filename
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(vtp_filename)
        reader.Update()
        self.reader = reader
    
        data = reader.GetOutput()
        self.vtkPolyData = data
    
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
        
        self.cells = mesh_triangles
        self.cell_ids = mesh_triangle_ids
        self.points = mesh_points
        self.cell_normals = mesh_normals
        
        
    def load_cell_label(self):
        '''
        update
            self.cell_labels
        '''
        self.cell_labels = np.zeros([self.cells.shape[0], 1], dtype=np.int32)
        try:
            for i in range(self.cells.shape[0]):
                self.cell_labels[i] = self.vtkPolyData.GetCellData().GetArray('Label').GetValue(i)
        except:
            print('No cell attribute named "Label" in file: {0}'.format(self.filename))
            print('Initialized label attritube')
            
            
    def get_edge_info(self):
        '''
        update
            self.edges
        '''
        self.edges = np.zeros([self.cell_ids.shape[0], 3])
    
        for i_count in range(self.cell_ids.shape[0]):
            v1 = self.points[self.cell_ids[i_count, 0], :] - self.points[self.cell_ids[i_count, 1], :]
            v2 = self.points[self.cell_ids[i_count, 1], :] - self.points[self.cell_ids[i_count, 2], :]
            v3 = self.points[self.cell_ids[i_count, 0], :] - self.points[self.cell_ids[i_count, 2], :]
            self.edges[i_count, 0] = np.linalg.norm(v1)
            self.edges[i_count, 1] = np.linalg.norm(v2)
            self.edges[i_count, 2] = np.linalg.norm(v3)
            
    def update_cell_ids_and_points(self):
        '''
        call when self.cells is modified
        update
            self.cell_ids
            self.points
        '''
        self.points = self.cells.reshape([int(self.cells.shape[0]*3), 3])
        self.points = np.unique(self.points, axis=0)
        self.cell_ids = np.zeros([self.cells.shape[0], 3], dtype='int64')
    
        for i_count in range(self.cells.shape[0]):
            counts0 = np.bincount(np.where(self.points==self.cells[i_count, 0:3])[0])
            counts1 = np.bincount(np.where(self.points==self.cells[i_count, 3:6])[0])
            counts2 = np.bincount(np.where(self.points==self.cells[i_count, 6:9])[0])
            self.cell_ids[i_count, 0] = np.argmax(counts0)
            self.cell_ids[i_count, 1] = np.argmax(counts1)
            self.cell_ids[i_count, 2] = np.argmax(counts2)
            
    def update_vtkPolyData(self):
        '''
        update
            self.vtkPolyData
        '''
        vtkPolyData = vtk.vtkPolyData()
        points = vtk.vtkPoints()
        cells = vtk.vtkCellArray()
    
        points.SetData(numpy_to_vtk(self.points))
        cells.SetCells(len(self.cell_ids),
                       numpy_to_vtkIdTypeArray(np.hstack((np.ones(len(self.cell_ids))[:, None] * 3,
                                                          self.cell_ids)).astype(np.int64).ravel(),
                                               deep=1))
        vtkPolyData.SetPoints(points)
        vtkPolyData.SetPolys(cells)
        
        #setup label
        if self.cell_labels.shape[0] != 0:
            labels = vtk.vtkUnsignedCharArray();
            labels.SetNumberOfComponents(1);
            labels.SetName("Label");
            for i_label in self.cell_labels:
                labels.InsertNextTuple1(i_label)
            vtkPolyData.GetCellData().SetScalars(labels);
        else:
            print('No self.label avaiable! Please assign cell_labels first!\nMaybe try self.load_cell_label()')
        
        vtkPolyData.Modified()
        self.vtkPolyData = vtkPolyData
    
    
    def to_vtp(self, vtp_filename):
        None
            
#------------------------------------------------------------------------------
#def GetCellLabelFromSTL(main_cells, label_file_list, tol=0.01):
#    '''
#    input:
#        main_cells: [n, 9] array
#        label_file_list: a list containing STL file names of labels
#    return:
#        cell_labels: [n, 1] array
#    '''
#    cell_labels = np.zeros([len(main_cells), 1], dtype='int32')
#    #initialize all values 
#    cell_labels += len(label_file_list)
#    
#    cell_centers = (main_cells[:, 0:3] + main_cells[:, 3:6] + main_cells[:, 6:9]) / 3.0
#    
#    for i_label in range(len(label_file_list)):
#        label_cells, label_cell_ids, label_normals, label_points = GetCellsFromSTL(label_file_list[i_label])
#        
#        label_cell_centers = (label_cells[:, 0:3] + label_cells[:, 3:6] + label_cells[:, 6:9]) / 3.0
#        D = distance_matrix(cell_centers, label_cell_centers)
#        
#        if len(np.argwhere(D<=tol)) > label_cell_centers.shape[0]:
#            sys.exit('tolerance ({0}) is too large, please adjust.'.format(tol))
#        elif len(np.argwhere(D<=tol)) < label_cell_centers.shape[0]:
#            sys.exit('tolerance ({0}) is too small, please adjust.'.format(tol))
#        else:
#            for i in range(label_cell_centers.shape[0]):
#                label_id = np.argwhere(D<=tol)[i][0]
#                cell_labels[label_id, 0] = i_label
#    
#    #Make gingival == 0
#    cell_labels += 1
#    cell_labels[cell_labels==(len(label_file_list)+1)] = 0
#                
#    return cell_labels
            
#def OutputVTPFromToothLabels(file_name, label_list, output_path_name):
#    '''
#    inputs:
#        file_name: vtp filename
#        label_list: list including teeth names
#        output_path_name: output_path + output_name, e.g., './tmp/Sample_01'
#    '''
#    cells, cell_ids, normals, points, labels = GetCellsFromVTP(file_name)
#    for i_label in range(len(label_list)):
#        selected_idx = np.where(labels==(i_label+1))[0]
#        tooth_cells = cells[selected_idx, :]
#        tooth_labels = labels[selected_idx, :]
#        tooth_cell_ids, tooth_points = GetCellIdsPointsFromCells(tooth_cells)
#        
#        vtk_poly = GetVTKPolyDataFromCellsWithLabels(tooth_cell_ids, tooth_labels, tooth_points)
#        OutputVTPFile(vtk_poly, output_path_name+label_list[i_label])
            
    
if __name__ == '__main__':
    
    # create a new mesh by loading a file
    mesh = Easy_Mesh()
#    mesh.read_vtp('A0_Sample_01.vtp')
#    mesh.read_stl('Test5.stl')
    mesh.read_obj('tmp.obj')
#    mesh.load_cell_label()
    mesh.get_edge_info()
    mesh.update_vtkPolyData()

    
    # create a new mesh from cells
#    mesh2 = Easy_Mesh()
#    mesh2.cells = mesh.cells[np.where(mesh.cell_labels==1)[0]]
#    mesh2.cell_labels = mesh.cell_labels[np.where(mesh.cell_labels==1)[0]]
#    mesh2.update_cell_ids_and_points()
    

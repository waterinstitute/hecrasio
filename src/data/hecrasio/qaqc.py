"""
PFRA Module for working with HEC-RAS model output files
"""
from osgeo import gdal
from time import time
import geopandas as gpd
from geopandas.tools import sjoin
from shapely.ops import unary_union
from shapely.geometry import Point, LineString, Polygon
import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt
from hecrasio.core import ResultsZip
from io import BytesIO
#import boto3
import rasterio
from rasterio.plot import show
import pathlib as pl
import os
import shutil
from collections import ChainMap
import json
from datetime import datetime
from windrose import WindroseAxes


# Add additional keys as needed
GEOMETRY_ATTRIBUTES = '/Geometry/2D Flow Areas/Attributes'
GEOMETRY_2DFLOW_AREA = '/Geometry/2D Flow Areas'

HECRAS_VERSION = '/Results/Summary'
PLAN_DATA = '/Plan Data'
EVENT_DATA_BC = '/Event Conditions/Unsteady/Boundary Conditions'
EVENT_2D_MET_DATA_BC= '/Event Conditions/Meteorology'

UNSTEADY_SUMMARY = '/Results/Unsteady/Summary'
TSERIES_RESULTS_2DFLOW_AREA = '/Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/2D Flow Areas'


class PFRAError:
    """
    Generic Error Class for PFRA
    """

    def __init__(self, error):
        self.Error = error


class HDFResultsFile:
    """
    HEC-RAS HDF Plan File Object to compute flow data at breaklines.
    Some functionality may be useful for other ras objects.
    """

    def __init__(self, model:ResultsZip, model_path:str, hdfResults_path:str):

        self.__model = model
        if '.zip' in model_path:
            self.__zip_path = hdfResults_path
        else:
            self.__path = hdfResults_path

        def decoder():
            """
            Decode bytes objects from hdf file
            :return:
            """
            if isinstance(x, bytes):
                return x.decode()
            else:
                return x

        def local_hdf():
            """
            Add Description
            :return:
            """
            try:
                self.__model.zipfile.extract(self.__zip_path)
                return h5py.File(self.__zip_path, 'r')
            except:
                return h5py.File(self.__path, 'r')

        def get_2dFlowArea_data():
            """
            Add Description
            :return:
            """
            table_data = self._hdfLocal[GEOMETRY_ATTRIBUTES]
            names = table_data.dtype.names
            domain_data = {}
            # Use [1:-1] to pull the name from the 0 element (row[0])
            for row in table_data:
                domain_data[row[0].decode()] = list(row)[1:-1]
            return pd.DataFrame(domain_data, index=names[1:-1])

        def get_planData(table):
            """
            Add Description
            :param table:
            :return:
            """
            table_data = self._hdfLocal['{}/{}'.format(PLAN_DATA, table)].attrs
            values = [table_data[n] for n in list(table_data.keys())]
            # Add wrapper here?
            values = [v[0] if isinstance(v, list) else v for v in values]
            values = [v.decode() if isinstance(v, bytes) else v for v in values]
            return pd.DataFrame(data=values, index=list(table_data.keys()), columns=['Results'])
        
        def get_geometry_data(table, domain):
            """Read in data from results tables"""
            data = '{}/{}/{}'.format(GEOMETRY_2DFLOW_AREA, domain, table)
            return np.array(self._plan_data[data])

        def get_perimeter(domain):
            """Creates a perimeter polygon from points"""
            d_array = get_geometry_data('Perimeter', domain)
            aoi = Polygon([tuple(p) for p in d_array])
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries(aoi))
        
        def get_domain_geometries():
            domains = self._domains
            if len(domains) > 1:
                poly_list = [get_perimeter(domain) for domain in domains]
                df = pd.concat(poly_list).reset_index(level=0, drop=True)
                return gpd.GeoDataFrame(df)
            else:
                print('Single domain found...')
                pass

        def get_2dSummary():
            """Add Description"""
            try:
                if self.Version.startswith('5'):
                    table_data = self._hdfLocal[UNSTEADY_SUMMARY].attrs
                    values = [table_data[n] for n in list(table_data.keys())]
                    values = [v.decode() if isinstance(v, bytes) else v for v in values]
                    values = [str(v) if isinstance(v, list) else v for v in values]
                    return pd.DataFrame(data=values, index=list(table_data.keys()), columns=['Results'])
                if self.Version.startswith('6'):
                    table_data = self._hdfLocal[UNSTEADY_SUMMARY].attrs
                    values = [table_data[n] for n in list(table_data.keys())]
                    values = [v.decode() if isinstance(v, bytes) else v for v in values]
                    values = [str(v) if isinstance(v, list) else v for v in values]
                    table_1 = pd.DataFrame(data=values, index=list(table_data.keys()), columns=['Results'])
                    table_data = self._hdfLocal[UNSTEADY_SUMMARY]['Volume Accounting'].attrs
                    values = [table_data[n] for n in list(table_data.keys())]
                    values = [v.decode() if isinstance(v, bytes) else v for v in values]
                    values = [str(v) if isinstance(v, list) else v for v in values]
                    table_2 = pd.DataFrame(data=values, index=list(table_data.keys()), columns=['Results'])
                    return pd.concat([table_1,table_2])
            except KeyError as e:
                print('You do not seem to have a summary table...')
                print('Exiting.')

        def get_2d_BC_Summary(table):
            """Add Description"""
            try:
                table_data = self._hdfLocal['{}/{}'.format(EVENT_2D_MET_DATA_BC, table)].attrs
                values = [table_data[n] for n in list(table_data.keys())]
                values = [v.decode() if isinstance(v, bytes) else v for v in values]
                values = [str(v) if isinstance(v, list) else v for v in values]
                return pd.DataFrame(data=values, index=list(table_data.keys()), columns=['Results'])
            except KeyError as e:
                print('You do not seem to have a {} 2D BC summary table...'.format(table))
                print('Exiting.')

        def get_version():
            '''Returns the version of HEC-RAS'''
            version = self._hdfLocal.attrs['File Version'].decode('UTF-8').split()[1]
            return version
        
        def get_vol_acct_error_percent():
            try:
                if self.Version.startswith('6'):
                    vol_error_path = "Volume Accounting"
                    return self._hdfLocal[UNSTEADY_SUMMARY][vol_error_path].attrs['Error Percent']
                elif self.Version.startswith('5'):
                    return self._summary["Results"]["Vol Accounting Error Percentage"]
            except KeyError as e:
                    print("Unable to grab volume accounting error")

        self._hdfLocal = local_hdf()
        self._plan_data = self._hdfLocal
        self._Plan_Information = get_planData('Plan Information')
        self._Plan_Parameters = get_planData('Plan Parameters')
        self._2dFlowArea = get_2dFlowArea_data()
        
        self._name = os.path.basename(self.__path)
        self._version = get_version()
        self._domains = self._2dFlowArea.columns.tolist()
        self._domain_polys = get_domain_geometries()
        self._summary = get_2dSummary()
        self._vol_acct_error = get_vol_acct_error_percent()

        try:
            self._Precip2dSummary = get_2d_BC_Summary('Precipitation')
        except KeyError as e:
            self._Precip2dSummary = None

        try:
            self._Wind2dSummary = get_2d_BC_Summary('Wind')
        except KeyError as e:
            #print(e)
            self._Wind2dSummary = None 

    # Getter functions
    @property
    def hdfLocal(self):
        """Add Description"""
        return self._hdfLocal

    @property
    def domains(self):
        """Add Description"""
        return self._domains
    
    @property
    def domain_polys(self):
        """Domain Polygons"""
        return self._domain_polys

    @property
    def Plan_Information(self):
        """Add Description"""
        return self._Plan_Information

    @property
    def Plan_Parameters(self):
        """Add Description"""
        return self._Plan_Parameters

    @property
    def summary(self):
        """Add Description"""
        return self._summary

    @property
    def get_2dFlowArea(self):
        """Add Description"""
        return self._2dFlowArea

    @property
    def vol_acct_error(self):
        return self._vol_acct_error

    @property
    def Name(self):
        """Add Description"""
        return self._name

    @property
    def Version(self):
        """Add Description"""
        return self._version

    @property
    def Precip2dSummary(self):
        """Flow boundary conditions"""
        return self._Precip2dSummary    
    
    @property
    def Wind2dSummary(self):
        """Flow boundary conditions"""
        return self._Wind2dSummary
    
class DomainResults:
    """
    HEC-RAS HDF Plan File Object to compute flow data at breaklines.
    Some functionality may be useful for other ras objects.
    """

    def __init__(self, model: ResultsZip, plan: HDFResultsFile, domain: str):
        # Specify Domain to instantiate Object
        self.__model = model
        self._plan = plan
        self._domain = domain
        self._plan_data = self._plan.hdfLocal    

        def get_domain_cell_size():
            """Identifies mean cell size for a domain"""
            flowData = self._plan.get_2dFlowArea.copy()
            flowData = flowData[self._domain]
            xspacing = flowData.loc['Spacing dx']
            yspacing = flowData.loc['Spacing dy']
            return np.mean([xspacing, yspacing])

        def get_tseries_results(table):
            """Read in data from results tables as a Pandas DataFrame"""
            try:
                data = '{}/{}/{}'.format(TSERIES_RESULTS_2DFLOW_AREA, self._domain, table)
                d_array = np.array(self._plan_data[data]).T
                return pd.DataFrame(d_array)
            except:
                print('{} is missing from the HDF!'.format(table))

        def get_tseries_forcing(table):
            """This table is not domain specific"""
            group = list(self._plan_data['{}/{}'.format(EVENT_DATA_BC, table)])
            table_data = {}
            for g in group:
                table_data[g] = np.array(self._plan_data['{}/{}/{}'.format(EVENT_DATA_BC, table, g)])
            return table_data

        def get_2d_tseries_forcing(table):
            """Reads in 2D boundary forcing data, not domain specific"""
            data_location = '{}/{}'.format(EVENT_2D_MET_DATA_BC, table)

            pdf = pd.DataFrame()

            group = list(self._plan_data[data_location])
            index_start =  group.index('Timestamp')
            for i, g in enumerate(group[index_start:]):
                data = self._plan_data[data_location ][g][:]
                pdf.insert(i, "{}_{}".format(table,g),list(data))
            
            pdf['Dates'] =pd.to_datetime(pdf.iloc[:,0].apply(lambda x: x.decode("utf-8")))
            pdf['Time_Step']=((pdf['Dates']-pdf['Dates'].shift(1)).apply(lambda x: x.total_seconds()/60/60/24)).fillna(0)
            pdf['Days'] = pdf['Time_Step'].cumsum()

            return pdf

        def get_geometry_data(table):
            """Read in data from results tables"""
            data = '{}/{}/{}'.format(GEOMETRY_2DFLOW_AREA, self._domain, table)
            return np.array(self._plan_data[data])

        def get_perimeter():
            """Creates a perimeter polygon from points"""
            d_array = get_geometry_data('Perimeter')
            aoi = Polygon([tuple(p) for p in d_array])
            return gpd.GeoDataFrame(geometry=gpd.GeoSeries(aoi))

        def get_face():
            """Returns GeoDataFrame with Faces per pair of Face Indices"""
            gdf = gpd.GeoDataFrame(self._Faces_FacePoint_Indexes, columns=['from_idx', 'to_idx'])
            gdf['face'] = gdf.apply(lambda row:
                                    LineString([self._Face_FacePoints_Coordinate[row['from_idx']],
                                                self._Face_FacePoints_Coordinate[row['to_idx']]]),
                                    axis=1)
            #gdf['geometry'] = gdf['face']
            gdf['face_copy'] = gdf['face']
            gdf = gdf.set_geometry('face_copy')
            gdf = gdf.rename_geometry('geometry')
            gdf = gdf.drop(['from_idx', 'to_idx', 'face'], axis=1)
            return gdf

        def get_centroids():
            """Returns GeoDataFrame with Face centroids per pair of Face Indices"""
            gdf = get_face()
            gdf['face_cnt'] = gdf.apply(lambda row: row.geometry.centroid, axis=1)
            gdf['geometry'] = gdf['face_cnt']
            gdf = gdf.drop(['face_cnt'], axis=1)
            return gdf

        def describe_depth():
            """Calculate max, min, and range of depths for each cell center"""
            # Pull in cell centroids and attribute them
            cc_array = self._Cells_Center_Coordinate
            cc_gdf = gpd.GeoDataFrame([Point([coord[0], coord[1]]) for coord in cc_array], columns=['geometry'])
            depth_array = self._Depth

            # Attribute cell centroids with depths
            # NOT USED?
            # cc_attr = pd.concat([cc_gdf, depth_array], axis=1)

            # Obtain descriptive statistics for each centroid
            max_attr = pd.DataFrame(depth_array.max(axis=1), columns=['max'])
            max_gdf = pd.concat([cc_gdf, max_attr], axis=1)
            max_gdf_nonzero = max_gdf[max_gdf['max'] != 0]

            min_attr = pd.DataFrame(depth_array.min(axis=1), columns=['min'])
            min_gdf = pd.concat([cc_gdf, min_attr], axis=1)
            min_gdf_nonzero = min_gdf[min_gdf['min'] != 0]
            return max_gdf_nonzero, min_gdf_nonzero

        def get_percent_cells_never_wet():
            """Calculates the percent of cells that never get wet during an event."""
            depth_array  = self._Depth
            depth_max = np.nan_to_num(depth_array.max(axis=1))
            num_cells = depth_max.shape[0]
            nvw = depth_max[depth_max<=0].shape[0]
            nvw/num_cells
            return round(nvw/num_cells*100.0,2)   

        def get_avg_depth():
            """Calculates average depth at faces returning an array."""
            depth_list = []
            for (c1_idx, c2_idx) in self._Faces_Cell_Indexes:
                # cat_depths = np.stack([self._Depth.loc[c1_idx], self._Depth.loc[c2_idx]])
                cat_depths = np.stack([self._Depth[c1_idx, :], self._Depth[c2_idx, :]])
                avg_face = np.average(cat_depths, axis=0)
                depth_list.append(np.around(avg_face, decimals=2))
                # np.stack use default axis=0
            return pd.DataFrame(np.stack(depth_list))

        def get_extreme_edge_depths():
            """Identifies Face Centroids with absolute, avgerage depths greater-than one foot"""
            # Obtain boundary line
            boundary_line = list(self._Perimeter['geometry'])[0].boundary
            # Identify external faces
            df = pd.DataFrame()
            df['exterior'] = self._Faces.geometry.apply(lambda lstring: lstring.intersects(boundary_line))

            # Identify minima
            attr = pd.DataFrame(abs(self._Avg_Face_Depth).max(axis=1), columns=['abs_max'])
            face_dp = pd.concat([self._Face_Centroid_Coordinates, attr], axis=1)
            exterior_faces = face_dp[df['exterior'] == True]
            return exterior_faces[exterior_faces['abs_max'] > 1]

        def get_extreme_edge_depths():
            """Identifies Face Centroids with absolute, avgerage depths greater-than one foot"""
            # Obtain boundary line
            boundary_line = list(self._Perimeter['geometry'])[0].boundary

            # Identify external faces
            df = pd.DataFrame()
            perimeter = gpd.GeoDataFrame(gpd.GeoSeries(boundary_line).to_frame(), geometry=0)
            intersections = gpd.sjoin(perimeter, self._Faces, how="inner", predicate='intersects')

            # Identify minima
            attr = pd.DataFrame(abs(self._Avg_Face_Depth).max(axis=1), columns=['abs_max'])
            face_dp = pd.concat([self._Face_Centroid_Coordinates, attr], axis=1)
            exterior_faces = face_dp.loc[intersections['index_right']]
            return exterior_faces[exterior_faces['abs_max'] > 1]

        try:
            self._StageBC = get_tseries_forcing('Stage Hydrographs')
        except KeyError as e:
            self._StageBC = None

        try:
            self._FlowBC = get_tseries_forcing('Flow Hydrographs')
        except KeyError as e:
            #print(e)
            self._FlowBC = None

        try:
            self._PrecipBC = get_tseries_forcing('Precipitation Hydrographs')
        except KeyError as e:
            #print(e)
            self._PrecipBC = None

        try:
            self._Wind2dBC = get_2d_tseries_forcing('Wind')
        except KeyError as e:
            #print(e)
            self._Wind2dBC = None

        try:
            self._Precip2dBC = get_2d_tseries_forcing('Precipitation')
        except KeyError as e:
            #print(e)
            self._Precip2dBC = None

        self._CellSize = get_domain_cell_size()
        self._Faces_FacePoint_Indexes = get_geometry_data('Faces FacePoint Indexes')
        self._Face_FacePoints_Coordinate = get_geometry_data('FacePoints Coordinate')
        self._Faces_Cell_Indexes = get_geometry_data('Faces Cell Indexes')
        self._Face_Velocity = abs(get_tseries_results('Face Velocity'))
        self._Face_Centroid_Coordinates = get_centroids()
        self._Cells_Center_Coordinate = get_geometry_data('Cells Center Coordinate')
        self._water_surface = np.array(get_tseries_results('Water Surface'))
        self._elevation = np.array(get_geometry_data('Cells Minimum Elevation'))
        self._Depth = np.subtract(self._water_surface, self._elevation.reshape(self._elevation.shape[0], -1))
        self._Describe_Depths = describe_depth()
        self._Percent_Cells_Never_Wet = get_percent_cells_never_wet()
        self._Avg_Face_Depth = get_avg_depth()
        self._Perimeter = get_perimeter()
        self._Faces = get_face()
        self._Extreme_Edges = get_extreme_edge_depths()

    @property
    def Percent_Cells_Never_Wet(self):
        """Domain percent of cells that never get wet during the event"""
        print('Domain ID: {}, Percent Cells that are Never Wet = {}'.format(self._domain, self._Percent_Cells_Never_Wet))
        return self._Percent_Cells_Never_Wet

    @property
    def CellSize(self):
        """Domain mean cell size"""
        print('Domain ID: {}, Average Cell Size = {}'.format(self._domain, self._CellSize))
        return self._CellSize

    @property
    def StageBC(self):
        """Stage boundary conditions"""
        return self._StageBC

    @property
    def FlowBC(self):
        """Flow boundary conditions"""
        return self._FlowBC

    @property
    def PrecipBC(self):
        """Precipitation boundary conditions"""
        return self._PrecipBC

    @property
    def Precip2dBC(self):
        """Precipitation boundary conditions"""
        return self._Precip2dBC

    @property
    def Wind2dBC(self):
        """Precipitation boundary conditions"""
        return self._Wind2dBC

    @property
    def Faces_FacePoint_Indexes(self):
        """Indices of face points used to create each Face"""
        return self._Faces_FacePoint_Indexes

    @property
    def Face_FacePoints_Coordinate(self):
        """Coordinates of face points"""
        return self._Face_FacePoints_Coordinate

    @property
    def Cells_Center_Coordinate(self):
        """Coordinates of cell centers"""
        return self._Cells_Center_Coordinate

    @property
    def Faces(self):
        """Faces created from face point indecies and coordinates"""
        return self._Faces

    @property
    def Face_Centroid_Coordinates(self):
        """Centroid of faces"""
        return self._Face_Centroid_Coordinates

    @property
    def Faces_Cell_Indexes(self):
        """Indecies of cells bounded by each face"""
        return self._Faces_Cell_Indexes

    @property
    def Face_Velocity(self):
        """Velocity measurements at each face"""
        return self._Face_Velocity

    @property
    def Depth(self):
        """Depth measurements at each cell center"""
        return self._Depth

    @property
    def Describe_Depths(self):
        """Max, min, and range of depths for each cell center"""
        return self._Describe_Depths

    @property
    def Avg_Face_Depth(self):
        """Average depth of cell centers bounding a face"""
        return self._Avg_Face_Depth

    @property
    def Perimeter(self):
        """Domain area polygon"""
        return self._Perimeter

    @property
    def Extreme_Edges(self):
        """Perimeter face centroids with absolute, average depths greater than one"""
        return self._Extreme_Edges

    def find_anomalous_attributes(self, attr: str = 'Face_Velocity', threshold: int = 30):
        """
        Returns attributed points with the maximum of their attributes exceeding a threshold
        :param attr:
        :param threshold:
        :return:
        """
        max_attr = pd.DataFrame(getattr(self, attr).max(axis=1), columns=['max'])
        df_thresh = max_attr[max_attr['max'] > threshold]
        gdf_thresh = self.Face_Centroid_Coordinates.iloc[df_thresh.index]
        try:
            return pd.concat([gdf_thresh, df_thresh], axis=1)
        except ValueError as e:
            print('No Anomolous Data Found')
            return None

    def count_anomalous_attributes(self, attr: str = 'Face_Velocity', threshold: int = 30):
        """
        Returns attributed points with a count of their attributes exceeding a threshold
        :param attr:
        :param threshold:
        :return:
        """
        dseries = getattr(self, attr).apply(lambda row: sum(row > threshold), axis=1)
        non_nan = dseries[dseries != 0].dropna()
        df_non_nan = pd.DataFrame(non_nan, columns=['count'])
        gdf_thresh = self.Face_Centroid_Coordinates.iloc[df_non_nan.index]
        try:
            return pd.concat([gdf_thresh, df_non_nan], axis=1)
        except ValueError as e:
            print('No Anomolous Data Found')
            return None




# Functions ---------------------------------------------------------------------

def all_aoi_gdf(domain_results:list) -> gpd.geodataframe.GeoDataFrame:
    """
    Creates a geodataframe containing polygons for all domains.
    :param domain_results:
    """
    perimeters = [domain.Perimeter for domain in domain_results]
    df = pd.concat(perimeters).reset_index(drop=True)
    return gpd.GeoDataFrame(df)

def group_excessive_points(gdf: gpd.geodataframe.GeoDataFrame, cell_size: float):
    """
    Creates groupings of collocated points exceeding a threshold.
        By default, a grouping is defined as three times the average
        cell size of the input file.
    :param gdf:
    :param cell_size:
    :return:
    """
    gdf_aois = gpd.GeoDataFrame()
    gdf_aois['point'] = gdf.geometry
    gdf_aois['polygon'] = gdf_aois.point.apply(lambda row: row.buffer(cell_size * 3))
    #gdf_aois['geometry'] = gdf_aois['polygon']
    gdf_aois['polygon_copy'] = gdf_aois['polygon']
    gdf = gdf.set_geometry('polygon_copy')
    gdf = gdf.rename_geometry('geometry')
    
    try:
        diss_aois = list(unary_union(gdf_aois.geometry))
        gdf_diss_aois = gpd.GeoDataFrame(diss_aois, columns=['geometry'])
    except:
        diss_aois = unary_union(gdf_aois.geometry)
        gdf_diss_aois = gpd.GeoDataFrame([diss_aois], columns=['geometry'])
    return gdf_diss_aois


def subset_data(grouping_polys: gpd.geodataframe.GeoDataFrame, thresheld_gdf: gpd.geodataframe.GeoDataFrame,
                count_gdf: gpd.geodataframe.GeoDataFrame, face_gdf: gpd.geodataframe.GeoDataFrame,
                buff_distance: int = 100) -> [list, list, list]:
    """
    Creates three lists of dataframes subset by a polygon where the polygon
        is a grouping of centroids. The first list contains maximum values for
        each face centroid, the second list contains counts of instances above
        a threshold, and the third lists faces within the buffered bounding
        box of a group of centroids.
        
    :param grouping_polys:
    :param thresheld_gdf:
    :param count_gdf:
    :param face_gdf:
    :param buff_distance:
    :return:
    """
    subset_max_list, subset_count_list, subset_face_list = [], [], []
    for i, poly in enumerate(grouping_polys.geometry):
        subset_max = thresheld_gdf[thresheld_gdf.within(poly)]
        subset_max_list.append(subset_max)

        # NOT USED?
        # subset_count = count_gdf.loc[subset_max.index]
        subset_count_list.append(count_gdf.loc[subset_max.index])

        x0, y0, x1, y1 = poly.buffer(buff_distance).bounds
        bbox = Polygon([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        subset_faces = face_gdf[face_gdf.within(bbox)]
        subset_face_list.append(subset_faces)
    return subset_max_list, subset_count_list, subset_face_list


def find_large_and_small_groups(count_list: list, max_list: list, face_list: list,
                                gdf_groups: gpd.geodataframe.GeoDataFrame,
                                min_count: int = 5) -> [dict, dict]:
    """
    Identifies large groupings, i.e. above minimum count, of points and
        small groupings. Returns two dictionaries. One with large idxs,
        maximums, counts, faces, and groups as well as one with small idxs,
        maximums, and counts.

    :param count_list:
    :param max_list:
    :param face_list:
    :param gdf_groups:
    :param min_count:
    :return:
    """
    large_dict, small_dict = {}, {}

    large_tuples = [(i, count) for i, count in enumerate(count_list) if len(count) > min_count]
    large_dict['idxs'] = [large_tuple[0] for large_tuple in large_tuples]
    large_dict['maxes'] = [max_list[i] for i in large_dict['idxs']]
    large_dict['counts'] = [large_tuple[1] for large_tuple in large_tuples]
    large_dict['faces'] = [face_list[i] for i in large_dict['idxs']]
    large_dict['groups'] = [gdf_groups.iloc[i] for i in large_dict['idxs']]

    small_tuples = [(i, count) for i, count in enumerate(count_list) if len(count) <= min_count]
    small_dict['idxs'] = [small_tuple[0] for small_tuple in small_tuples]
    small_dict['maxes'] = [max_list[i] for i in small_dict['idxs']]
    small_dict['counts'] = [small_tuple[1] for small_tuple in small_tuples]
    return large_dict, small_dict

def velCheckMain(results, domain, plot_tseries=5):
    """
    Add Description
    :param results:
    :param plot_tseries:
    :param domain:
    """
    # Identify face velocities above a given threshold
    df_thresh = results.find_anomalous_attributes()
    df_count = results.count_anomalous_attributes()

    if df_count.shape[0] > 1 and df_thresh.shape[0] > 1:

        # Identify groups of excessive centroids
        gdf_groups = group_excessive_points(df_thresh, results.CellSize)

        # Using a method nearly doubles the time
        max_list, count_list, face_list = subset_data(gdf_groups, df_thresh, df_count, results.Faces)

        # Split groups into large (n > 5) clusters vs. everything else
        l_dict, s_dict = find_large_and_small_groups(count_list, max_list, face_list, gdf_groups)

        # Identify group of interest
        for idx in range(len(l_dict['groups'])):
            plot_instabilities(l_dict['maxes'], l_dict['counts'], l_dict['faces'], results.Perimeter,
                               l_dict['groups'], idx)

            # NOT USED?
            maxes = l_dict['maxes'][idx]
            # counts = l_dict['counts'][idx]
            # faces = l_dict['faces'][idx]
            # group = l_dict['groups'][idx]

            max_vFaceIDs = list(maxes.sort_values(by='max', ascending=False)[0:plot_tseries].index)

            # NOT USED?
            # groupID = idx
            depths = results.Avg_Face_Depth.iloc[max_vFaceIDs]
            velocities = results.Face_Velocity.iloc[max_vFaceIDs]

            for i in depths.index:
                DepthVelPlot(depths.loc[i], velocities.loc[i], i)
        try:
            plot_disparate_instabilities(s_dict['maxes'], s_dict['counts'], results.Perimeter, domain)
        except:
            print('No disparate instabilities found. All instabilities must be grouped!')
        return pd.DataFrame(data=[len(pd.concat(count_list)), max(pd.concat(max_list)['max'])],
                            columns=['Results'],
                            index=['Instability Count', 'Max Velocity'])
    else:
        max_vel = results.Face_Velocity.values.max()
        return pd.DataFrame(data=[0, max_vel],
                            columns=['Results'],
                            index=['Instability Count', 'Max Velocity'])
        print('No Velocity Errors Found in Domain {}'.format(domain))

# Plotting Functions ------------------------------------------------------------

def show_results(domains:list, model, rasPlan, plot_tseries:int=3) -> None:
    """Wrapper function plotting descriptive statistics, extreme edges, boundary
    conditions and velocity values.
    """
    if len(domains) > 1:
        results = {domain: DomainResults(model, rasPlan, domain) for domain in domains}
        results_table = {}
        result_depth = pd.DataFrame()
        for domain, result in results.items():
            depth_array = result.Depth
            depth_max = np.nan_to_num(depth_array.max(axis=1))
            depth_min = np.nan_to_num(depth_array.min(axis=1))
            num_cells = depth_max.shape[0]

            nvw_min_len = depth_min[depth_min<=0].shape[0]
            nvw_max_len = depth_max[depth_max<=0].shape[0]
            nvw_min_per = round(nvw_min_len/num_cells*100.0, 2)
            nvw_max_per = round(nvw_max_len/num_cells*100.0, 2)

            index_depth_min = ['Cells Not Wet (at min. depth), %', 'Avg. Min. Depth (wet cells)', 
                   'Median Min. Depth (wet cells)', 'Avg. Min. Depth', 
                   'Median Min. Depth' ]
            index_depth_max = ['Cells Not Wet (at max. depth), %', 'Avg. Max. Depth (wet cells)', 
                   'Median Max. Depth (wet cells)', 'Avg. Max. Depth', 'Median Max. Depth' ]
            values_depth_min = [nvw_min_per, depth_min[depth_min>0].mean(), 
                    np.median(depth_min[depth_min>0]), depth_min.mean(), np.median(depth_min) ]
            values_depth_max = [nvw_max_per, depth_max[depth_max>0].mean(), 
                    np.median(depth_max[depth_max>0]), depth_max.mean(), np.median(depth_max) ]

            table_depth = pd.DataFrame(data = values_depth_min + values_depth_max , 
                            index = index_depth_min +index_depth_max, 
                            columns=['Results_{}'.format(domain)] )  
            result.Percent_Cells_Never_Wet
            plot_descriptive_stats(result.Describe_Depths, result.Perimeter, domain)
            plot_extreme_edges(result.Extreme_Edges, result.Perimeter, mini_map=rasPlan.domain_polys)
            plotBCs(result, domain) 
            results_table[domain] = velCheckMain(result, domain, plot_tseries)
            result_depth = pd.concat([result_depth,table_depth], axis=1 )
        instability_count = sum([value.loc['Instability Count'] for value in list(results_table.values())])[0]
        max_velocity = max([value.loc['Max Velocity'].values[0] for value in list(results_table.values())])
        return result_depth, pd.DataFrame(data=[instability_count, max_velocity],
                            columns=['Results'],
                            index=['Instability Count', 'Max Velocity'])

    else:
        domain = domains[0]
        result = DomainResults(model, rasPlan, domain)
        
        depth_array = result.Depth
        depth_max = np.nan_to_num(depth_array.max(axis=1))
        depth_min = np.nan_to_num(depth_array.min(axis=1))
        num_cells = depth_max.shape[0]

        nvw_min_len = depth_min[depth_min<=0].shape[0]
        nvw_max_len = depth_max[depth_max<=0].shape[0]
        nvw_min_per = round(nvw_min_len/num_cells*100.0, 2)
        nvw_max_per = round(nvw_max_len/num_cells*100.0, 2)

        index_depth_min = ['Percent Dry Cells at Min. Flood Extent', 'Avg. Depth at Min. Flood Extent (wet cells)', 
                   'Median Depth at Min. Flood Extent (wet cells)', 'Avg. Depth at Min. Flood Extent', 
                   'Median Depth at Min. Flood Extent' ]
        index_depth_max = ['Percent Dry Cells at Max. Flood Extent', 'Avg. Depth at Max. Flood Extent (wet cells)', 
                   'Median Depth at Max. Flood Extent (wet cells)', 'Avg. Depth at Max. Flood Extent', 'Median Depth at Max. Flood Extent' ]
        values_depth_min = [nvw_min_per, depth_min[depth_min>0].mean(), 
                    np.median(depth_min[depth_min>0]), depth_min.mean(), np.median(depth_min) ]
        values_depth_max = [nvw_max_per, depth_max[depth_max>0].mean(), 
                    np.median(depth_max[depth_max>0]), depth_max.mean(), np.median(depth_max) ]

        table_depth = pd.DataFrame(data = values_depth_min + values_depth_max , 
                            index = index_depth_min +index_depth_max, 
                            columns=['Results_{}'.format(domain)] )  
        result.Percent_Cells_Never_Wet
        plot_descriptive_stats(result.Describe_Depths, result.Perimeter, domain)
        plot_extreme_edges(result.Extreme_Edges, result.Perimeter)
        plotBCs(result, domain)
        return table_depth, velCheckMain(result, domain, plot_tseries)

def plot_instabilities(max_list, count_list, gdf_face, gdf_face_all, ex_groups, idx):
    """
    Add Description
    :param max_list:
    :param count_list:
    :param gdf_face:
    :param gdf_face_all:
    :param ex_groups:
    :param idx:
    """
    fig, _ = plt.subplots(2, 2, figsize=(20, 8))
    x0, y0, x1, y1 = ex_groups[idx].geometry.buffer(100).bounds

    # Plot Max Velocities
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    max_list[idx].plot(column='max', cmap='viridis', legend=True, ax=ax1)
    gdf_face[idx].plot(alpha=0.1, color='black', ax=ax1)
    ax1.set_title('Maximum Velocity recorded at Cell Face (ft/s)')
    ax1.set_xlim(x0, x1)
    ax1.set_ylim(y0, y1)

    # Plot Number of instabilities recorded (timesteps above threshold)
    ax2 = plt.subplot2grid((2, 2), (1, 0))
    ax2 = count_list[idx].plot(column='count', cmap='viridis', legend=True, ax=ax2)
    ax2 = gdf_face[idx].plot(alpha=0.1, color='black', ax=ax2)
    ax2.set_title('Number of Instabilities recorded at Cell Face (n)')
    ax2.set_xlim(x0, x1)
    ax2.set_ylim(y0, y1)

    # Plot Map Key (domain)
    ax3 = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    gdf_face_all.plot(alpha=0.05, color='black', ax=ax3)
    pnt_group = gpd.GeoDataFrame(geometry=gpd.GeoSeries(ex_groups[idx].geometry.buffer(1000)))
    pnt_group.plot(alpha=0.5, color='Red', legend=False, ax=ax3)
    ax3.set_title('Map Legend')

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')
    fig.suptitle('Group {}'.format(idx + 1), fontsize=16, fontweight='bold')


def plot_disparate_instabilities(max_list, count_list, bounding_polygon, domain):
    """
    Add Description
    :param max_list:
    :param count_list:
    :param bounding_polygon:
    :param domain:
    """
    small_maxes = pd.concat(max_list)
    small_counts = pd.concat(count_list)

    fig, _ = plt.subplots(1, 2, figsize=(20, 8))

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    small_maxes.plot(column='max', cmap='viridis', legend=True, ax=ax1)
    bounding_polygon.plot(alpha=0.1, color='black', ax=ax1)
    ax1.set_title('Maximum Velocity recorded at Cell Face (ft/s)')

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    ax2 = small_counts.plot(column='count', cmap='viridis', legend=True, ax=ax2)
    ax2 = bounding_polygon.plot(alpha=0.1, color='black', ax=ax2)
    ax2.set_title('Number of Instabilities recorded at Cell Face (n)')

    ax1.axis('off')
    ax2.axis('off')
    fig.suptitle('Isolated Points above Threshold for Domain {}'.format(domain), fontsize=16, fontweight='bold')


def plot_descriptive_stats(stat_lists: tuple, aoi: gpd.geodataframe.GeoDataFrame, domain:str) -> None:
    """
    Plots the descriptive statistics (Max, Min) for
        cell centers with the area of interest underneath.
    :param stat_lists:
    :param aoi:
    """
    maximums, minimums = stat_lists

    # Plot descriptive statistics
    fig, (ax_string) = plt.subplots(1, 2, figsize=(20, 8))

    ax1 = plt.subplot2grid((1, 2), (0, 0))
    aoi.plot(color='k', alpha=0.25, ax=ax1)
    maximums.plot(column='max', cmap='viridis', markersize=0.1, legend=True, ax=ax1)
    ax1.set_title('Maximum Depth (ft)')

    ax2 = plt.subplot2grid((1, 2), (0, 1))
    aoi.plot(color='k', alpha=0.25, ax=ax2)
    ax2 = minimums.plot(column='min', cmap='viridis', markersize=0.1, legend=True, ax=ax2, s=1)
    ax2.set_title('Minimum Depth (ft)')

    ax1.axis('off')
    ax2.axis('off')
    fig.suptitle('Depths at Cell Centers of Domain {}'.format(domain),
                 fontsize=16, fontweight='bold')

def speed(x1, x2):
    return np.sqrt(x1**2+x2**2)

def direction(x1, x2):
    return (270-np.arctan2(x1,x2)*180/np.pi+360)%360

def divide(x1, x2):
    return np.divide(x1, x2, out=np.zeros_like(x1), where=x2!=0)

def degToCompass(num):
    val=int((num/22.5)+.5)
    arr=["N","NNE","NE","ENE","E","ESE", "SE", "SSE","S","SSW","SW","WSW","W","WNW","NW","NNW"]
    return arr[(val % 16)]

def remove_zeros(x1, x2):
    return np.take(x1, np.where((x1!=0) | (x2 != 0))[0])

def remove_unrealistic(x1, x2, unreal_value):
    return np.take(x1, np.where(((x1 <= unreal_value) & (x1 >= -unreal_value)) & ((x2 <= unreal_value) & (x2 >= -unreal_value)))[0])

def plot_2d_forcing(model, rasPlan, domains):
    results = DomainResults(model, rasPlan, domains[0])
    data_wind = []
    data_precip = []
    index_wind = []
    index_precip = []
    if results.Precip2dBC is not None:
        pdf = results.Precip2dBC
        pdf["Precipitation > 0"]=pdf["Precipitation_Values"].map(lambda x: x[x !=0])
        pdf['Extent %']=pdf["Precipitation > 0"].apply(lambda x: x.shape[0])/pdf["Precipitation_Values"].apply(lambda x: x.shape[0])*100
        pdf["Precipitation > 0"]=pdf["Precipitation > 0"].apply(lambda x: x if x.shape[0]>0 else [0])
        pdf['Median']=pdf["Precipitation > 0"].apply(np.median)
        pdf['q95']=pdf["Precipitation > 0"].apply( lambda x: np.quantile(x,.95))
        pdf['q05']=pdf["Precipitation > 0"].apply( lambda x: np.quantile(x,.05))
        
        max_precip_intensity = pdf['Precipitation_Values'].apply(lambda x: np.amax(x)).max()
        max_precip_extent = round(pdf['Extent %'].max(),2)
        total_precip = sum(pdf['Precipitation_Values'].tolist())
        mean_total_precip = np.mean(total_precip)
        var_total_precip = np.var(total_precip)

        fig = plt.figure(figsize=(15, 8))

        ax1 = plt.subplot2grid((2, 2), (0, 0),  colspan=3)
        plt.subplots_adjust(hspace=0.3)
        pdf.plot('Days',y=['q05', 'q95'],color=['blue', 'blue'], ax=ax1, linewidth=.1 ,zorder=2,legend=True)
        pdf.plot('Days',y=['Median'],color=['darkblue'], ax=ax1, linewidth=.8 ,zorder=2,legend=True)
        plt.fill_between(pdf['Days'],pdf['q05'], pdf['q95'], interpolate=True, color='grey', alpha=0.5)
        ax1.set_title('{}\nPeak Precipitation Intensity of {:.2f} in/hr'.format(domains[0], max_precip_intensity))
        ax1.set_ylabel('Preipitation, in./hr.',fontsize=12)
        ax1.grid()

        ax2 = plt.subplot2grid((2, 2), (1, 1))
        pdf.plot('Days',y=['Extent %'], ax=ax2)
        ax2.set_title('{}\nPeak Aerial Extent of Precipitation {} %'.format(domains[0], max_precip_extent))
        ax2.set_ylabel('Precipitaton Aerial Extent, %',fontsize=12)
        ax2.set_ylim([-5, 105])
        ax2.grid()

        ax3 = plt.subplot2grid((2, 2), (1, 0))
        ax3.hist(total_precip,density=1, bins=60,edgecolor='black', linewidth=1.2, color = "skyblue")
        ax3.set_title('{}\n Point Precipitation Total: Mean of {:.2f} in. and Variance of {:.2f} in.'.format(domains[0], mean_total_precip, var_total_precip))
        ax3.set_xlabel('Total Precipitation Depth, in.')
        ax3.set_ylabel('Probability')

        fig.suptitle('Precipitation {}'.format(domains[0]),
                 fontsize=16, fontweight='bold')
        fig.show()

        data_precip = ['inches', max_precip_intensity, max_precip_extent, mean_total_precip, var_total_precip]
        index_precip = ['Precipitation Units', 'Max. Precip. Intensity, in./hr.', 'Max. Precip. Extent, %', 'Avg. Total Point Precip.', 'Var. Total Point Precip.']

    if results.Wind2dBC is not None:
        pdf = results.Wind2dBC
        unrealistic_wind_vel = 500
        #https://stackoverflow.com/questions/28457149/how-to-map-a-function-using-multiple-columns-in-pandas
        
        pdf["Wind_VY_Valid"]= list(map(remove_unrealistic, pdf["Wind_VX"], pdf["Wind_VY"], [unrealistic_wind_vel]*pdf["Wind_VY"].shape[0])) #pdf["Wind_VY"].map(lambda x: x[ (x <= unrealistic_wind_vel) & (x >= -unrealistic_wind_vel)]) 
        pdf["Wind_VX_Valid"]= list(map(remove_unrealistic, pdf["Wind_VY"], pdf["Wind_VX"], [unrealistic_wind_vel]*pdf["Wind_VY"].shape[0])) #pdf["Wind_VX"].map(lambda x: x[ (x <= unrealistic_wind_vel) & (x >= -unrealistic_wind_vel)])
        pdf['% Valid']=pdf["Wind_VX_Valid"].apply(lambda x: x.shape[0])/pdf["Wind_VX"].apply(lambda x: x.shape[0])*100
        pdf["Wind_VY_Valid_nonzero"]= list(map(remove_zeros, pdf["Wind_VX_Valid"], pdf["Wind_VY_Valid"])) 
        pdf["Wind_VX_Valid_nonzero"]= list(map(remove_zeros, pdf["Wind_VY_Valid"], pdf["Wind_VX_Valid"])) 
        pdf['% Nonzero']=pdf["Wind_VX_Valid_nonzero"].apply(lambda x: x.shape[0])/pdf["Wind_VX_Valid"].apply(lambda x: x.shape[0])*100
        pdf['Wind_Speed']=list(map(speed, pdf["Wind_VY_Valid_nonzero"], pdf["Wind_VX_Valid_nonzero"]))
        pdf['Wind_Direction']=list(map(direction, pdf["Wind_VY_Valid_nonzero"], pdf["Wind_VX_Valid_nonzero"]))
        sin_list = list(map(divide, pdf['Wind_VY_Valid_nonzero'], pdf['Wind_Speed']))
        cos_list = list(map(divide, pdf['Wind_VX_Valid_nonzero'], pdf['Wind_Speed']))
        sinsum = [sin_list[i].sum() for i in  range(len(sin_list))]
        cossum = [cos_list[i].sum() for i in  range(len(cos_list))]
        pdf['Avg_Wind_Direction']=list(map(direction, sinsum, cossum))
        pdf['Median']=pdf["Wind_Speed"].apply(lambda x: x if x.shape[0]>0 else [0]).apply(np.median)
        pdf['q95']=pdf["Wind_Speed"].apply(lambda x: x if x.shape[0]>0 else [0]).apply( lambda x: np.quantile(x,.95))
        pdf['q05']=pdf["Wind_Speed"].apply(lambda x: x if x.shape[0]>0 else [0]).apply( lambda x: np.quantile(x,.05))

        max_wind_speed = pdf['Wind_Speed'].apply(lambda x: x if x.shape[0]>0 else [0]).apply(lambda x: np.amax(x)).max()
        avg_wind_speed =  np.mean(np.concatenate(pdf['Wind_Speed'].values))
        median_wind_speed = np.median(np.concatenate(pdf['Wind_Speed'].values))
        wind_direction_avg =  direction(np.concatenate(sin_list).sum(),np.concatenate(cos_list).sum())
        wind_direction_avg_compass =degToCompass(wind_direction_avg)

        fig = plt.figure(figsize=(15, 4))

        ax1 = plt.subplot2grid((1, 2), (0, 0),  colspan=3)
        plt.subplots_adjust(hspace=1)
        pdf.plot('Days',y=['q05', 'q95'],color=['blue', 'blue'], ax=ax1, linewidth=.1 ,zorder=2,legend=True)
        pdf.plot('Days',y=['Median'],color=['darkblue'], ax=ax1, linewidth=.8 ,zorder=2,legend=True)
        plt.fill_between(pdf['Days'],pdf['q05'], pdf['q95'], interpolate=True, color='grey', alpha=0.5)
        ax1.set_title('{}\nPeak Wind Speed of {:.2f} ft./sec.'.format(domains[0], max_wind_speed))
        ax1.set_ylabel('Wind Speed, ft./sec.',fontsize=12)
        ax1.grid()

        wax=WindroseAxes.from_ax()
        wax.bar(np.concatenate(pdf['Wind_Direction'].values), np.concatenate(pdf['Wind_Speed'].values), normed=True, opening=0.8, edgecolor='white')
        wax.set_legend()

        fig.suptitle('Wind {}'.format(domains[0]),
                 fontsize=16, fontweight='bold',y=1.04)
        fig.show()

        data_wind = ['ft./sec.', max_wind_speed, avg_wind_speed, median_wind_speed, wind_direction_avg, wind_direction_avg_compass]
        index_wind = ['Wind Speed Units', 'Max. Wind Speed', 'Avg. Wind Speed', 'Median Wind Speed', 'Avg. Wind Dir., Deg.', 'Avg. Wind Dir., Compass']

    if len(data_wind+data_precip)>0:
        table = pd.DataFrame(data = data_precip+data_wind, index = index_precip + index_wind, columns=['Results'])
    else:
        table = None
    return table

def export_results(rasPlan, domain, indices_file_path, results_path, file_name):
    os.makedirs(os.path.join(results_path), exist_ok=True) 
    
    fr= rasPlan.hdfLocal
    fw = h5py.File(os.path.join(results_path, file_name+'.hdf'), 'w')
    
    indices = []
    with open(indices_file_path, "r") as f:
        for line in f:
            indices.append(int(line.strip()))

    data_path ='{}/{}/{}'.format(GEOMETRY_2DFLOW_AREA, domain, 'Cells Center Coordinate')
    coords = np.array(fr[data_path])
    coords_selected =np.take(coords , indices, axis = 0)
    fw.create_dataset(data_path, data=coords_selected)
    group_path = '{}/{}/{}'.format(TSERIES_RESULTS_2DFLOW_AREA, domain, 'Max Values')
    group_id = fw.require_group(group_path)
    fr.copy(data_path, group_id)

    data_path = '{}/{}/{}'.format(TSERIES_RESULTS_2DFLOW_AREA, domain, 'Water Surface')
    wse_array = np.array(fr[data_path])
    wse_array_max = wse_array.max(axis=0)
    wse_array_selected =np.take(wse_array, indices, axis = 1)
    fw.create_dataset(data_path, data=wse_array_selected)

    data_path = '{}/{}/{}/{}'.format(TSERIES_RESULTS_2DFLOW_AREA, domain, 'Max Values','Water Surface')
    fw.create_dataset(data_path, data=wse_array_max)

    data_path = '{}/{}/{}'.format(TSERIES_RESULTS_2DFLOW_AREA, domain, 'Time Step')
    group_path = fr[data_path].parent.name
    group_id = fw.require_group(group_path)
    fr.copy(data_path, group_id)

    fw.close()

    return None

def plot_extreme_edges(gdf: gpd.geodataframe.GeoDataFrame,
                       aoi: gpd.geodataframe.GeoDataFrame,
                       **kwargs) -> None:
    """
    Plots extreme depths along edges along with an overview map showing current
    plotted domain versus all other domains.
    :param gdf:
    :param aoi:
    :param \**kwargs:
        See below
    
    :Keyword Arguments:
        * *mini_map* (gpd.geodataframe.GeoDataFrame) -- Multiple domain perimeters.
    """
    if 'mini_map' in kwargs.keys():
        mini_map = list(kwargs.values())[0]
        
        fig, (ax_string) = plt.subplots(1, 2, figsize=(20, 8))
        ax1 = plt.subplot2grid((1, 2), (0, 0))
        aoi.plot(color='k', alpha=0.25, ax=ax1)
        gdf.plot(column='abs_max', cmap='viridis', legend=True, ax=ax1, markersize=16)
        ax1.set_title('Cell Locations with Depths > 1 ft\n(Check for Ponding)'.format(len(gdf)),
                     fontsize=12, fontweight='bold')
        ax1.axis('off')

        ax2 = plt.subplot2grid((1, 2), (0, 1))
        mini_map.plot(color='#BFBFBF', edgecolor='k', ax=ax2, markersize=16)
        aoi.plot(color='#FFC0CB', edgecolor='k', ax=ax2)
        ax2.set_title('Current domain (pink) compared to all domains (grey)'.format(len(gdf)),
                     fontsize=12, fontweight='bold')
        ax2.axis('off')
    else:
        fig, ax = plt.subplots(figsize = (7,7))
        aoi.plot(color='k', alpha=0.25, ax=ax)
        gdf.plot(column='abs_max', cmap='viridis', legend=True, ax=ax, markersize=16)
        ax.set_title('Cell Locations with Depths > 1 ft\n(Check for Ponding)'.format(len(gdf)),
                     fontsize=12, fontweight='bold')
        ax.axis('off')


def DepthVelPlot(depths: pd.Series, velocities: pd.Series, groupID: int, velThreshold: int = 30):
    """
    Add Description
    :param depths:
    :param velocities:
    :param groupID:
    :param velThreshold:
    """
    t = depths.index
    data1 = depths
    data2 = velocities

    fig, ax1 = plt.subplots(figsize=(10, 2))
    fig.suptitle('Velocity Anomalies at face {}'.format(groupID), fontsize=12, fontweight='bold', x=0.49, y=1.1)

    color = 'blue'
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Depth (ft)', color=color)
    ax1.plot(data1, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'red'
    ax2.set_ylabel('Velocity (ft/s)', color=color)  # we already handled the x-label with ax1
    ax2.plot(data2, color=color, alpha=0.5)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.hlines(velThreshold, t.min(), t.max(), colors='k', linestyles='--', alpha=0.5, label='Threshold')
    ax2.hlines(velThreshold * -1, t.min(), t.max(), colors='k', linestyles='--', alpha=0.5, label='Threshold')

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def plotBCs(results, domain:str):
    """
    Add Description
    """
    if results.FlowBC is not None:
        for k, v in results.FlowBC.items():
            if domain in k:
                fig, ax = plt.subplots(figsize=(20, 2))
                ax.set_title('{}\nPeak Flow of {} cfs'.format(k, int(v[:, 1].max())))
                ax.set_ylabel('Flow (cfs)')
                ax.set_xlabel('Days')
                ax.plot(v[:, 0], v[:, 1])
                ax.grid()

    if results.StageBC is not None:
        for k, v in results.StageBC.items():
            if domain in k:
                fig, ax = plt.subplots(figsize=(20, 2))
                ax.set_title(k)
                ax.set_ylabel('Stage (ft)')
                ax.set_xlabel('Days')
                ax.plot(v[:, 0], v[:, 1])
                ax.grid()

    if results.PrecipBC is not None:
        for k, v in results.PrecipBC.items():
            if domain in k:
                fig, ax = plt.subplots(figsize=(20, 2))
                ax.set_title(k)
                ax.set_ylabel('Precipitation (inches)')
                ax.set_xlabel('Days')
                ax.plot(v[:, 0], v[:, 1])
                ax.grid()

def identify_unique_values(result_table:pd.core.frame.DataFrame,
                           desired_columns:list) -> pd.core.frame.DataFrame:
    """
    Identifies unique values within a results table for a given attribute.
    """
    df = pd.DataFrame(columns=['Unique_Values'])
    df['Result_Attribute'] = pd.Index(desired_columns)
    df.set_index('Result_Attribute', drop=True, inplace=True)
    
    for i in df.index:
        dtype = type(result_table[i][0])
        if dtype is int or dtype is float:
            df.loc[i]['Unique_Values'] = list(np.unique(result_table[i]))
        elif dtype is str:
            df.loc[i]['Unique_Values'] = list(set(result_table[i]))
        elif dtype is list:
            series_list = [elem[0] for elem in result_table[i]]
            list_dtype = type(series_list[0])
            if list_dtype is int or list_dtype is float:
                df.loc[i]['Unique_Values'] = list(np.unique(series_list))
            elif list_dtype is str:
                df.loc[i]['Unique_Values'] = list(set(series_list))
            else:
                print("Dtype {} is not currently supported for variable {}".format(dtype, i))
        else:
            print("Dtype {} is not currently supported for variable {}".format(dtype, i))
    return df

def validate_by_threshold(pd_df, attr, value_list, threshold, results_table_df):
    """Validate the results table raising warnings if any values reported in the data frame are above a given
    value. Report which notebooks are above that value.
    """
    pd_df.loc[attr]['Warnings'] = 'WARNING' if any([value > threshold for value in value_list]) else 'PASS'
    pd_df.loc[attr]['Offending_Nbs'] = [results_table_df.index[i] for i, value in enumerate(list(results_table_df[attr])) if value > threshold]
    return pd_df

def make_qaqc_table(books:list) -> pd.core.frame.DataFrame:
    """Takes a list of tuples representing notebook scraps and creates
    a Pandas DataFrame showing results.
    """
    results_dict = {}
    for tup in books:
        nb, results = tup
        scrap_data = []
        for scrap in results.scraps:
            if scrap == 'Global Errors':
                errors = results.scraps['Global Errors'].data
                print("WARNING! {} had the following global errors: {}".format(nb, errors))
            elif scrap != 'Global Errors':
                logged_information = list(json.loads(scrap).values())[0]
                scrap_data.append(logged_information)
            else:
                print('Unexpected scrap name identified, please troubleshoot!')
        results_dict[nb] = dict(ChainMap(*scrap_data))
    df = pd.DataFrame.from_dict(results_dict).T
    drop_nbs = list(df[df['1D Cores'] != df['1D Cores'].isnull()].index)
    print('WARNING! The following Notebooks had null 1D Cores values. '+
          'These notebooks were dropped from the QA/QC table with the'+
          ' assumption these notebooks are bad: {}'.format(drop_nbs))
    return df[df['1D Cores'] == df['1D Cores'].isnull()]

def fancy_report(nbs:list, values:list, units:str) -> None:
    print("The following notebooks have alarming values for this attribute\n")
    print("{0: <30} {1} ({2})".format('Notebook', 'Value', units))
    print("-"*79)
    for i in range(len(nbs)):
        print("{0: <30} {1}".format(nbs[i], values[i]))

def report_header(variable:str):
    print("\nNow evaluating: {}\n".format(variable))
    
def create_summary_table(df:pd.core.frame.DataFrame, results_df:pd.core.frame.DataFrame):
    """Evaluates the unique dataframe for values which fail thresholds and string matches"""
    for i in df.index:
        if i == 'Vol Accounting Error':
            report_header(i)
            nbs = results_df[abs(results_df[i]) > 0][i].index
            values = results_df[abs(results_df[i]) > 0][i].values
            fancy_report(nbs, values, 'none')
            print("-"*79)
        elif i == 'Solution':
            report_header(i)
            nbs = results_df[results_df[i] != 'Unsteady Finished Successfully'][i].index
            values = results_df[results_df[i] != 'Unsteady Finished Successfully'][i].values
            if len(nbs) < 1:
                print("No errors found.\n\nMoving along...")
            else:
                fancy_report(nbs, values, 'none')
            print("-"*79)
        elif i == 'Instability Count':
            report_header(i)
            nbs = results_df[results_df[i] > 0 ][i].index
            values = results_df[results_df[i] > 0][i].values
            if len(nbs) < 1:
                print("No errors found.\n\nMoving along...")
            else:
                fancy_report(nbs, values, 'n')
            print("-"*79)
        elif i == 'Max Velocity':
            report_header(i)
            nbs = results_df[results_df[i] > 0 ][i].index
            values = results_df[results_df[i] > 0][i].values
            if len(nbs) < 1:
                print("No errors found.\n\nMoving along...")
            else:
                fancy_report(nbs, values, 'ft/s')
            print("-"*79)

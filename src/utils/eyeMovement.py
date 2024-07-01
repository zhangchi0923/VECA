"""
 author: zhangchi
 email: zcsjtu0923@gmail.com
 create date: 2022-11-22 11:48:52
 modify date: 2022-11-22 11:48:52
"""


import pandas as pd
import numpy as np

from math import pi
from config.settings import *

class DataUtils():
    def __init__(self, s):
        self._s = s

    def prepare_data(self) -> pd.DataFrame:
        df = self._s.copy()
        df['timestamp'] = df['timestamp'].astype('uint64', errors='ignore')
        df['level'] = df['level'].astype(int, errors='ignore')
        df['state'] = df['state'].astype(int, errors='ignore')
        df['pos_x'] = df['pos_x'].astype(float, errors='ignore')
        df['pos_y'] = df['pos_y'].astype(float, errors='ignore')
        df['left' ] = df['left' ].astype(float, errors='ignore')
        df['right'] = df['right'].astype(float, errors='ignore')
        start_time = df['timestamp'].min()
        df['timestamp'] = df['timestamp'] - start_time
        na_idx = df.isna().sum(axis=1) == 0
        df = df[na_idx]
        return df
    
    def get_lvl_state(self, df: pd.DataFrame, level: int, state: int):
        idx1 = df['level'] == level
        idx2 = df['state'] == state
        idx = idx1 & idx2
        tmpDf = df[idx].copy()
        x, y, time = np.array(tmpDf['pos_x']), np.array(tmpDf['pos_y']), np.array(tmpDf['timestamp'])
        return x, y, time

class EyeMovement():

    def __init__(self, x, y, time_list, AOI, ps) -> None:
        self._x = x
        self._y = y
        self._time_list = time_list
        self._aoi = AOI
        self._bazier_points = np.array(ps)

    

    def eye_movements_detector(self, x, y, time, maxvel=100):
        """
        :param x: numpy array of pos_x
        :param y: numpy array of pos_y
        :param time: numpy array of timestamp
        :param maxvel: criteria between fixation and saccade
        :return: dict of event list,
                e.g. the value of 'sac' is a list of lists which contains
                start[0] & end[1] time of each sacade, duration[2], start[3] & end[4] coordinates of saccade point
        """
        fix_list = []
        sac_list = []
        res_dict = {'fix':[], 'sac':[], 'gap':[]}
        x_diff = np.diff(x)
        y_diff = np.diff(y)
        time_int = np.diff(time)
        dist = (x_diff**2 + y_diff**2)**0.5
        ang_diff = np.arctan(dist / 5.9)*180/pi
        vel = np.divide(ang_diff, time_int)         # degree / ms
        threshold_vel = vel.copy()
        # print(threshold_vel)

        for i in range(len(threshold_vel)):
            if threshold_vel[i] < maxvel/1000:
                if i == 0:
                    fix_list.append(time[i])
                elif threshold_vel[i-1] < maxvel/1000:
                    if i < len(threshold_vel)-1:
                        fix_list.append(time[i])
                    else:
                        fix_list.append(time[i])
                        fix_x = round(np.mean(x[list(time).index(fix_list[0]) : list(time).index(fix_list[-1]) + 1]), 4)
                        fix_y = round(np.mean(y[list(time).index(fix_list[0]) : list(time).index(fix_list[-1]) + 1]), 4)
                        res_dict['fix'].append([fix_list[0], fix_list[-1], fix_list[-1] - fix_list[0], (fix_x, fix_y)])
                        fix_list = []
                elif threshold_vel[i-1] >= maxvel/1000:
                    if i < len(threshold_vel)-1:
                        fix_list.append(time[i])
                    else:
                        fix_list.extend([time[i], time[-1]])
                        fix_x = round(np.mean(x[list(time).index(fix_list[0]) : list(time).index(fix_list[-1]) + 1]), 4)
                        fix_y = round(np.mean(y[list(time).index(fix_list[0]) : list(time).index(fix_list[-1]) + 1]), 4)
                        res_dict['fix'].append([fix_list[0], fix_list[-1], fix_list[-1] - fix_list[0], (fix_x, fix_y)])
                        fix_list = []
                    sac_list.append(time[i])
                    sac_sx, sac_sy = x[list(time).index(sac_list[0])], y[list(time).index(sac_list[0])]
                    sac_ex, sac_ey = x[list(time).index(sac_list[-1])], y[list(time).index(sac_list[-1])]
                    res_dict['sac'].append([sac_list[0], sac_list[-1], sac_list[-1] - sac_list[0], (sac_sx, sac_sy), (sac_ex, sac_ey)])
                    sac_list = []
            else:
                if i == 0:
                    sac_list.append(time[i])
                elif threshold_vel[i-1] >= maxvel/1000:
                    if i < len(threshold_vel)-1:
                        sac_list.append(time[i])
                    else:
                        sac_list.append(time[i])
                        sac_sx, sac_sy = x[list(time).index(sac_list[0])], y[list(time).index(sac_list[0])]
                        sac_ex, sac_ey = x[list(time).index(sac_list[-1])], y[list(time).index(sac_list[-1])]
                        res_dict['sac'].append([sac_list[0], sac_list[-1], sac_list[-1] - sac_list[0], (sac_sx, sac_sy), (sac_ex, sac_ey)])
                        sac_list = []
                else:
                    if i < len(threshold_vel)-1:
                        sac_list.append(time[i])
                    else:
                        sac_list.extend([time[i], time[-1]])
                        sac_sx, sac_sy = x[list(time).index(sac_list[0])], y[list(time).index(sac_list[0])]
                        sac_ex, sac_ey = x[list(time).index(sac_list[-1])], y[list(time).index(sac_list[-1])]
                        res_dict['sac'].append([sac_list[0], sac_list[-1], sac_list[-1] - sac_list[0], (sac_sx, sac_sy), (sac_ex, sac_ey)])
                        sac_list = []
                    fix_list.append(time[i])
                    fix_x = round(np.mean(x[list(time).index(fix_list[0]) : list(time).index(fix_list[-1]) + 1]), 4)
                    fix_y = round(np.mean(y[list(time).index(fix_list[0]) : list(time).index(fix_list[-1]) + 1]), 4)
                    res_dict['fix'].append([fix_list[0], fix_list[-1], fix_list[-1] - fix_list[0], (fix_x, fix_y)])
                    fix_list = []
        return res_dict

    def merge_fixation(self, detected_res: dict, min_dur=60, max_ang_diff=0.5, max_time_interval=75):
        """

        :param detected_res:
        :param min_dur:
        :param max_ang_diff:
        :param max_time_interval:
        :return: fix_num before[0] and after[1] merging, combined fixation list[2]
        """
        fix_list = detected_res['fix']
        prev_num = len(fix_list)
        prev_fix = fix_list[0]
        new_fix_list = [prev_fix]

        for fix in fix_list[1:]:
            x, y = fix[-1]
            prev_x, prev_y = prev_fix[-1]
            dist = ((x-prev_x)**2 + (y-prev_y)**2)**0.5
            ang_diff = np.arctan(dist / 5.9)*180/pi
            time_interval = fix[0] - prev_fix[1]
            if ang_diff <= max_ang_diff and time_interval <= max_time_interval:
                fix_x, fix_y = round((x + prev_x)/2, 4), round((y +prev_y)/2, 4)                    # other approaches
                new_fix = [prev_fix[0], fix[1], fix[1]-prev_fix[0], (fix_x, fix_y)]
                new_fix_list.pop()
                new_fix_list.append(new_fix)
                prev_fix = new_fix
            else:
                new_fix_list.append(fix)
                prev_fix = fix

        res_list = [fix for fix in new_fix_list if fix[2] >= min_dur]
        return prev_num, len(res_list), res_list
    
    def _extend_AOI(self, percent=0.1):
        aoi = self._aoi
        X0 = min(aoi[0][0],aoi[1][0])
        X1 = max(aoi[0][0],aoi[1][0])
        Y0 = min(aoi[0][1],aoi[1][1]) 
        Y1 = max(aoi[0][1],aoi[1][1])
        width = X1-X0
        height = Y1-Y0
        margin_x = percent*width
        margin_y = percent*height
        X0 = X0 - margin_x
        X1 = X1 + margin_x
        Y0 = Y0 - margin_y
        Y1 = Y1 + margin_y
        return [(X0,Y0),(X1,Y1)]

    def if_in_aoi(self, fix_data) -> bool:
        aoi = self._extend_AOI()
        fx, fy = fix_data[-1][0], fix_data[-1][1]
        X0,X1 = min(aoi[0][0], aoi[1][0]), max(aoi[0][0], aoi[1][0])
        Y0,Y1 = min(aoi[0][1], aoi[1][1]), max(aoi[0][1], aoi[1][1])
        # print(fx, X0, X1, fy, Y0, Y1)
        if (fx-X0)*(fx-X1)<0 and (fy-Y0)*(fy-Y1)<0:
            return True
        else:
            return False


    def AOI_fixations(self, fix_data):
        '''
        List of merged fixations located in ROI.
        :param fix_data:merged_fixations - list of lists which contain start_time and end_time, durations and mean fixation center
        return list of merged fixations located in AOI.
        '''
        assert len(fix_data) > 0, 'Null fixation data.'
        fix_in_aoi = [x for x in fix_data if self.if_in_aoi(x)]
        return fix_in_aoi
    
    
    def nAOI_fixations(self, fix_data):
        '''
        List of merged fixations not located in ROI.
        :param fix_data:merged_fixations - list of lists which contain start_time and end_time, durations and mean fixation center
        return list of merged fixations not located in AOI.
        '''
        assert len(fix_data) > 0, 'Null fixation data.'
        fix_not_in_aoi = [x for x in fix_data if not self.if_in_aoi(x)]
        return fix_not_in_aoi

    def AOI_fixation_ratio(self, fix_data):
        '''
        Calculate ratio of duration of AOI fixations over total fixation duration.
        '''
        fix_in_aoi = self.AOI_fixations(fix_data)
        dur_in_aoi = sum([x[2] for x in fix_in_aoi])
        dur_total_fix = sum([x[2] for x in fix_data])
        return dur_in_aoi / dur_total_fix
    

    def _bezier_order3(self, t, points):
        p0 = points[0,:]
        p1 = points[1,:]
        p2 = points[2,:]
        p3 = points[3,:]
        y = (1-t)**3*p0+3*(1-t)**2*t*p1+3*(1-t)*t**2*p2+t**3*p3
        return y

    def _get_live_position(self, trueTime, start_time):                                                      
        rel_time = (trueTime-start_time)/2.5                                       # time parameter of bezier curves
        i = int(rel_time)                                                          # for detecting current section in 4 bezier curves
        t = rel_time-i                                                             # time parameter of bezier curves
        k = i%4                                                                    # detect current section in 4 bezier curves
        points = self._bazier_points[4*k:4*k+4,:]                                            
        y = self._bezier_order3(t, points)
        return y
    
    def measureFollowRate(self):
        X, Y, time = self._x, self._y, self._time_list
        nObserv = len(time)
        cnt = 0
        eps = 0.33                                                                 # distance to rocket
        start_time = time[0]/1000
        for i in range(nObserv):
            x = X[i]
            y = Y[i]
            t = time[i]/1000
            pos_rocket = self._get_live_position(t, start_time)
            X0 = pos_rocket[0]
            Y0 = pos_rocket[1]
            if (x-X0)**2+(y-Y0)**2 < eps**2:
                cnt += 1
        ratio = cnt/nObserv
        return ratio
    
    def findMinMax(self, a, b):
        if a>b:
            return b,a
        else:
            return a,b
        
    def measureGazeRate(self, subDf, ROI):
        """
        measure the ratio of the gaze time on ROI 

        Parameters
        ----------
        subDf : pandas dataFrame with column names of 'timestamp','pos_x','pos_y'             
        ROI: list of 4 float numbers, Rectangular diagonal vertex coordinates. 
            just as [(x0,y0),(x1,y1)],express a rectangle with 4 corners(x0,y0),(x1,y0),(x1,y1),(x0,y1)
                
        Returns
        -------
        ratio :     float, time ratio of gaze time

        Example
        -------
        >>> measureGazeRate(df,[(5,6),(0,1)])
        0.5
        """
        ROI = self._extend_AOI()                                                               # margin of ROI
        X0,X1 = self.findMinMax(ROI[0][0],ROI[1][0])
        Y0,Y1 = self.findMinMax(ROI[0][1],ROI[1][1])
        nObserv = len(subDf)
        total_Time = subDf['timestamp'].iat[nObserv-1] - subDf['timestamp'].iat[0]
        gaze_time = 0
        for i in range(1,nObserv):
            x = subDf['pos_x'].iat[i]
            y = subDf['pos_y'].iat[i]
            #t = subDf['timestamp'].iat[i]
            if (x-X0)*(x-X1)<0 and (y-Y0)*(y-Y1)<0:
                gaze_time += subDf['timestamp'].iat[i]-subDf['timestamp'].iat[i-1]
        ratio = gaze_time/total_Time
        return ratio
    

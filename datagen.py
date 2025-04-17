import sys
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
import os
import matplotlib.pyplot as plt
from dataset_utils import add_noise, apply_threshold, quantize_manual, apply_offset

sensor_pitch_X = 50 # in um
sensor_pitch_Y = 12.5 # in um
sensor_thickness = 100 #um
pixel_array_sizeX = 21
pixel_array_sizeY = 13                                                          
debug_offcentering = False
noise_mu = 0
noise_sig = 80
charge_threshold = 400

def split(index,df1,df2,df2_uncentered,df3,df3_uncentered,run_offcentering):
        df1.columns = df1.columns.astype(str)
        df2.columns = df2.columns.astype(str)
        df3.columns = df3.columns.astype(str)
        # unflipped, all charge              
        if("unflipped" not in os.listdir()):
            os.mkdir("unflipped")                                               
        df1[df1['z-entry']==100].to_parquet("unflipped/labels_d"+str(index)+".parquet")
        df2[df1['z-entry']==100].to_parquet("unflipped/recon2D_d"+str(index)+".parquet")
        df3[df1['z-entry']==100].to_parquet("unflipped/recon3D_d"+str(index)+".parquet")
        if(run_offcentering):
                df2_uncentered.columns = df2_uncentered.columns.astype(str)
                df3_uncentered.columns = df3_uncentered.columns.astype(str)
                df2_uncentered[df1['z-entry']==100].to_parquet("unflipped/recon2D_uncentered_d"+str(index)+".parquet")
                df3_uncentered[df1['z-entry']==100].to_parquet("unflipped/recon3D_uncentered_d"+str(index)+".parquet")

def parseFile(filein,tag,nevents=-1):
        with open(filein) as f:
                lines = f.readlines()
        header = lines[0].strip()
        #header = lines.pop(0).strip()
        pixelstats = lines[1].strip()
        #pixelstats = lines.pop(0).strip()
        print("Header: ", header)
        print("Pixelstats: ", pixelstats)
        readyToGetTruth = False
        readyToGetTimeSlice = False
        clusterctr = 0
        cluster_truth =[]
        timeslice = 0
        cur_slice = []
        cur_cluster = []
        events = []
        for line in lines:
                ## Start of the cluster
                if "<cluster>" in line:
                        readyToGetTruth = True
                        readyToGetTimeSlice = False
                        clusterctr += 1
                        # Create an empty cluster
                        cur_cluster = []
                        timeslice = 0
                        # move to next line
                        continue
                # the line after cluster is the truth
                if readyToGetTruth:
                        cluster_truth.append(line.strip().split())
                        readyToGetTruth = False
                        # move to next line
                        continue
                ## Put cluster information into np array
                if "time slice" in line:
                        readyToGetTimeSlice = True
                        cur_slice = []
                        timeslice += 1
                        # move to next line
                        continue
                if readyToGetTimeSlice:
                        cur_row = line.strip().split()
                        cur_slice += [float(item) for item in cur_row]
                        # When you have all elements of the 2D image:
                        if len(cur_slice) == pixel_array_sizeX*pixel_array_sizeX:
                                cur_cluster.append(cur_slice)
                        # When you have all time slices:
                        if len(cur_cluster) == 20:
                                events.append(cur_cluster)
                                readyToGetTimeSlice = False
        print("Number of clusters = ", len(cluster_truth))
        print("Number of events = ",len(events))
        print("Number of time slices in cluster = ", len(events[0]))
        arr_truth = np.array(cluster_truth)
        arr_events = np.array( events )
        return arr_events, arr_truth

def main():
        index = int(sys.argv[1])
        run_offcentering = sys.argv[2].lower() in ['true', '1', 'y']
        add_noise = sys.argv[3].lower() in ['true', '1', 'y']
        apply_charge_threshold = sys.argv[4].lower() in ['true', '1', 'y']
        tag = "d"+str(index)
        inputdir = "./"
        arr_events, arr_truth = parseFile(filein=inputdir+"pixel_clusters_d"+str(index)+".out",tag=tag)
        #truth quantities - all are dumped to DF                        
        df = pd.DataFrame(arr_truth, columns = ['x-entry', 'y-entry','z-entry', 'n_x', 'n_y', 'n_z', 'number_eh_pairs', 'y-local', 'pt'])
        cols = df.columns
        for col in cols:
                df[col] = df[col].astype(float)
        if(run_offcentering):
                df['offset1'] = np.random.randint(-5, 6, size=len(df))
                df['offset2'] = np.random.randint(-9, 10, size=len(df))
                #df['offset1'] = np.random.randint(-3, 4, size=len(df))
                #df['offset2'] = np.random.randint(-7, 8, size=len(df))
                df['y-entry'] = df['y-entry'] + df['offset1']*sensor_pitch_Y
                df['x-entry'] = df['x-entry'] + df['offset2']*sensor_pitch_X
        df['cotAlpha'] = df['n_x']/df['n_z']
        df['cotBeta'] = df['n_y']/df['n_z']
        df['y-midplane'] = df['y-entry'] + df['cotBeta']*(sensor_thickness/2 - df['z-entry'])
        df['x-midplane'] = df['x-entry'] + df['cotAlpha']*(sensor_thickness/2 - df['z-entry'])
        print("The shape of the event array: ", arr_events.shape)
        print("The ndim of the event array: ", arr_events.ndim)
        print("The dtype of the event array: ", arr_events.dtype)
        print("The size of the event array: ", arr_events.size)
#        print("The max value in the array is: ", np.amax(arr_events))
        # print("The shape of the truth array: ", arr_truth.shape)
        df2 = {}
        df2list = []
        df2_uncentered = {}
        df2list_uncentered = []
        df3 = {}
        df3list = []
        df3_uncentered = {}
        df3list_uncentered = []
        offset_values = []
        for i, e in enumerate(arr_events):
                if(add_noise):
                        e = add_noise(np.array(e), noise_mu, noise_sig)
                if(apply_charge_threshold):
                        e = apply_threshold(np.array(e), thresh=400)
                # Only last time slice
                df2list.append(np.array(e[-1]).flatten())
                # All time slices
                df3list.append(np.array(e).flatten())
                if(debug_offcentering):
                        print("old block:")
                        for idx, block in enumerate(e.reshape(20, pixel_array_sizeX, pixel_array_sizeX)):
                                print(f'<time slice {idx}>')
                        for row in block:
                                print(' '.join(map(str, row)))
                if(run_offcentering):
                        random_integer = df['offset1'].iloc[i]
                        random_integer2 = df['offset2'].iloc[i]
                        offset = (random_integer, random_integer2) #(4, 5) #4 is up/down, 5 is left/right
                        offset_values.append(offset)
                        new_blocks = [apply_offset(block, offset) for block in e.reshape(20, pixel_array_sizeX, pixel_array_sizeX)]
                        df2list_uncentered.append(np.array(new_blocks[-1]).flatten())
                        df3list_uncentered.append(np.array(new_blocks).flatten())
                if(debug_offcentering):
                        print("\nNew block:")
                        for idx, new_block in enumerate(new_blocks):
                                print(f'<time slice {idx}>')
                        for row in new_block:
                                print(' '.join(map(str, row)))
        df2 = pd.DataFrame(df2list)
        df3 = pd.DataFrame(df3list)
        if(run_offcentering):
                df2_uncentered = pd.DataFrame(df2list_uncentered)
                df3_uncentered = pd.DataFrame(df3list_uncentered)  

        # split into flipped/unflipped, pos/neg charge
        split(index,df,df2,df2_uncentered,df3,df3_uncentered,run_offcentering)
        if(run_offcentering):
                offsets_x = [offset[0] for offset in offset_values]
                offsets_y = [offset[1] for offset in offset_values]
                plt.hist2d(offsets_x, offsets_y, bins=[20, 20], cmap='Blues')
                plt.colorbar(label='Count')
                plt.xlabel('Offset X')
                plt.ylabel('Offset Y')
                plt.title('2D Histogram of Offset Values')
                plt.savefig("unflipped/offset_histogram_d"+str(index)+".png")

if __name__ == "__main__":
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
